import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]  # slm-service -> U-HLM
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import threshold_calc
import speculate
from rpc_client import LLMRPCClient
import asyncio
import utils
import argparse
from data_collector import DataCollector

from transformers import AutoTokenizer


# IMPORTANT: SLM and LLM must use compatible tokenizers!
# Current setup: Llama 3 family
# SLM: Llama-3.2-3B-Instruct, LLM: Llama-3.1-8B (or similar Llama 3.x)

# For Q&A with base models, use Q: A: formatting
model, tokenizer = utils.setup("meta-llama/Llama-3.2-3B-Instruct")
# Alternative chat model option: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

async def generate_response(prompt, max_tokens=50, K=20, theta_max=2.0, use_chat_template=False, 
                            simulate_network=False, data_collector=None):
    """Generate a complete response using U-HLM with gRPC LLM verification.
    
    Args:
        data_collector: Optional DataCollector instance for training data collection
    """
    print(f"\nGenerating response for: '{prompt}'")
    print("-" * 60)
    
    # When collecting data, force threshold to 0 so all tokens are transmitted
    if data_collector is not None:
        print("📊 Data collection mode: threshold set to 0.0 (all tokens transmitted)")

    if simulate_network:
        print("⚠️  Network latency simulation enabled (50ms per RPC call)")

    # Format prompt for chat model if enabled and tokenizer has chat template
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Using chat template. Formatted prompt: {repr(formatted_prompt)}")
    else:
        # For base models: format questions as Q: A: for better completion
        if prompt.strip().endswith('?'):
            formatted_prompt = f"Q: {prompt}\nA:"
            print(f"Base model Q&A format: {repr(formatted_prompt)}")
        else:
            formatted_prompt = prompt
            print("Using raw prompt")

    # Tokenize prompt and remove EOS tokens
    current_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    slm_eos_id = tokenizer.eos_token_id
    current_token_ids = [t for t in current_token_ids if t != slm_eos_id]
    
    if not current_token_ids:
        print("Prompt contains only EOS tokens, stopping.")
        return "", {"transmitted": 0, "skipped": 0, "total": 0, "transmission_rate": 0.0}
    
    response_token_ids = []
    transmitted_count = 0
    skipped_count = 0

    async with LLMRPCClient(host="127.0.0.1", port=8081, simulate_latency=simulate_network) as llm:
        # Get session ID and LLM's EOS token ID (use formatted prompt for both SLM and LLM)
        session_id, llm_eos_token_id = await llm.begin_session(formatted_prompt)
        print(f"LLM EOS token ID: {llm_eos_token_id}")
        print(f"SLM EOS token ID: {slm_eos_id}")

        try:
            for step in range(max_tokens):
                # 2. Run the SLM to propose a draft token + distribution
                context_tensor = torch.tensor([current_token_ids]).to(model.device)
                draft = speculate.sample_draft_tokens(
                    model, context_tensor, K=K, theta_max=theta_max, device=model.device
                )

                base_draft_id = draft["base_draft_id"]
                base_probs = draft["base_probs"].detach().cpu().numpy()

                # 3. Measure uncertainty
                sampled_ids = draft["sampled_ids"]
                u_t = sum(d_k != base_draft_id for d_k in sampled_ids) / len(sampled_ids)
                
                # When collecting data, force threshold to 0
                if data_collector is not None:
                    u_th = 0.0
                else:
                    u_th = threshold_calc.get_threshold()

                if u_t > u_th:
                    # 4a. Offload to LLM because the SLM is uncertain
                    transmitted_count += 1
                    verify_result = await llm.verify(
                        session_id=session_id,
                        draft_id=base_draft_id,
                        probs=base_probs,
                    )
                    # Handle both old (3 values) and new (5 values) return formats
                    if len(verify_result) == 5:
                        accepted, final_token_id, _, rejection_prob, y_d_lt_x_d = verify_result
                    else:
                        accepted, final_token_id, _ = verify_result
                        rejection_prob = 0.0
                        y_d_lt_x_d = False
                    
                    # Record data point if collecting
                    if data_collector is not None:
                        data_collector.record_data_point(
                            uncertainty=u_t,
                            rejection_prob=rejection_prob,
                            y_d_lt_x_d=y_d_lt_x_d,
                            session_id=session_id,
                            draft_id=base_draft_id
                        )
                    
                    decision = "TRANSMITTED"
                else:
                    # 4b. Trust the SLM draft directly
                    skipped_count += 1
                    accepted = True
                    final_token_id = base_draft_id
                    decision = "SKIPPED"
                    await llm.sync(session_id, [final_token_id])

                # 5. Check EOS BEFORE appending (use LLM's EOS for transmitted, SLM's for skipped)
                if decision == "TRANSMITTED":
                    # Token came from LLM, use LLM's EOS token ID
                    if int(final_token_id) == int(llm_eos_token_id):
                        print(f"Hit LLM EOS token (id={final_token_id}); stopping generation.")
                        break
                else:
                    # Token came from SLM, use SLM's EOS token ID
                    if int(final_token_id) == int(slm_eos_id):
                        print(f"Hit SLM EOS token (id={final_token_id}); stopping generation.")
                        break

                # 6. Append token (only if not EOS)
                current_token_ids.append(final_token_id)
                response_token_ids.append(final_token_id)

                # 7. Log the choice
                token_text = tokenizer.decode([final_token_id], skip_special_tokens=True).strip()
                if final_token_id == tokenizer.eos_token_id:
                    print(f"[SLM DEBUG] Generated EOS token: final_token_id={final_token_id}, SLM eos_token_id={tokenizer.eos_token_id}")
                display_text = token_text if token_text else f'<EOS or empty, id={final_token_id}>'
                print(
                    f"Token {step+1:>3}: [{decision}] "
                    f"uncertainty={u_t:.3f} vs threshold={u_th:.3f} "
                    f"accepted={accepted} -> '{display_text}'"
                )

        finally:
            # 8. Cleanly close the session
            await llm.end_session(session_id)

    # 9. Decode and report statistics
    decoded = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    total = len(response_token_ids) or 1
    print(f"\nComplete Response:")
    print(decoded if decoded.strip() else "<empty>")
    print(
        f"\nStats: transmitted={transmitted_count}, skipped={skipped_count}, "
        f"total={total}, transmission_rate={(transmitted_count/total)*100:.1f}%"
    )

    return decoded, {
        "transmitted": transmitted_count,
        "skipped": skipped_count,
        "total": total,
        "transmission_rate": transmitted_count / total if total else 0.0,
    }


# Moved argument parsing inside main() to prevent execution on import

def main():
    """Main inference loop"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='U-HLM: Uncertainty-Aware Hybrid Language Model Inference')
    parser.add_argument('--latency', '--simulate-latency', action='store_true',
                        help='Simulate 50ms network latency for RPC calls (default: False)')
    parser.add_argument('--use-chat-template', action='store_true',
                        help='Use chat template formatting for prompts (default: False)')
    parser.add_argument('--collect-data', action='store_true',
                        help='Collect training data for threshold calculation (sets threshold to 0)')
    parser.add_argument('--data-file', type=str, default=None,
                        help='File to save training data (default: slm-service/training_data.jsonl)')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum tokens to generate per prompt (default: 50)')
    args = parser.parse_args()

    data_collector = None
    if args.collect_data:
        data_collector = DataCollector(data_file=args.data_file)
        print(f"📊 Data collection enabled. Data will be saved to: {data_collector.data_file}")
        print(f"   Current data points: {data_collector.get_data_count()}")
        print(f"   Threshold is set to 0.0 (all tokens will be transmitted)")
    
    while True:
        prompt = input("\nEnter prompt (or 'q'/'quit' to exit): ").strip()
        
        if prompt.lower() == 'quit' or prompt.lower() == 'q':
            if data_collector:
                print(f"\n📊 Total data points collected: {data_collector.get_data_count()}")
            break
        
        if not prompt:
            continue
            
        try:
            asyncio.run(generate_response(
                prompt, 
                max_tokens=args.max_tokens,
                use_chat_template=args.use_chat_template, 
                simulate_network=args.latency,
                data_collector=data_collector
            ))
            if data_collector:
                print(f"   Data points so far: {data_collector.get_data_count()}")
        except Exception as e:
            print(f"Error generating response: {e}")
            continue

if __name__ == "__main__":
    main()