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

from transformers import AutoTokenizer


model, tokenizer = utils.setup("TinyLlama/TinyLlama-1.1B-Chat-v1.0") # meta-llama/Llama-3.2-1B-Instruct
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

async def generate_response(prompt, max_tokens=50, K=20, theta_max=2.0):
    """Generate a complete response using U-HLM with gRPC LLM verification."""
    print(f"\nGenerating response for: '{prompt}'")
    print("-" * 60)

    # Format prompt for chat model if tokenizer has chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Using chat template. Formatted prompt: {repr(formatted_prompt)}")
    else:
        formatted_prompt = prompt
        print("No chat template found, using raw prompt")

    # Tokenize prompt and remove EOS tokens
    current_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    slm_eos_id = tokenizer.eos_token_id
    print(slm_eos_id)
    current_token_ids = [t for t in current_token_ids if t != slm_eos_id]
    
    if not current_token_ids:
        print("Prompt contains only EOS tokens, stopping.")
        return "", {"transmitted": 0, "skipped": 0, "total": 0, "transmission_rate": 0.0}
    
    response_token_ids = []
    transmitted_count = 0
    skipped_count = 0

    async with LLMRPCClient(host="127.0.0.1", port=8081) as llm:
        # Get session ID and LLM's EOS token ID (use formatted prompt for both SLM and LLM)
        session_id, llm_eos_token_id = await llm.begin_session(formatted_prompt)
        print(f"LLM EOS token ID: {llm_eos_token_id}")

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
                u_th = threshold_calc.get_threshold()

                if u_t > u_th:
                    # 4a. Offload to LLM because the SLM is uncertain
                    transmitted_count += 1
                    accepted, final_token_id, _ = await llm.verify(
                        session_id=session_id,
                        draft_id=base_draft_id,
                        probs=base_probs,
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


# Main inference loop
while True:
    prompt = input("\nEnter prompt (or 'q'/'quit' to exit): ").strip()
    
    if prompt.lower() == 'quit' or prompt.lower() == 'q':
        break
    
    if not prompt:
        continue
        
    try:
        asyncio.run(generate_response(prompt)) # turned into asyncio call
    except Exception as e:
        print(f"Error generating response: {e}")
        continue