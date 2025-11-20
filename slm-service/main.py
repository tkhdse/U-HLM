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

    # Tokenize prompt once
    current_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_token_ids = []

    transmitted_count = 0
    skipped_count = 0

    async with LLMRPCClient(host="127.0.0.1", port=8081) as llm:
        # 1. Start a session on the LLM service
        session_id = await llm.begin_session(prompt)

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

                # 5. Append token and keep session text in sync
                current_token_ids.append(final_token_id)
                response_token_ids.append(final_token_id)

                # 6. Log the choice
                token_text = tokenizer.decode([final_token_id], skip_special_tokens=True).strip()
                print(
                    f"Token {step+1:>3}: [{decision}] "
                    f"uncertainty={u_t:.3f} vs threshold={u_th:.3f} "
                    f"accepted={accepted} -> '{token_text or '<EOS>'}'"
                )

                if final_token_id == tokenizer.eos_token_id:
                    print("Hit EOS token; stopping generation.")
                    break

        finally:
            # 7. Cleanly close the session
            await llm.end_session(session_id)

    # 8. Decode and report statistics
    decoded = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    total = len(response_token_ids) or 1
    print("\nComplete Response:")
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

# import speculate
# import utils
# import threshold_calc
# import torch

# # setup model
# model, tokenizer = utils.setup("meta-llama/Llama-3.2-1B-Instruct")

# def generate_response(prompt, max_tokens=50):
#     """Generate a complete response using U-HLM inference"""
#     print(f"\nGenerating response for: '{prompt}'")
#     print("-" * 50)
    
#     # Start with the prompt tokens
#     current_tokens = tokenizer.encode(prompt)
#     response_tokens = []
#     transmitted_count = 0
#     skipped_count = 0
    
#     for i in range(max_tokens):
#         # Convert current tokens to tensor for model input
#         inputs = torch.tensor([current_tokens]).to(model.device)
        
#         # Get next token using SLM
#         result = speculate.sample_draft_tokens(model, inputs, K=20, theta_max=2.0, device="cpu")
#         sampled_ids = result["sampled_ids"]
#         base_draft_id = result["base_draft_id"]
        
#         # Calculate uncertainty
#         u_t = sum(d_k != base_draft_id for d_k in sampled_ids) / len(sampled_ids)
#         u_th = threshold_calc.get_threshold()
        
#         # Decide whether to transmit or skip
#         if u_t > u_th:
#             # High uncertainty - send to LLM
#             final_token_id, was_accepted = speculate.adaptive_offload(u_t, prompt, result)
#             transmitted_count += 1
#             print(f"Token {i+1}: [TRANSMITTED] uncertainty={u_t:.3f} > {u_th:.3f}")
#         else:
#             # Low uncertainty - use SLM's guess
#             final_token_id = base_draft_id
#             skipped_count += 1
#             token_text = tokenizer.decode([final_token_id])
#             print(f"Token {i+1}: [SKIPPED] uncertainty={u_t:.3f} <= {u_th:.3f} -> '{token_text}'")
        
#         # Add token to response
#         response_tokens.append(final_token_id)
#         current_tokens.append(final_token_id)
        
#         # Stop if we hit an end token
#         if final_token_id == tokenizer.eos_token_id:
#             print(f"Hit EOS token, stopping generation")
#             break
    
#     # Decode the complete response
#     full_response = tokenizer.decode(response_tokens)
#     print(f"\nComplete Response: {full_response}")
#     print(f"Stats: {transmitted_count} transmitted, {skipped_count} skipped, {len(response_tokens)} total tokens")
#     print(f"Transmission Rate: {(transmitted_count/len(response_tokens)*100):.1f}%")
    
#     return full_response

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