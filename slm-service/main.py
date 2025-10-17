import speculate
import utils
import threshold_calc
import torch

# setup model
model, tokenizer = utils.setup("meta-llama/Llama-3.2-1B-Instruct")

def generate_response(prompt, max_tokens=50):
    """Generate a complete response using U-HLM inference"""
    print(f"\nGenerating response for: '{prompt}'")
    print("-" * 50)
    
    # Start with the prompt tokens
    current_tokens = tokenizer.encode(prompt)
    response_tokens = []
    transmitted_count = 0
    skipped_count = 0
    
    for i in range(max_tokens):
        # Convert current tokens to tensor for model input
        inputs = torch.tensor([current_tokens]).to(model.device)
        
        # Get next token using SLM
        result = speculate.sample_draft_tokens(model, inputs, K=20, theta_max=2.0, device="cpu")
        sampled_ids = result["sampled_ids"]
        base_draft_id = result["base_draft_id"]
        
        # Calculate uncertainty
        u_t = sum(d_k != base_draft_id for d_k in sampled_ids) / len(sampled_ids)
        u_th = threshold_calc.get_threshold()
        
        # Decide whether to transmit or skip
        if u_t > u_th:
            # High uncertainty - send to LLM
            final_token_id, was_accepted = speculate.adaptive_offload(u_t, prompt, result)
            transmitted_count += 1
            print(f"Token {i+1}: [TRANSMITTED] uncertainty={u_t:.3f} > {u_th:.3f}")
        else:
            # Low uncertainty - use SLM's guess
            final_token_id = base_draft_id
            skipped_count += 1
            token_text = tokenizer.decode([final_token_id])
            print(f"Token {i+1}: [SKIPPED] uncertainty={u_t:.3f} <= {u_th:.3f} -> '{token_text}'")
        
        # Add token to response
        response_tokens.append(final_token_id)
        current_tokens.append(final_token_id)
        
        # Stop if we hit an end token
        if final_token_id == tokenizer.eos_token_id:
            print(f"Hit EOS token, stopping generation")
            break
    
    # Decode the complete response
    full_response = tokenizer.decode(response_tokens)
    print(f"\nComplete Response: {full_response}")
    print(f"Stats: {transmitted_count} transmitted, {skipped_count} skipped, {len(response_tokens)} total tokens")
    print(f"Transmission Rate: {(transmitted_count/len(response_tokens)*100):.1f}%")
    
    return full_response

# Main inference loop
while True:
    prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
    
    if prompt.lower() == 'quit':
        break
    
    if not prompt:
        continue
        
    try:
        generate_response(prompt)
    except Exception as e:
        print(f"Error generating response: {e}")
        continue