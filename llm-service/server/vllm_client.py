import numpy as np
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
# from vllm.sequence import SequenceGroup, SequenceData

class VLLMClient:
    def __init__(self, model_id="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1):
        # Initialize vLLM engine
        engine_args = EngineArgs(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.engine.tokenizer
        
    async def logits(self, session_text):
        """
        Get logits for next token given session text.
        Returns softmax-normalized probs as numpy array.
        """
        # Tokenize the input
        input_ids = self.tokenizer.encode(session_text)
        
        # Create request
        request_id = f"req_{id(session_text)}"
        prompt_token_ids = input_ids
        
        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=session_text,
                params=SamplingParams(temperature=1.0, logprobs=1)
            )
        except ValueError as e:
            # Handle cases where request validation fails (e.g., prompt too long)
            print(f"Error adding request {request_id}: {e}")
            return None


        final_output = None

        while True:
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.request_id == request_id:
                    final_output = request_output
                    if final_output.finished:
                        break

            if not self.engine.has_unfinished_requests():
                break


        print("output: ", final_output)

        # Extract logits and convert to probabilities
        if final_output and final_output.outputs:
                # Access the custom logits tensor (Requires your custom vLLM build)
                raw_logits = final_output.outputs[0].logits_tensor 
                verified_text = final_output.outputs[0].text

                print(raw_logits)
                print(verified_text)


        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return probs.astype(np.float32)