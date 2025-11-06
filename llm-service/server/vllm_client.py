import numpy as np
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroup, SequenceData

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
        
        # Get logits from vLLM engine
        # Note: This is simplified - vLLM's engine API is async but complex
        # You may need to adapt based on vLLM version
        outputs = self.engine.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(temperature=1.0, logprobs=1),
            request_id=request_id
        )

        print(outputs)
        
        # Extract logits and convert to probabilities
        # This depends on vLLM's output format
        logits = outputs[0].logprobs  # Adjust based on actual API
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return probs.astype(np.float32)

# res = requests.get("http://10.42.22.29:8000/healthz")
# print(res)