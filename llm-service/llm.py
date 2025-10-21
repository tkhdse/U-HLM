from fastapi import FastAPI, HTTPException
from vllm import LLM


MODEL_ID = "meta-llama/Llama-2-7b-hf"
# MODEL_ID=""
# GPU_NODES = 0


llm = LLM(
        model=MODEL_ID,
        # "facebook/opt-13b",
        # tensor_parallel_size=GPU_NODES,
        distributed_executor_backend="ray"
    )
output = llm.generate("San Franciso is a")

print(output)