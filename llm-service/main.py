import asyncio
import grpc.aio as grpc_aio
import numpy as np
import torch
import uuid
from concurrent import futures

# proto compiled files
import verifier_pb2
import verifier_pb2_grpc

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from typing import AsyncGenerator, List, Tuple

# Config
HOST_ADDR = "0.0.0.0:50051"
MODEL_ID = "meta-llama/Llama-2-7b-hf" 
TOTAL_GPUS = 1 # Set to 1 for testing, or total GPUs in your Ray cluster


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Converts a torch tensor (e.g., logits) into raw bytes for Protobuf."""
    # Ensure tensor is on CPU and is a recognized float type
    # For a vocabulary size of 50k, this array is large, hence the use of 'bytes'
    np_array = tensor.cpu().float().numpy()
    return np_array.tobytes()


class LLMVerifierServicer(verifier_pb2_grpc.LLMVerifierServicer):
    """
    Implements the methods defined in the LLMVerifier service from verifier.proto.
    This class handles the business logic for incoming RPC calls.
    """
    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine

    # This method signature MUST match the RPC definition in verifier.proto
    async def VerifyTokens(
        self, 
        request: verifier_pb2.VerificationRequest, 
        context: grpc_aio.ServicerContext
    ) -> verifier_pb2.VerificationResponse:
        """
        The RPC handler for verifying tokens using the vLLM engine.
        """
        # 1. Prepare Sampling Parameters using data from the Protobuf request
        sampling_params = SamplingParams(
            temperature=request.temperature, 
            logprobs=request.vocab_size # Request full distribution size
        )
        
        # We generate a request ID if none is provided, though the .proto suggests it is present
        request_id = request.request_id if request.request_id else str(uuid.uuid4())

        # 2. Pass request to vLLM Engine
        outputs_generator: AsyncGenerator = self.engine.generate(
            request.prompt, 
            sampling_params, 
            request_id
        )

        try:
            final_output = await outputs_generator.__anext__()
            

            # NOTE: This line accesses the custom 'logits_tensor' you must add to vLLM's output structure
            raw_logits_tensor = final_output.outputs[0].logits_tensor 
            
            # Extract final verified token IDs
            verified_token_ids = final_output.outputs[0].token_ids
            
            # 3. Construct and Return Protobuf Response
            return verifier_pb2.VerificationResponse(
                final_text=final_output.outputs[0].text,
                raw_logits=tensor_to_bytes(raw_logits_tensor), # Convert tensor to binary bytes
                verified_token_ids=verified_token_ids
            )

        except Exception as e:
            print(f"Inference error for request {request_id}: {e}")
            # Set the gRPC status code (Critical for production reliability)
            context.set_code(grpc_aio.StatusCode.INTERNAL)
            context.set_details(f"LLM Inference failed: {str(e)}")
            return verifier_pb2.VerificationResponse() # Return an empty/default response

# --- Server Startup (Replaces Uvicorn Execution) ---

async def serve() -> None:
    """Initializes the vLLM engine and starts the asynchronous gRPC server."""
    
    # 1. Initialize vLLM Engine
    print(f"Initializing vLLM Engine for model {MODEL_ID}...")
    # llm_engine = AsyncLLMEngine.from_engine_args(
    #     model=MODEL_ID,
    #     tensor_parallel_size=TOTAL_GPUS, 
    #     # Using multiprocessing executor for simpler local testing without full Ray setup:
    #     distributed_executor_backend="mp" if TOTAL_GPUS > 1 else "ray"
    # )

    # 2. Configure and Start gRPC Server
    server = grpc_aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # verifier_pb2_grpc.add_LLMVerifierServicer_to_server(
    #     LLMVerifierServicer(llm_engine), server
    # )
    
    server.add_insecure_port(HOST_ADDR)
    print(f"Starting gRPC server on {HOST_ADDR}...")
    
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    # Ensure this script is executed with proper environment variables if using Ray
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    except Exception as e:
        print(f"Server experienced a critical error: {e}")
