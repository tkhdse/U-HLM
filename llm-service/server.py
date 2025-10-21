import asyncio
import grpc.aio as grpc_aio
import numpy as np
import uuid
from concurrent import futures

# proto compiled files
import verifier_pb2
import verifier_pb2_grpc


async def serve() -> None:
    """Initializes the vLLM engine and starts the asynchronous gRPC server."""
    
    # 1. Initialize vLLM Engine

    # 2. Configure and Start gRPC Server
    # Note: We use a thread pool executor for running synchronous code, but the server itself is async.
    server = grpc_aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port(HOST_ADDR)
    print(f"Starting gRPC server on {HOST_ADDR}...")
    
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    except Exception as e:
        print(f"Server experienced a critical error: {e}")
