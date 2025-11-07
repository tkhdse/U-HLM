from server import uhlm_pb2, uhlm_pb2_grpc
import asyncio
import grpc
import numpy as np


async def query(prompt: str, max_tokens: int = 20):
    """Send a prompt, get token IDs back"""
    channel = grpc.aio.insecure_channel("localhost:8081")
    stub = uhlm_pb2_grpc.UHLMStub(channel)
    
    try:
        # Begin
        session_id = (await stub.BeginSession(uhlm_pb2.BeginReq(prompt=prompt))).session_id
        
        # Generate
        tokens = []
        for _ in range(max_tokens):
            resp = await stub.VerifyToken(uhlm_pb2.VerifyReq(
                session_id=session_id,
                draft_id=np.random.randint(0, 32000),
                sparse=uhlm_pb2.SparseTopK(
                    indices=list(range(100, 110)),
                    probs=[0.1] * 10
                )
            ))
            tokens.append(resp.token_id)
            if resp.token_id == 2:  # EOS
                break
        
        await stub.EndSession(uhlm_pb2.EndReq(session_id=session_id))
        return tokens
        
    finally:
        await channel.close()


if __name__ == "__main__":
    prompt = input("Enter your query: ") if len(__import__('sys').argv) == 1 else __import__('sys').argv[1]
    tokens = asyncio.run(query(prompt))
    print(f"\nGenerated tokens: {tokens}")