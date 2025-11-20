# slm-service/llm_rpc_client.py
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]  # slm-service -> U-HLM
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import grpc
from common.uhlm import uhlm_pb2, uhlm_pb2_grpc # adjust import path
import numpy as np

class LLMRPCClient:
    def __init__(self, host="127.0.0.1", port=8081):
        self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = uhlm_pb2_grpc.UHLMStub(self.channel)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.channel.close()

    async def begin_session(self, prompt):
        resp = await self.stub.BeginSession(uhlm_pb2.BeginReq(prompt=prompt))
        return resp.session_id


    async def verify(self, session_id, draft_id, probs):
        # probs is [vocab_size] array, get top-K actual token IDs
        vocab_size = len(probs)
        # Get top-K token IDs (indices where probs are highest)
        top_k = min(100, vocab_size)  # Limit to top 100
        top_indices = np.argsort(probs)[-top_k:][::-1]  # Descending order
        top_probs = probs[top_indices]
        
        # Normalize probabilities
        top_probs = top_probs / np.sum(top_probs)
        
        sparse = uhlm_pb2.SparseTopK(
            indices=top_indices.astype(np.uint32).tolist(),  # Actual token IDs
            probs=top_probs.astype(np.float32).tolist(),
        )
        resp = await self.stub.VerifyToken(
            uhlm_pb2.VerifyReq(session_id=session_id,
                            draft_id=draft_id,
                            sparse=sparse)
        )
        return resp.accepted, resp.token_id, resp.new_length

    # async def verify(self, session_id, draft_id, probs):
    #     sparse_indices, sparse_probs = zip(*[(i, p) for i, p in enumerate(probs) if p > 1e-6])
    #     sparse = uhlm_pb2.SparseTopK(
    #         indices=sparse_indices,
    #         probs=sparse_probs,
    #     )
    #     resp = await self.stub.VerifyToken(
    #         uhlm_pb2.VerifyReq(session_id=session_id,
    #                            draft_id=draft_id,
    #                            sparse=sparse)
    #     )
    #     return resp.accepted, resp.token_id, resp.new_length

    async def end_session(self, session_id):
        await self.stub.EndSession(uhlm_pb2.EndReq(session_id=session_id))