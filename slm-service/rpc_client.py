# slm-service/llm_rpc_client.py
import grpc
from common.uhlm import uhlm_pb2, uhlm_pb2_grpc # adjust import path

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
        sparse_indices, sparse_probs = zip(*[(i, p) for i, p in enumerate(probs) if p > 1e-6])
        sparse = uhlm_pb2.SparseTopK(
            indices=sparse_indices,
            probs=sparse_probs,
        )
        resp = await self.stub.VerifyToken(
            uhlm_pb2.VerifyReq(session_id=session_id,
                               draft_id=draft_id,
                               sparse=sparse)
        )
        return resp.accepted, resp.token_id, resp.new_length

    async def end_session(self, session_id):
        await self.stub.EndSession(uhlm_pb2.EndReq(session_id=session_id))