import numpy as np
from . import verifier
from .vllm_client import VLLMClient
from .session_manager import SessionManager
from . import uhlm_pb2, uhlm_pb2_grpc

class UHLMService(uhlm_pb2_grpc.UHLMServicer):
    def __init__(self, model_id):
        self.llm = VLLMClient(model_id=model_id)
        self.sessions = SessionManager(tokenizer=self.llm.tokenizer)

    async def BeginSession(self, request, context):
        sid, pos = self.sessions.begin(request.prompt)
        return uhlm_pb2.BeginResp(session_id=sid, position=pos)

    async def VerifyToken(self, request, context):
        # This now gets properly tokenized text
        s = self.sessions.get_text(request.session_id)
        y = await self.llm.logits(s)
        
        if request.HasField("dense"):
            x = np.array(request.dense.probs, dtype=np.float32)
        else:
            x = np.zeros_like(y)
            for idx, p in zip(request.sparse.indices, request.sparse.probs):
                x[idx] = p
        
        accepted, token_id = verifier.accept_or_resample(request.draft_id, x, y)
        # Store actual token ID (integer)
        self.sessions.append(request.session_id, token_id)
        
        return uhlm_pb2.VerifyResp(
            accepted=accepted,
            token_id=token_id,
            new_length=self.sessions.sessions[request.session_id]["length"],
        )

    async def Sync(self, request, context):
        self.sessions.sync_tail(request.session_id, [f"<tok{i}>" for i in request.tail_ids])
        length = self.sessions.sessions[request.session_id]["length"]
        return uhlm_pb2.SyncResp(new_length=length)

    async def EndSession(self, request, context):
        ok = self.sessions.end(request.session_id)
        return uhlm_pb2.EndResp(success=ok)
