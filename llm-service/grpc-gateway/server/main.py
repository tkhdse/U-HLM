import asyncio
import grpc
from fastapi import FastAPI
from grpc_reflection.v1alpha import reflection

from . import uhlm_pb2, uhlm_pb2_grpc
from .handler import UHLMService

from concurrent import futures
import uvicorn

app = FastAPI(title="UHLM Gateway", version="0.1")

@app.get("/healthz")
def healthz():
    print("hit endpoint")
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return {"sessions": 0, "kv_bytes": 0}

async def serve_grpc():
    server = grpc.aio.server()
    uhlm_pb2_grpc.add_UHLMServicer_to_server(UHLMService(), server)
    SERVICE_NAMES = (uhlm_pb2.DESCRIPTOR.services_by_name["UHLM"].full_name,
                     reflection.SERVICE_NAME)
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:8081")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import threading

    def run_grpc():
        asyncio.run(serve_grpc())

    grpc_thread = threading.Thread(target=run_grpc, daemon=True)
    # loop = asyncio.get_event_loop()
    # loop.create_task(serve_grpc())
    uvicorn.run(app, host="0.0.0.0", port=8000)
