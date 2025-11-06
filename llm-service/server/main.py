import asyncio
import grpc
from grpc_reflection.v1alpha import reflection

from . import uhlm_pb2, uhlm_pb2_grpc
from .handler import UHLMService

async def serve():
    server = grpc.aio.server()
    uhlm_pb2_grpc.add_UHLMServicer_to_server(UHLMService(model_id="meta-llama/Llama-2-7b-hf"), server)
    
    SERVICE_NAMES = (
        uhlm_pb2.DESCRIPTOR.services_by_name["UHLM"].full_name,
        reflection.SERVICE_NAME
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port("[::]:8081")
    await server.start()
    print("gRPC server started on [::]:8081")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())