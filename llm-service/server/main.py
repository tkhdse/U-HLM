import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import asyncio
import grpc
from grpc_reflection.v1alpha import reflection

from common.uhlm import uhlm_pb2, uhlm_pb2_grpc
from .handler import UHLMService
import argparse

async def serve(collect_data=False, data_file=None):
    server = grpc.aio.server()
    service = UHLMService(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        collect_data=collect_data,
        data_file=data_file
    )
    uhlm_pb2_grpc.add_UHLMServicer_to_server(service, server)
    
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
    parser = argparse.ArgumentParser(description='U-HLM LLM Service')
    parser.add_argument('--collect-data', action='store_true',
                        help='Enable data collection for threshold training')
    parser.add_argument('--data-file', type=str, default='llm_data.jsonl',
                        help='File to save collected data (default: llm_data.jsonl)')
    args = parser.parse_args()
    asyncio.run(serve(collect_data=args.collect_data, data_file=args.data_file))