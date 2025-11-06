from server import uhlm_pb2, uhlm_pb2_grpc
import asyncio
import grpc
import os
import numpy as np


async def test_uhlm_service():

    try:
        # Test 1: Begin Session
        print("=" * 60)
        print("Test 1: BeginSession")
        print("=" * 60)
        prompt = "The capital of France is"
        print(f"Prompt: '{prompt}'")
        
        begin_resp = await stub.BeginSession(
            uhlm_pb2.BeginReq(prompt=prompt)
        )
        print(f"✓ Session started!")
        print(f"  Session ID: {begin_resp.session_id}")
        print(f"  Position: {begin_resp.position}")
        session_id = begin_resp.session_id

    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await channel.close()


if __name__ == "__main__":
    print("U-HLM gRPC Service Test Client")
    print("Make sure the server is running on localhost:8081\n")
    asyncio.run(test_uhlm_service())