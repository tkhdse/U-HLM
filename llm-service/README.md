This is the U-HLM LLM-servce.

To run: 

1. Install dependencies:
```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

2. Run FastAPI server:
```
# run as a module
python3 -m grpc-gateway.server.main

```


3. Test the llama_client:
```
python3 -m grpc-gateway.server.llama_client 
```