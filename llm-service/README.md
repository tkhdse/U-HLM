# U-HLM LLM-Service

## System Architecture
<img width="1462" height="484" alt="image" src="https://github.com/user-attachments/assets/f7029ae4-6ea6-4030-af32-03b43b95d8b3" />


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
