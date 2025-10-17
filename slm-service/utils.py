import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup(model_name="meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float16):
    # load .env
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") # create .env file with your own token
    login(hf_token)

    # Load the SLM (e.g. Llama 3.2 1B Instruct)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,       # or bfloat16/8bit/4bit for on-device
        device_map="auto"
    )

    return model, tokenizer