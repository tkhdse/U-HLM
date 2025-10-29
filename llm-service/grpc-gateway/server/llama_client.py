import httpx
import numpy as np

class LlamaClient:
    def __init__(self, host="127.0.0.1", port=8080):
        self.base = f"http://{host}:{port}"

    async def logits(self, session_text):
        """
        Calls llama.cpp HTTP server for logits at next token position.
        Returns softmax-normalized probs as numpy array.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base}/completion",
                                  json={"prompt": session_text,
                                        "n_predict": 1,
                                        "logits_all": True,
                                        "temperature": 1.0})
            r.raise_for_status()
            logits = np.array(r.json()["logits"][0], dtype=np.float32)
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            return probs

