import uuid
import asyncio

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def begin(self, prompt):
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "prompt": prompt,
            "text": prompt,
            "length": len(prompt.split()),
        }
        return sid, self.sessions[sid]["length"]

    def append(self, sid, token_str):
        self.sessions[sid]["text"] += token_str
        self.sessions[sid]["length"] += 1

    def get_text(self, sid):
        return self.sessions[sid]["text"]

    def sync_tail(self, sid, tail):
        self.sessions[sid]["text"] += "".join(tail)
        self.sessions[sid]["length"] += len(tail)

    def end(self, sid):
        if sid in self.sessions:
            del self.sessions[sid]
            return True
        return False
