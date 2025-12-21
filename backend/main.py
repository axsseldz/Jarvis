from contextlib import asynccontextmanager
from db import init_db, add_memory, search_memory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Local AI Backend", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str
    mode: str = "general"  # general | local | web (later)
    model: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

class RememberRequest(BaseModel):
    content: str
    source: str = "manual"

@app.post("/remember")
def remember(req: RememberRequest):
    mem_id = add_memory(req.content, req.source)
    return {"ok": True, "id": mem_id}

@app.get("/memory/search")
def memory_search(q: str, limit: int = 10):
    return {"results": search_memory(q, limit)}

@app.post("/ask")
async def ask(req: AskRequest):


    print(req)
    
    if req.mode != "general":
        raise HTTPException(status_code=400, detail="Only mode='general' is implemented right now.")

    model = req.model or DEFAULT_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": req.question}
        ],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Ollama HTTP error: {e.response.status_code} {e.response.text}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

    answer = (data.get("message") or {}).get("content", "")

    return {
        "answer": answer,
        "mode": req.mode,
        "sources": [],
        "used_tools": ["ollama:/api/chat"],
        "model": model,
    }
