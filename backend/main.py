from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="Local AI Backend")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"

class AskRequest(BaseModel):
    question: str
    mode: str = "general"  # general | local | web (later)
    model: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

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
