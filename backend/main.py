from contextlib import asynccontextmanager
from db import init_db, add_memory, search_memory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embeddings import embed_texts
from faiss_store import search as faiss_search
from db import get_chunks_by_vector_ids
from pathlib import Path
import httpx
import faiss

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
    top_k: int = 5

def load_faiss_index():
    idx_path = Path(__file__).parent / "vector_index" / "index.faiss"
    if not idx_path.exists():
        return None
    idx = faiss.read_index(str(idx_path))
    if not isinstance(idx, faiss.IndexIDMap2):
        idx = faiss.IndexIDMap2(idx)
    return idx

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
    model = req.model or DEFAULT_MODEL

    if req.mode == "general":
        payload = {"model": model, "prompt": req.question, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {"answer": data.get("response", ""), "mode": req.mode, "sources": [], "used_tools": ["ollama"], "model": model}

    if req.mode == "local":
        # Embed the question
        q_vec = (await embed_texts([req.question]))[0]

        # Search FAISS
        index = load_faiss_index()
        if index is None:
            raise HTTPException(status_code=400, detail="No FAISS index found. Run `python index.py` first.")

        scores, vector_ids = faiss_search(index, q_vec, top_k=req.top_k)

        # Fetch chunk texts for those vector IDs
        chunks = get_chunks_by_vector_ids(vector_ids)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in your local documents yet. Try re-indexing or asking differently.",
                "mode": req.mode,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
            }

        # Build a grounded prompt with citations
        context_blocks = []
        sources = []
        for i, (ch, score) in enumerate(zip(chunks, scores)):
            label = f"S{i+1}"
            context_blocks.append(
                f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
            )
            sources.append({
                "label": label,
                "doc_path": ch["doc_path"],
                "chunk_index": ch["chunk_index"],
                "score": score,
            })

        prompt = (
            "You are a local, privacy-first assistant. Answer ONLY using the provided SOURCES.\n"
            "If the answer is not in the sources, say you don't know.\n\n"
            f"QUESTION:\n{req.question}\n\n"
            "SOURCES:\n" + "\n\n".join(context_blocks) + "\n\n"
            "INSTRUCTIONS:\n"
            "- Provide a clear answer.\n"
            "- Cite sources inline like [S1], [S2].\n"
        )

        payload = {"model": model, "prompt": prompt, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": data.get("response", ""),
            "mode": req.mode,
            "sources": sources,
            "used_tools": ["ollama", "faiss", "sqlite"],
            "model": model,
        }

    raise HTTPException(status_code=400, detail="Unsupported mode. Use 'general' or 'local' for now.")

