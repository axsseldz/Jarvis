from contextlib import asynccontextmanager
from db import init_db, add_memory, search_memory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embeddings import embed_texts
from faiss_store import search as faiss_search
from db import get_chunks_by_vector_ids
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
from index import run_index  
from fastapi import UploadFile, File
import httpx
import faiss
import json
import asyncio
import os

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
BASE_DIR = Path(__file__).resolve().parent.parent  # local-ai/
DOCS_DIR = BASE_DIR / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_INTERVAL_SECONDS = 180  # 3 minutes (tune later)

INDEX_LOCK = asyncio.Lock()
INDEX_STATUS = {
    "state": "idle",               # idle | running | ok | error
    "is_indexing": False,
    "last_trigger": None,          # "startup" | "scheduled" | "manual"
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "stats": None,
}

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

async def _run_index_job(trigger: str):
    # Prevent overlapping runs
    if INDEX_LOCK.locked():
        return False

    async with INDEX_LOCK:
        INDEX_STATUS["state"] = "running"
        INDEX_STATUS["is_indexing"] = True
        INDEX_STATUS["last_trigger"] = trigger
        INDEX_STATUS["last_started_at"] = _now_iso()
        INDEX_STATUS["last_error"] = None

        try:
            stats = await run_index()
            INDEX_STATUS["stats"] = stats
            INDEX_STATUS["state"] = "ok"
        except Exception as e:
            INDEX_STATUS["state"] = "error"
            INDEX_STATUS["last_error"] = f"{type(e).__name__}: {str(e)}"
            # Optional: keep traceback for debugging
            # print(traceback.format_exc())
        finally:
            INDEX_STATUS["is_indexing"] = False
            INDEX_STATUS["last_finished_at"] = _now_iso()

    return True

async def _index_daemon(stop_event: asyncio.Event):
    # Run once on startup
    await _run_index_job("startup")

    while not stop_event.is_set():
        try:
            # wait N seconds or stop
            await asyncio.wait_for(stop_event.wait(), timeout=INDEX_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            # time to run scheduled indexing
            await _run_index_job("scheduled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    stop_event = asyncio.Event()
    task = asyncio.create_task(_index_daemon(stop_event))

    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except Exception:
            pass

app = FastAPI(title="Local AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    mode: str = "auto"          # auto | local | general
    model: str | None = None
    top_k: int = 6
    task: str | None = None     # qa | summary | None (auto-infer)

class RememberRequest(BaseModel):
    content: str
    source: str = "manual"

def infer_task(question: str) -> str:
    q = question.lower()
    summary_keywords = ["summarize", "summary", "overview", "bullet", "bullets", "tl;dr", "high level"]
    if any(k in q for k in summary_keywords):
        return "summary"
    return "qa"

def should_fallback_to_general(vector_ids: list[int], scores: list[float]) -> bool:
    """
    Heuristic: if we retrieved nothing OR the best similarity score is low,
    the question is probably not answerable from local docs.
    Tune threshold based on your embeddings/model.
    """
    if not vector_ids:
        return True
    best = scores[0] if scores else 0.0
    return best < 0.38  # adjust if needed (0.33-0.45 typical)

async def ollama_stream(prompt: str, model: str):
    """
    Yields text chunks from Ollama as they arrive (NDJSON stream).
    """
    payload = {"model": model, "prompt": prompt, "stream": True}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as r:
            r.raise_for_status()

            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Ollama streams partial text in `response`
                chunk = obj.get("response", "")
                if chunk:
                    yield chunk

                if obj.get("done"):
                    break

async def ollama_generate(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

def load_faiss_index():
    idx_path = Path(__file__).parent / "vector_index" / "index.faiss"
    if not idx_path.exists():
        return None
    idx = faiss.read_index(str(idx_path))
    if not isinstance(idx, faiss.IndexIDMap2):
        idx = faiss.IndexIDMap2(idx)
    return idx

async def build_local_prompt_and_sources(question: str, task: str, top_k: int) -> tuple[str | None, list, dict]:
    q_vec = (await embed_texts([question]))[0]

    index = load_faiss_index()
    if index is None:
        return None, [], {"top_k": top_k, "best_score": 0.0}

    k = top_k
    if task == "summary":
        k = max(k, 12)

    scores, vector_ids = faiss_search(index, q_vec, top_k=k)
    chunks = get_chunks_by_vector_ids(vector_ids)

    if not chunks:
        return None, [], {"top_k": k, "best_score": float(scores[0]) if scores else 0.0}

    context_blocks = []
    sources = []
    for i, ch in enumerate(chunks):
        label = f"S{i+1}"
        context_blocks.append(
            f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
        )
        src_score = scores[i] if i < len(scores) else None
        sources.append({
            "label": label,
            "doc_path": ch["doc_path"],
            "chunk_index": ch["chunk_index"],
            "score": float(src_score) if src_score is not None else None,
        })

    if task == "summary":
        instructions = (
            "TASK: Summarize using ONLY the provided SOURCES.\n"
            "You MUST synthesize a summary even if the sources are split across chunks.\n"
            "Do NOT say 'I don't know' just because a summary isn't explicitly written.\n"
            "If important sections are missing, make a partial summary and say what seems missing.\n"
            "Follow the user's format request (e.g. bullet points).\n"
            "Cite sources inline like [S1], [S2].\n"
        )
    else:
        instructions = (
            "TASK: Answer using ONLY the provided SOURCES.\n"
            "If the answer cannot be found in the sources, say 'I don't know'.\n"
            "Cite sources inline like [S1], [S2].\n"
        )

    prompt = (
        "You are a local, privacy-first assistant.\n\n"
        f"QUESTION:\n{question}\n\n"
        "SOURCES:\n" + "\n\n".join(context_blocks) + "\n\n"
        f"{instructions}"
    )

    return prompt, sources, {"top_k": k, "best_score": float(scores[0]) if scores else 0.0}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/docs/list")
def docs_list():
    items = []
    for p in sorted(DOCS_DIR.glob("*")):
        if not p.is_file():
            continue
        # keep it simple; you can filter extensions later
        st = p.stat()
        items.append({
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"docs": items}

@app.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
    # Basic safety: no directories
    filename = os.path.basename(file.filename or "upload.bin")
    dest = DOCS_DIR / filename

    # Avoid overwriting: add suffix if exists
    if dest.exists():
        stem = dest.stem
        suf = dest.suffix
        i = 1
        while True:
            candidate = DOCS_DIR / f"{stem}_{i}{suf}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    content = await file.read()
    dest.write_bytes(content)

    # Kick indexing in background (do NOT block)
    try:
        asyncio.create_task(_run_index_job("upload"))
    except Exception:
        pass

    return {"ok": True, "saved_as": dest.name, "path": str(dest)}

@app.get("/index/status")
def index_status():
    return INDEX_STATUS

@app.post("/index/run")
async def index_run():
    # fire-and-forget (don’t block request)
    if INDEX_LOCK.locked():
        return {"ok": False, "started": False, "status": INDEX_STATUS}

    asyncio.create_task(_run_index_job("manual"))
    return {"ok": True, "started": True, "status": INDEX_STATUS}

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
    task = req.task or infer_task(req.question)

    # ---------- GENERAL MODE ----------
    async def run_general() -> dict:
        try:
            answer = await ollama_generate(req.question, model)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": answer,
            "mode": "general",
            "task": task,
            "sources": [],
            "used_tools": ["ollama"],
            "model": model,
        }

    # ---------- LOCAL MODE ----------
    async def run_local() -> dict:
        # 1) Embed the question
        q_vec = (await embed_texts([req.question]))[0]

        # 2) Load FAISS index
        index = load_faiss_index()
        if index is None:
            # No local KB yet
            return {
                "answer": "No local index found yet. Run indexing first, or use General mode.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
            }

        # 3) Retrieval settings
        k = req.top_k
        if task == "summary":
            k = max(k, 12)  # summaries usually need more context

        scores, vector_ids = faiss_search(index, q_vec, top_k=k)
        chunks = get_chunks_by_vector_ids(vector_ids)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in your local documents.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
            }

        # 4) Build context blocks + sources
        context_blocks = []
        sources = []
        for i, ch in enumerate(chunks):
            label = f"S{i+1}"
            context_blocks.append(
                f"[{label}] doc={ch['doc_path']} chunk={ch['chunk_index']}\n{ch['text']}"
            )
            src_score = scores[i] if i < len(scores) else None
            sources.append({
                "label": label,
                "doc_path": ch["doc_path"],
                "chunk_index": ch["chunk_index"],
                "score": float(src_score) if src_score is not None else None,
            })

        # 5) Task-aware instructions (QA vs Summary)
        if task == "summary":
            instructions = (
                "TASK: Summarize using ONLY the provided SOURCES.\n"
                "You MUST synthesize a summary even if the sources are split across chunks.\n"
                "Do NOT say 'I don't know' just because a summary isn't explicitly written.\n"
                "If important sections are missing, make a partial summary and say what seems missing.\n"
                "Follow the user's format request (e.g. bullet points).\n"
                "Cite sources inline like [S1], [S2].\n"
            )
        else:
            instructions = (
                "TASK: Answer using ONLY the provided SOURCES.\n"
                "If the answer cannot be found in the sources, say 'I don't know'.\n"
                "Cite sources inline like [S1], [S2].\n"
            )

        prompt, sources, retrieval = await build_local_prompt_and_sources(req.question, task, req.top_k)

        if prompt is None:
            return {
                "answer": "I couldn't find anything relevant in your local documents.",
                "mode": "local",
                "task": task,
                "sources": [],
                "used_tools": ["ollama", "faiss", "sqlite"],
                "model": model,
                "retrieval": retrieval,
            }

        try:
            answer = await ollama_generate(prompt, model)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is `ollama serve` running?")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

        return {
            "answer": answer,
            "mode": "local",
            "task": task,
            "sources": sources,
            "used_tools": ["ollama", "faiss", "sqlite"],
            "model": model,
            "retrieval": retrieval,
        }

    # ---------- ROUTING ----------
    mode = (req.mode or "auto").lower().strip()

    if mode == "general":
        return await run_general()

    if mode == "local":
        return await run_local()

    if mode == "auto":
        local_result = await run_local()

        # If local QA can't answer, fall back to general automatically
        if local_result.get("task") == "qa":
            ans = (local_result.get("answer") or "").strip().lower()
            if ans.startswith("i don't know") or "i don't know" in ans:
                general_result = await run_general()
                general_result["mode"] = "auto->general"
                return general_result

        # Also fallback if retrieval is missing/weak (still useful)
        retrieval = local_result.get("retrieval", {})
        best_score = float(retrieval.get("best_score", 0.0))
        has_sources = bool(local_result.get("sources"))

        if should_fallback_to_general(
            vector_ids=[1] if has_sources else [],
            scores=[best_score] if has_sources else [],
        ):
            general_result = await run_general()
            general_result["mode"] = "auto->general"
            return general_result

        local_result["mode"] = "auto->local"
        return local_result

    raise HTTPException(status_code=400, detail="Unsupported mode. Use 'auto', 'local', or 'general'.")

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    model = req.model or DEFAULT_MODEL
    task = req.task or infer_task(req.question)
    mode = (req.mode or "auto").lower().strip()

    async def sse():
        # Decide route + build prompt (without generating yet)
        final_mode = mode
        sources = []
        retrieval = {}

        if mode == "general":
            prompt = req.question

        elif mode == "local":
            prompt, sources, retrieval = await build_local_prompt_and_sources(req.question, task, req.top_k)
            if prompt is None:
                meta = {"mode": "local", "task": task, "sources": [], "model": model, "retrieval": retrieval}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                yield f"data: I couldn't find anything relevant in your local documents.\n\n"
                yield "event: done\ndata: {}\n\n"
                return

        elif mode == "auto":
            prompt_local, sources_local, retrieval_local = await build_local_prompt_and_sources(req.question, task, req.top_k)

            # fallback decision (same heuristic as non-stream)
            if prompt_local is None or should_fallback_to_general(
                vector_ids=[1] if sources_local else [],
                scores=[float(retrieval_local.get("best_score", 0.0))] if sources_local else [],
            ):
                final_mode = "auto->general"
                prompt = req.question
                sources = []
                retrieval = {}
            else:
                final_mode = "auto->local"
                prompt = prompt_local
                sources = sources_local
                retrieval = retrieval_local
        else:
            meta = {"mode": "error", "task": task, "sources": [], "model": model}
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            yield f"data: Unsupported mode. Use 'auto', 'local', or 'general'.\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Send meta first (frontend uses this to set sources/mode/task)
        meta = {"mode": final_mode, "task": task, "sources": sources, "model": model, "retrieval": retrieval}
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Stream tokens
        try:
            async for chunk in ollama_stream(prompt, model):
                # SSE message
                yield f"data: {chunk}\n\n"
        except httpx.ConnectError:
            yield f"data: Cannot connect to Ollama. Is `ollama serve` running?\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


