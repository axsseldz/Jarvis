import asyncio
from pathlib import Path

from db import init_db, sha256_file, get_doc_hash_and_status, upsert_document, mark_document_deleted, insert_chunks
from loaders import load_document
from chunking import chunk_text
from embeddings import embed_texts
from faiss_store import load_or_create, save, add_vectors, remove_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # local-ai/
DATA_ROOT = PROJECT_ROOT / "data"

INDEX_PATHS = [
    DATA_ROOT / "documents",
    DATA_ROOT / "notes",
]

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

def iter_files():
    for root in INDEX_PATHS:
        root.mkdir(parents=True, exist_ok=True)
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p.resolve()

async def run_index():
    init_db()

    files_on_disk = list(iter_files())
    disk_paths = set(str(p) for p in files_on_disk)

    # Detect deletions 
    # We can only detect deletions for docs we previously indexed; easiest is:
    # - Look at docs table, compare to disk_paths.
    from db import list_active_document_paths
    active_db_paths = set(list_active_document_paths())

    deleted_paths = sorted(active_db_paths - disk_paths)
    # Load FAISS to remove vectors
    # We'll load when we know dim — but for deletions we need it now.
    # If no index exists yet, nothing to remove.
    # We'll try loading with a default dim by reading existing index; if missing, skip.
    index = None
    try:
        # If index exists, load it (dim irrelevant because it’s saved)
        from faiss_store import INDEX_PATH as _IDX_PATH
        if _IDX_PATH.exists():
            import faiss
            index = faiss.read_index(str(_IDX_PATH))
            if not isinstance(index, faiss.IndexIDMap2):
                index = faiss.IndexIDMap2(index)
    except Exception:
        index = None

    if deleted_paths and index is not None:
        all_removed_vec_ids = []
        for dp in deleted_paths:
            vec_ids = mark_document_deleted(dp)
            all_removed_vec_ids.extend(vec_ids)
        remove_ids(index, all_removed_vec_ids)
        save(index)
        print(f"Removed deleted docs: {len(deleted_paths)} (vectors removed: {len(all_removed_vec_ids)})")
    elif deleted_paths:
        # mark deleted even if no index yet
        for dp in deleted_paths:
            mark_document_deleted(dp)
        print(f"Removed deleted docs: {len(deleted_paths)} (no FAISS index yet)")

    # Index new/changed files
    # We'll load/create FAISS once we know embedding dim.
    next_vector_id = 1
    # To avoid collisions across runs, we can use a stable increasing id.
    # MVP approach: compute next id from SQLite max(vector_id).
    from db import get_conn
    with get_conn() as conn:
        row = conn.execute("SELECT COALESCE(MAX(vector_id), 0) AS mx FROM chunks").fetchone()
        next_vector_id = int(row["mx"]) + 1

    indexed_count = 0
    skipped_count = 0
    chunks_indexed = 0

    for path in files_on_disk:
        path_str = str(path)

        file_hash = sha256_file(path)
        prev = get_doc_hash_and_status(path_str)
        if prev and prev["status"] == "active" and prev["file_hash"] == file_hash:
            skipped_count += 1
            continue  # unchanged, skip

        # Load and chunk
        text = load_document(path)
        chunks = chunk_text(text)
        if not chunks:
            skipped_count += 1
            continue

        # Embed chunks
        vecs = await embed_texts(chunks)
        dim = len(vecs[0])

        # Load/create FAISS index if not loaded
        if index is None:
            index = load_or_create(dim)
        else:
            # if an index already exists, assume dim matches
            pass

        # Upsert doc row
        doc_id = upsert_document(path_str, file_hash)

        # If doc changed and existed, strict approach: remove old vectors first
        # We do this by marking deleted then immediately re-inserting active.
        # Simpler: if prev existed active but hash changed, delete old chunks/vectors.
        if prev and prev["status"] == "active" and prev["file_hash"] != file_hash:
            # remove old vectors for this doc
            vec_ids = mark_document_deleted(path_str)  # marks deleted + deletes chunks
            if vec_ids and index is not None:
                remove_ids(index, vec_ids)
            # re-activate doc with new hash
            doc_id = upsert_document(path_str, file_hash)

        # Assign vector ids
        vector_ids = list(range(next_vector_id, next_vector_id + len(vecs)))
        next_vector_id += len(vecs)

        # Save chunks rows
        insert_chunks(doc_id, chunks, vector_ids)

        chunks_indexed += len(chunks)

        # Add to FAISS
        add_vectors(index, vecs, vector_ids)
        save(index)

        indexed_count += 1
        print(f"Indexed: {path.name} (chunks={len(chunks)})")

    print(f"Done. indexed={indexed_count}, skipped={skipped_count}, deleted={len(deleted_paths)}")

    return {
        "files_on_disk": len(files_on_disk),
        "indexed_files": indexed_count,
        "skipped_files": skipped_count,
        "deleted_files": len(deleted_paths),
        "chunks_indexed": chunks_indexed,
    }

if __name__ == "__main__":
    asyncio.run(run_index())
