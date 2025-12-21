def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Simple character-based chunking.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = max(j - overlap, j)  # overlap
        if i == j:
            i = j
    return chunks
