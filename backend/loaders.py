from pathlib import Path
import pymupdf 

TEXT_EXTS = {".txt", ".md"}

def load_pdf_text(path: Path) -> str:
    doc = pymupdf.open(str(path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_text(path)
    if suffix in TEXT_EXTS:
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {path.name}")
