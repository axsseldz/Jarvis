import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  source TEXT DEFAULT 'manual'
);

CREATE INDEX IF NOT EXISTS idx_episodic_created_at ON episodic_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_episodic_content ON episodic_memory(content);
"""

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

def add_memory(content: str, source: str = "manual") -> int:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO episodic_memory(content, created_at, source) VALUES (?, ?, ?)",
            (content, now, source),
        )
        return int(cur.lastrowid)

def search_memory(query: str, limit: int = 10):
    like = f"%{query}%"
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at, source FROM episodic_memory WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (like, limit),
        ).fetchall()
        return [dict(r) for r in rows]
