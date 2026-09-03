"""Single SQLite connection + schema. All seven memory stores live in one on-disk file
so they survive process restarts (D5/D14).
"""

import sqlite3

from config import DB_PATH


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    # conversational memory (SQL-style, exact/chronological/thread-keyed)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversational_memory (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            summary_id TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversational_memory(thread_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversational_memory(timestamp)"
    )

    # tool log memory (SQL-style)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_log_memory (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            tool_call_id TEXT,
            tool_name TEXT NOT NULL,
            tool_args TEXT NOT NULL,
            result TEXT NOT NULL,
            result_preview TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            metadata TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )

    # vector stores: text + JSON metadata + embedding (cosine brute-force at fixture scale)
    _vector_table(
        conn, "knowledge_base", ["source", "chunk_id", "num_chunks", "title", "timestamp"]
    )
    _vector_table(
        conn,
        "workflow_memory",
        ["query", "answer_excerpt", "num_steps", "success", "timestamp", "steps_json"],
    )
    _vector_table(
        conn, "toolbox_memory", ["tool_name", "description", "signature", "synthetic_queries"]
    )
    _vector_table(conn, "entity_memory", ["name", "type", "description"])
    _vector_table(
        conn, "summary_memory", ["summary_id", "summary", "description", "full_content", "thread_id"]
    )

    conn.commit()


def _vector_table(conn: sqlite3.Connection, table: str, extra_cols: list[str]) -> None:
    cols = ["id TEXT PRIMARY KEY", "text TEXT NOT NULL", "embedding TEXT NOT NULL"]
    for c in extra_cols:
        cols.append(c)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)})")
