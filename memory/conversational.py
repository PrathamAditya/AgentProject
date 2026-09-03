"""Conversational memory store (SQL-style; exact, chronological, thread-keyed).

R3: reads filter to the thread, order by timestamp ascending, and skip rows whose
summary_id is set.
"""

import json
import sqlite3
import uuid

from config import CONVERSATION_LIMIT


def _r(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("metadata"):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (TypeError, ValueError):
            pass
    return d


class ConversationalStore:
    def __init__(self, conn: sqlite3.Connection, limit: int = CONVERSATION_LIMIT):
        self._conn = conn
        self.limit = limit

    def add(
        self,
        thread_id: str,
        role: str,
        content: str,
        timestamp: str,
        *,
        row_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        rid = row_id or uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO conversational_memory (id, thread_id, role, content, timestamp, metadata, summary_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, thread_id, role, content, timestamp, json.dumps(metadata) if metadata else None, None),
        )
        self._conn.commit()
        return rid

    def read(
        self, thread_id: str, limit: int | None = None
    ) -> list[dict]:
        limit = limit if limit is not None else self.limit
        rows = self._conn.execute(
            "SELECT * FROM conversational_memory WHERE thread_id = ? AND summary_id IS NULL "
            "ORDER BY timestamp ASC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        return [_r(r) for r in rows]

    def read_all(
        self, thread_id: str
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM conversational_memory WHERE thread_id = ? ORDER BY timestamp ASC",
            (thread_id,),
        ).fetchall()
        return [_r(r) for r in rows]

    def read_unconsolidated_all(self, thread_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM conversational_memory WHERE thread_id = ? AND summary_id IS NULL "
            "ORDER BY timestamp ASC",
            (thread_id,),
        ).fetchall()
        return [_r(r) for r in rows]

    def unconsolidated_count(self, thread_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversational_memory "
            "WHERE thread_id = ? AND summary_id IS NULL",
            (thread_id,),
        ).fetchone()
        return row["n"]

    def thread_messages(self, thread_id: str) -> list[dict]:
        return self.read_all(thread_id)

    def list_threads(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT thread_id, COUNT(*) AS n, MAX(timestamp) AS last_ts "
            "FROM conversational_memory GROUP BY thread_id ORDER BY last_ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversational_memory"
        ).fetchone()
        return row["n"]
