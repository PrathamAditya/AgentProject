"""Tool log memory store (SQL-style). R10: every tool execution fully persisted;
model receives a bounded excerpt.
"""

import json
import sqlite3
import uuid


class ToolLogStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def log(
        self,
        *,
        thread_id: str | None,
        tool_name: str,
        tool_args: dict,
        result: str,
        result_preview: str,
        status: str,
        error_message: str | None = None,
        tool_call_id: str | None = None,
        timestamp: str,
        metadata: dict | None = None,
        row_id: str | None = None,
    ) -> str:
        rid = row_id or uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO tool_log_memory (id, thread_id, tool_call_id, tool_name, tool_args, "
            "result, result_preview, status, error_message, metadata, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                thread_id,
                tool_call_id,
                tool_name,
                json.dumps(tool_args),
                result,
                result_preview,
                status,
                error_message,
                json.dumps(metadata) if metadata else None,
                timestamp,
            ),
        )
        self._conn.commit()
        return rid

    def get(self, log_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tool_log_memory WHERE id = ?", (log_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["tool_args"] = json.loads(d["tool_args"])
        except (TypeError, ValueError):
            pass
        return d
