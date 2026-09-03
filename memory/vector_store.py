"""Vector store base: text + JSON metadata + embedding, exact cosine similarity at
fixture scale. One distance strategy from config (R15, AC17).
"""

import json
import sqlite3
import uuid

from config import DISTANCE_STRATEGY, cosine_similarity


def _parse_embedding(row: sqlite3.Row) -> list[float]:
    raw = row["embedding"]
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)


class VectorStore:
    def __init__(self, conn: sqlite3.Connection, table: str, k_default: int):
        self._conn = conn
        self._table = table
        self.k_default = k_default

    @property
    def distance_strategy(self) -> str:
        return DISTANCE_STRATEGY

    def _row_cols(self) -> list[str]:
        # discover actual columns of the table
        cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({self._table})")]
        return cols

    def add(
        self,
        text: str,
        embedding: list[float],
        *,
        extras: dict | None = None,
        row_id: str | None = None,
    ) -> str:
        rid = row_id or uuid.uuid4().hex
        cols = ["id", "text", "embedding"]
        cols += list((extras or {}).keys())
        ph = ", ".join(["?"] * len(cols))
        values = [rid, text, json.dumps(embedding)]
        values += [extras.get(c) if extras else None for c in cols[3:]]
        self._conn.execute(
            f"INSERT INTO {self._table} ({', '.join(cols)}) VALUES ({ph})", values
        )
        self._conn.commit()
        return rid

    def search(
        self,
        query_embedding: list[float],
        k: int | None = None,
        *,
        meta_filter: str | None = None,
        meta_filter_params: tuple = (),
        order_by: str | None = None,
    ) -> list[tuple[float, sqlite3.Row]]:
        k = k or self.k_default
        where = ""
        if meta_filter:
            where = f"WHERE {meta_filter}"
        rows = self._conn.execute(
            f"SELECT * FROM {self._table} {where}", meta_filter_params
        ).fetchall()

        if order_by:
            rows = sorted(rows, key=lambda r: r[order_by])

        scored = []
        for row in rows:
            emb = _parse_embedding(row)
            sim = cosine_similarity(query_embedding, emb)
            scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def all(self) -> list[sqlite3.Row]:
        return self._conn.execute(f"SELECT * FROM {self._table}").fetchall()

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) AS n FROM {self._table}").fetchone()["n"]
