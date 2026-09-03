"""Toolbox memory store. R4/R5: registration idempotent per tool name; retrieval returns
focused top-k unique schemas in OpenAI function format.
"""

import json

from config import TOOLBOX_K


class ToolboxStore:
    def __init__(self, vs):
        self._vs = vs
        self.k_default = TOOLBOX_K

    @property
    def distance_strategy(self) -> str:
        return self._vs.distance_strategy

    def _exists(self, tool_name: str) -> bool:
        row = self._vs._conn.execute(
            "SELECT 1 FROM toolbox_memory WHERE tool_name = ?", (tool_name,)
        ).fetchone()
        return row is not None

    def register(
        self,
        *,
        tool_name: str,
        description: str,
        signature: str,
        embedding_text: str,
        embedding: list[float],
        synthetic_queries: list[str] | None = None,
    ) -> bool:
        """Returns True if a new row was inserted, False if the tool already existed (idempotent)."""
        if self._exists(tool_name):
            return False
        self._vs.add(
            embedding_text,
            embedding,
            extras={
                "tool_name": tool_name,
                "description": description,
                "signature": signature,
                "synthetic_queries": json.dumps(synthetic_queries or []),
            },
        )
        return True

    def search_raw(self, query_embedding, k=None):
        results = self._vs.search(query_embedding, k)
        return [row for _, row in results]

    def retrieve_schemas(self, query_embedding, k: int | None = None) -> list[dict]:
        k = k or self.k_default
        seen: set[str] = set()
        out: list[dict] = []
        for _, row in self._vs.search(query_embedding, k=k):
            name = row["tool_name"]
            if name in seen:
                continue
            seen.add(name)
            out.append(self._to_schema(row))
        return out

    def _to_schema(self, row) -> dict:
        params = row["signature"]
        try:
            params = json.loads(params)
        except (TypeError, ValueError):
            params = {"type": "object", "properties": {}, "required": []}
        return {
            "type": "function",
            "function": {
                "name": row["tool_name"],
                "description": row["description"],
                "parameters": params,
            },
        }

    def count(self) -> int:
        return self._vs.count()
