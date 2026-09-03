"""Workflow memory store. R11: one record per run with >=1 tool call; reads exclude
records with zero steps.
"""

import uuid

from config import WORKFLOW_K


class WorkflowStore:
    def __init__(self, vs):
        self._vs = vs
        self.k_default = WORKFLOW_K

    @property
    def distance_strategy(self) -> str:
        return self._vs.distance_strategy

    def add(
        self,
        *,
        query: str,
        steps: list[str],
        answer_excerpt: str,
        success: bool,
        timestamp: str,
        row_id: str | None = None,
    ) -> str:
        import json as _json

        rid = row_id or uuid.uuid4().hex
        # embed text combines query + steps so semantic search matches similar tasks
        text = query + "\n" + "\n".join(steps)
        # store steps JSON in the text? No: keep JSON in metadata via extras is not supported;
        # serialize steps into the 'text' column alongside for retrieval, and store them in
        # a dedicated column by passing through extras.
        from embeddings import get_embedder

        emb = get_embedder().embed(text)
        return self._vs.add(
            text,
            emb,
            extras={
                "query": query,
                "answer_excerpt": answer_excerpt,
                "num_steps": len(steps),
                "success": int(bool(success)),
                "timestamp": timestamp,
                "steps_json": _json.dumps(steps),
            },
            row_id=rid,
        )

    def search(self, query_embedding, k=None):
        # reads exclude records with zero steps (R11)
        return self._vs.search(
            query_embedding, k, meta_filter="num_steps > 0"
        )

    def all(self):
        return self._vs.all()

    def count(self) -> int:
        return self._vs.count()
