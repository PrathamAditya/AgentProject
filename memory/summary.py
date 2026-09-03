"""Summary memory store. R9: recoverable consolidation — summary records hold the summary
text and the full original content, retrievable by id and by query+thread similarity.
"""

from config import SUMMARY_K


class SummaryStore:
    def __init__(self, vs):
        self._vs = vs
        self.k_default = SUMMARY_K

    @property
    def distance_strategy(self) -> str:
        return self._vs.distance_strategy

    def add(
        self,
        *,
        summary_id: str,
        summary: str,
        description: str,
        full_content: str,
        thread_id: str | None,
        embedding: list[float],
    ) -> str:
        text = summary + "\n" + description
        return self._vs.add(
            text,
            embedding,
            extras={
                "summary": summary,
                "summary_id": summary_id,
                "description": description,
                "full_content": full_content,
                "thread_id": thread_id,
            },
        )

    def get_by_id(self, summary_id: str):
        row = self._vs._conn.execute(
            "SELECT * FROM summary_memory WHERE summary_id = ?", (summary_id,)
        ).fetchone()
        return row

    def search(self, query_embedding, k=None, thread_id: str | None = None):
        if thread_id is not None:
            return self._vs.search(
                query_embedding, k, meta_filter="thread_id = ?", meta_filter_params=(thread_id,)
            )
        return self._vs.search(query_embedding, k)

    def all(self):
        return self._vs.all()

    def count(self) -> int:
        return self._vs.count()
