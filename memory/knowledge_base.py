"""Knowledge-base (semantic) memory store. R14: acquisition tools persist what they
find with source metadata in the same call. Chunk sizes respect the 1,500/200 default.
"""

from typing import TYPE_CHECKING

from config import KB_K

if TYPE_CHECKING:
    from memory.vector_store import VectorStore


class KnowledgeBaseStore:
    def __init__(self, vs: "VectorStore"):
        self._vs = vs
        self.k_default = KB_K

    @property
    def distance_strategy(self) -> str:
        return self._vs.distance_strategy

    def add_chunk(
        self,
        chunk_text: str,
        embedding: list[float],
        *,
        source: str,
        chunk_id: int,
        num_chunks: int,
        title: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        return self._vs.add(
            chunk_text,
            embedding,
            extras={
                "source": source,
                "chunk_id": chunk_id,
                "num_chunks": num_chunks,
                "title": title,
                "timestamp": timestamp,
            },
        )

    def search(self, query_embedding: list[float], k: int | None = None):
        return self._vs.search(query_embedding, k)

    def all(self):
        return self._vs.all()

    def count(self) -> int:
        return self._vs.count()
