"""Entity memory store. R12: entities (PERSON/PLACE/SYSTEM) written from the LLM.
"""

from config import ENTITY_K


class EntityStore:
    def __init__(self, vs):
        self._vs = vs
        self.k_default = ENTITY_K

    @property
    def distance_strategy(self) -> str:
        return self._vs.distance_strategy

    def add(self, *, name: str, type_: str, description: str, embedding: list[float]):
        return self._vs.add(
            name,
            embedding,
            extras={"name": name, "type": type_, "description": description},
        )

    def search(self, query_embedding, k=None):
        return self._vs.search(query_embedding, k)

    def all(self):
        return self._vs.all()

    def count(self) -> int:
        return self._vs.count()

    def format_bullets(self, query_embedding, k=None):
        k = k or self.k_default
        bullets = []
        for _, row in self._vs.search(query_embedding, k=k):
            bullets.append(f"- {row['name']} ({row['type']}): {row['description']}")
        return bullets
