"""Embeddings wrapper around sentence-transformers.

One local model serves every store and both write/read paths (section 2, CTX-C5).
"""

from functools import lru_cache

from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed(self, text: str) -> list[float]:
        encoder = self._load()
        vec = encoder.encode([text], normalize_embeddings=True)
        return [float(x) for x in vec[0]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        encoder = self._load()
        vecs = encoder.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
