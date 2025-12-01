from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self._warmup()

    def _warmup(self):
        """Warm up the model with a dummy embedding to avoid cold start."""
        try:
            _ = self.get_embedding("warmup")
        except Exception:
            pass

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            raise

    @lru_cache(maxsize=256)
    def get_embedding_cached(self, text: str) -> tuple:
        """Get cached embedding as tuple for hashability."""
        embedding = self.get_embeddings([text])[0]
        return tuple(embedding)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.get_embeddings([text])[0]


_embedding_provider_cache = None


def get_embedding_provider() -> EmbeddingService:
    global _embedding_provider_cache
    if _embedding_provider_cache is None:
        _embedding_provider_cache = EmbeddingService()
    return _embedding_provider_cache
