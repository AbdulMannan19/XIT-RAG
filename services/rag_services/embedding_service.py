import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        return self.get_embeddings([text])[0]


def get_embedding_provider() -> LocalEmbedding:
    return LocalEmbedding()
