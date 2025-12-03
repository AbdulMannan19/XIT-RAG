from typing import Union, overload
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    @overload
    def get_embedding(self, text: str) -> np.ndarray: ...

    @overload
    def get_embedding(self, text: list[str]) -> np.ndarray: ...

    def get_embedding(self, text: Union[str, list[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        
        return embeddings[0] if len(text) == 1 else embeddings

