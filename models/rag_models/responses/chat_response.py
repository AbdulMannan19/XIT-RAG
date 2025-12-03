from typing import Literal
from pydantic import BaseModel, Field
from models.rag_models.responses.source import Source


class ChatResponse(BaseModel):


    answer_text: str
    sources: list[Source]
    confidence: Literal["low", "medium", "high"]
    query_embedding_similarity: list[float] = Field(
        ..., description="Similarity scores for retrieved chunks"
    )
