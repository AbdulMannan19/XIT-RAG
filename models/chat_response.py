"""Chat response model."""

from typing import Literal

from pydantic import BaseModel, Field

from models.source import Source


class ChatResponse(BaseModel):
    """Chat response schema."""

    answer_text: str
    sources: list[Source]
    confidence: Literal["low", "medium", "high"]
    query_embedding_similarity: list[float] = Field(
        ..., description="Similarity scores for retrieved chunks"
    )
