"""Vector chunk model for Qdrant."""

from typing import Literal, Optional

from pydantic import BaseModel


class VectorChunk(BaseModel):
    """Vector chunk schema for Qdrant."""

    id: str
    url: str
    title: str
    section_heading: Optional[str] = None
    text: str
    char_start: int
    char_end: int
    content_type: Literal["html", "pdf", "faq", "form"]
    crawl_ts: str  # ISO8601
    last_modified: Optional[str] = None  # ISO8601
    language: str = "en"
    embedding_model: str
    tokens: int = 0
    hash: str
