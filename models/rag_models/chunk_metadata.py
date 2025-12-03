"""Chunk metadata model."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl


class ChunkMetadata(BaseModel):
    """Chunk metadata schema."""

    chunk_id: str
    page_url: HttpUrl
    chunk_text: str
    chunk_order: int
    section_heading: Optional[str] = None
    char_offset_start: int
    char_offset_end: int
    crawl_timestamp: datetime
    content_type: Literal["html", "pdf", "faq", "form"]
    raw_html_snippet: Optional[str] = None
