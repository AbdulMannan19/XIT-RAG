"""Chunk model."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl

from models.rag_models.content_type import ContentType


class Chunk(BaseModel):
    """Model for text chunk."""

    chunk_id: str
    page_url: HttpUrl
    chunk_text: str
    chunk_order: int
    section_heading: Optional[str] = None
    char_offset_start: int
    char_offset_end: int
    crawl_timestamp: datetime
    content_type: ContentType
    raw_html_snippet: Optional[str] = None
    page_number: Optional[int] = None  # For PDFs
