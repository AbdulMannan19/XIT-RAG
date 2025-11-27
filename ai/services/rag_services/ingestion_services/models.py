"""Data models for ingestion pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, HttpUrl


class ContentType(str, Enum):
    """Content type enumeration."""

    HTML = "html"
    PDF = "pdf"
    FAQ = "faq"
    FORM = "form"


class CrawledPage(BaseModel):
    """Model for crawled page data."""

    url: HttpUrl
    title: str
    crawl_timestamp: datetime
    last_modified: Optional[datetime] = None
    content_type: ContentType
    raw_content: bytes  # Raw HTML or PDF binary
    cleaned_text: str
    content_hash: str
    etag: Optional[str] = None
    status_code: int = 200


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


