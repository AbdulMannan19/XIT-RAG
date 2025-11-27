"""Crawled page model."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl

from models.content_type import ContentType


class CrawledPage(BaseModel):

    url: HttpUrl
    title: str
    crawl_timestamp: datetime
    last_modified: Optional[datetime] = None
    content_type: ContentType
    raw_content: bytes 
    cleaned_text: str
    content_hash: str
    etag: Optional[str] = None
    status_code: int = 200
