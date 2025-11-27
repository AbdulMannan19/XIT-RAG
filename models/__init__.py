"""Models package - Data structures."""

from .content_type import ContentType
from .crawled_page import CrawledPage
from .chunk import Chunk

__all__ = [
    "ContentType",
    "CrawledPage",
    "Chunk",
]
