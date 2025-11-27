"""Content type enumeration."""

from enum import Enum


class ContentType(str, Enum):
    """Content type enumeration."""

    HTML = "html"
    PDF = "pdf"
    FAQ = "faq"
    FORM = "form"
