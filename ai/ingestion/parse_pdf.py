"""PDF parsing and text extraction."""

import logging
from typing import Optional

import fitz  # PyMuPDF
from io import BytesIO

from core.utils import normalize_text
from ingestion.models import ContentType, CrawledPage

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, dict]:
    """Extract text from PDF with page numbers and headings."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        page_texts = []
        headings = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            normalized = normalize_text(text)

            if normalized:
                full_text.append(f"[Page {page_num}]\n{normalized}\n")
                page_texts.append({"page": page_num, "text": normalized})

            # Try to extract headings (bold text, larger font)
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            font_size = span.get("size", 0)
                            flags = span.get("flags", 0)
                            text_span = span.get("text", "").strip()

                            # Heuristic: headings are often bold and larger
                            if text_span and (flags & 16) and font_size > 10:  # Bold flag is 16
                                if len(text_span) < 200:  # Reasonable heading length
                                    headings.append(
                                        {
                                            "page": page_num,
                                            "text": normalize_text(text_span),
                                            "size": font_size,
                                        }
                                    )

        page_count = len(doc)
        doc.close()

        return "\n".join(full_text), {
            "page_count": page_count,
            "page_texts": page_texts,
            "headings": headings,
        }

    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        # Fallback to basic extraction
        try:
            # Try with pdfminer as fallback
            from pdfminer.high_level import extract_text

            # pdfminer expects a file path or a file-like object
            text = extract_text(BytesIO(pdf_bytes))
            return normalize_text(text), {"page_count": 0, "page_texts": [], "headings": []}
        except Exception as fallback_error:
            logger.error(f"Fallback PDF extraction also failed: {fallback_error}")
            return "", {"page_count": 0, "page_texts": [], "headings": []}


def parse_pdf(page: CrawledPage) -> CrawledPage:
    """Parse PDF page and extract content."""
    try:
        text, metadata = extract_pdf_text(page.raw_content)

        # Extract title from first page or filename
        title = page.title
        if title == "Untitled" or not title:
            # Try to extract from first heading or first line
            if metadata.get("headings"):
                title = metadata["headings"][0]["text"]
            else:
                # Use filename or URL
                url_str = str(page.url)
                title = url_str.split("/")[-1].replace(".pdf", "").replace("_", " ").title()

        page.title = title
        page.cleaned_text = text

        # Store PDF metadata (can be accessed during chunking)
        # For now, we'll store page numbers in chunk model

        return page

    except Exception as e:
        logger.error(f"Error parsing PDF for {page.url}: {e}")
        page.cleaned_text = ""
        return page


