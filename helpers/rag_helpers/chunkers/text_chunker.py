import uuid

from .chunking_helpers import detect_sections, chunk_by_sections, chunk_by_sliding_window
from models import Chunk, ContentType, CrawledPage


def chunk_page(page: CrawledPage, chunk_order_start: int = 0) -> list[Chunk]:
    
    text = page.cleaned_text

    if not text or len(text.strip()) < 100:
        return []

    sections = detect_sections(text)
    if sections:
        chunk_ranges = chunk_by_sections(text, sections)
    else:
        chunk_ranges = chunk_by_sliding_window(text)

    if not chunk_ranges:
        return []

    chunks = []
    for i, (char_start, char_end, section_heading) in enumerate(chunk_ranges):
        chunk_text = text[char_start:char_end].strip()

        if len(chunk_text) < 50:
            continue

        chunk_id_data = f"{page.url}{char_start}{char_end}{chunk_text[:100]}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id_data))

        raw_html_snippet = None
        if page.content_type == ContentType.HTML:
            raw_html_snippet = chunk_text[:200]

        chunk = Chunk(
            chunk_id=chunk_id,
            page_url=page.url,
            chunk_text=chunk_text,
            chunk_order=chunk_order_start + i,
            section_heading=section_heading,
            char_offset_start=char_start,
            char_offset_end=char_end,
            crawl_timestamp=page.crawl_timestamp,
            content_type=page.content_type,
            raw_html_snippet=raw_html_snippet,
            page_number=None,
        )

        chunks.append(chunk)

    return chunks
