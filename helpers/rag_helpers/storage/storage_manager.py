from datetime import datetime
from pathlib import Path

import orjson

from models import Chunk, CrawledPage
from utils import compute_content_hash


class StorageManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.clean_dir = self.base_dir / "clean"
        self.chunks_dir = self.base_dir / "chunks"

        for dir_path in [self.raw_dir, self.clean_dir, self.chunks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_raw_page(self, page: CrawledPage) -> str:
        page.content_hash = compute_content_hash(page.raw_content)

        url_hash = compute_content_hash(str(page.url))[:16]
        filename = f"{url_hash}.jsonl"

        filepath = self.raw_dir / filename

        page_dict = {
            "url": str(page.url),
            "title": page.title,
            "crawl_timestamp": page.crawl_timestamp.isoformat(),
            "last_modified": page.last_modified.isoformat() if page.last_modified else None,
            "content_type": page.content_type.value,
            "content_hash": page.content_hash,
            "etag": page.etag,
            "status_code": page.status_code,
            "raw_content_size": len(page.raw_content),
        }

        if len(page.raw_content) > 100000:
            content_file = self.raw_dir / f"{url_hash}.content"
            content_file.write_bytes(page.raw_content)
            page_dict["content_file"] = str(content_file)

        with open(filepath, "wb") as f:
            f.write(orjson.dumps(page_dict) + b"\n")

        return str(filepath)

    def save_cleaned_page(self, page: CrawledPage) -> str:
        url_hash = compute_content_hash(str(page.url))[:16]
        filename = f"{url_hash}.jsonl"

        filepath = self.clean_dir / filename

        page_dict = {
            "url": str(page.url),
            "title": page.title,
            "crawl_timestamp": page.crawl_timestamp.isoformat(),
            "last_modified": page.last_modified.isoformat() if page.last_modified else None,
            "content_type": page.content_type.value,
            "content_hash": page.content_hash,
            "cleaned_text": page.cleaned_text,
            "text_length": len(page.cleaned_text),
        }

        with open(filepath, "wb") as f:
            f.write(orjson.dumps(page_dict) + b"\n")

        return str(filepath)

    def save_chunks(self, chunks: list[Chunk], page_url: str) -> str:
        url_hash = compute_content_hash(page_url)[:16]
        filename = f"{url_hash}_chunks.jsonl"

        filepath = self.chunks_dir / filename

        with open(filepath, "wb") as f:
            for chunk in chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "page_url": str(chunk.page_url),
                    "chunk_text": chunk.chunk_text,
                    "chunk_order": chunk.chunk_order,
                    "section_heading": chunk.section_heading,
                    "char_offset_start": chunk.char_offset_start,
                    "char_offset_end": chunk.char_offset_end,
                    "crawl_timestamp": chunk.crawl_timestamp.isoformat(),
                    "content_type": chunk.content_type.value,
                    "raw_html_snippet": chunk.raw_html_snippet,
                    "page_number": chunk.page_number,
                }
                f.write(orjson.dumps(chunk_dict) + b"\n")

        return str(filepath)

    def load_chunks(self, page_url: str) -> list[Chunk]:
        url_hash = compute_content_hash(page_url)[:16]
        filename = f"{url_hash}_chunks.jsonl"
        filepath = self.chunks_dir / filename

        if not filepath.exists():
            return []

        chunks = []
        with open(filepath, "rb") as f:
            for line in f:
                chunk_dict = orjson.loads(line)
                chunk = Chunk(
                    chunk_id=chunk_dict["chunk_id"],
                    page_url=chunk_dict["page_url"],
                    chunk_text=chunk_dict["chunk_text"],
                    chunk_order=chunk_dict["chunk_order"],
                    section_heading=chunk_dict.get("section_heading"),
                    char_offset_start=chunk_dict["char_offset_start"],
                    char_offset_end=chunk_dict["char_offset_end"],
                    crawl_timestamp=datetime.fromisoformat(chunk_dict["crawl_timestamp"]),
                    content_type=chunk_dict["content_type"],
                    raw_html_snippet=chunk_dict.get("raw_html_snippet"),
                    page_number=chunk_dict.get("page_number"),
                )
                chunks.append(chunk)

        return chunks
