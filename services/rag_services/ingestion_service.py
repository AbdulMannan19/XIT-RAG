from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from services.core_services import settings, compute_content_hash
from services.rag_services.embedding_service import get_embedding_provider
from services.rag_services.qdrant_service import get_client as get_qdrant_client
from helpers.rag_helpers import TextChunker, HtmlParser, PdfParser, StorageManager


class IngestionService:
    def __init__(self, client: QdrantClient = None):
        self.client = client or get_qdrant_client()
        self.embedding_provider = get_embedding_provider()
        self.html_parser = HtmlParser()
        self.pdf_parser = PdfParser()
        self.text_chunker = TextChunker()

    def upsert_chunks_to_qdrant(self, chunks: list, embeddings: list, collection_name: str):
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding.tolist(),
                payload={
                    "url": str(chunk.page_url),
                    "title": chunk.chunk_text[:100] if chunk.chunk_text else "",
                    "section_heading": chunk.section_heading,
                    "text": chunk.chunk_text,
                    "char_start": chunk.char_offset_start,
                    "char_end": chunk.char_offset_end,
                    "content_type": chunk.content_type.value,
                    "crawl_ts": chunk.crawl_timestamp.isoformat(),
                    "language": "en",
                    "embedding_model": settings.openai_embed_model if settings.embeddings_provider == "openai" else "local",
                    "tokens": len(chunk.chunk_text) // 4,
                    "hash": compute_content_hash(chunk.chunk_text),
                },
            )
            points.append(point)

        self.client.upsert(collection_name=collection_name, points=points)

    def process_page(self, url: str, crawler, storage: StorageManager, collection_name: str):
        try:
            page = crawler.fetch(url)
            if not page:
                return None

            page.content_hash = compute_content_hash(page.raw_content)
            storage.save_raw_page(page)

            if page.content_type.value == "pdf":
                page = self.pdf_parser.parse(page)
            else:
                page = self.html_parser.parse(page)

            storage.save_cleaned_page(page)

            chunks = self.text_chunker.chunk(page)
            if not chunks:
                return None

            storage.save_chunks(chunks, str(page.url))

            chunk_texts = [chunk.chunk_text for chunk in chunks]
            embeddings = self.embedding_provider.get_embeddings(chunk_texts)

            self.upsert_chunks_to_qdrant(chunks, embeddings, collection_name)

            return len(chunks)

        except Exception as e:
            return None


_ingestion_service_cache = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service_cache
    if _ingestion_service_cache is None:
        _ingestion_service_cache = IngestionService()
    return _ingestion_service_cache


def process_page(url: str, crawler, storage: StorageManager, collection_name: str):
    """Backward compatibility wrapper."""
    service = get_ingestion_service()
    return service.process_page(url, crawler, storage, collection_name)
