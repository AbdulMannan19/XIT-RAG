from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from qdrant_client import QdrantClient

from services.rag_services.embedding_service import get_embedding_provider
from services.rag_services.qdrant_service import ensure_collection, get_client
from services.rag_services.ingestion_service import process_page
from helpers.rag_helpers import WebCrawler, SitemapFetcher, StorageManager

COLLECTION_NAME = "irs_rag_v1"
RATE_LIMIT_RPS = 0.5


class IngestionHandler:
    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.qdrant_client = qdrant_client or get_client()
        self.embedding_provider = get_embedding_provider()
        self.collection_name = COLLECTION_NAME
        self.storage = StorageManager()

    def handle_ingestion(
        self, seed_url: str, max_pages: int, concurrency: int
    ) -> dict:
        crawler = WebCrawler(
            base_url=seed_url, rate_limit_rps=RATE_LIMIT_RPS
        )
        crawler._check_robots_txt()

        ensure_collection(
            self.qdrant_client,
            self.collection_name,
            self.embedding_provider.vector_size,
        )

        target_urls = SitemapFetcher().get_seed_urls(seed_url, max_urls=max_pages)
        target_urls = target_urls[:max_pages]

        processed = 0
        total_chunks = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    process_page, url, crawler, self.storage, self.collection_name
                ): url
                for url in target_urls
            }

            for future in as_completed(futures):
                try:
                    chunks_count = future.result()
                    if chunks_count:
                        processed += 1
                        total_chunks += chunks_count
                except Exception:
                    pass

        crawler.close()

        return {
            "status": "completed",
            "message": "Ingestion completed successfully",
            "seed_url": seed_url,
            "max_pages": max_pages,
            "concurrency": concurrency,
            "pages_processed": processed,
            "total_chunks": total_chunks,
        }


_ingestion_handler_instance = None


def get_ingestion_handler() -> IngestionHandler:
    global _ingestion_handler_instance
    if _ingestion_handler_instance is None:
        _ingestion_handler_instance = IngestionHandler()
    return _ingestion_handler_instance
