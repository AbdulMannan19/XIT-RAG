from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from qdrant_client import QdrantClient

from models import IngestionRequest
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

    def _filter_urls(self, urls: list[str], request: IngestionRequest) -> list[str]:
        filtered = urls

        if request.allow_prefix:
            filtered = [
                url for url in filtered
                if any(request.allow_prefix in url for request.allow_prefix in request.allow_prefix)
            ]

        if request.only_html:
            filtered = [url for url in filtered if not url.lower().endswith('.pdf')]

        if request.only_pdf:
            filtered = [url for url in filtered if url.lower().endswith('.pdf')]

        if request.forms:
            form_patterns = [f.lower() for f in request.forms]
            filtered = [
                url for url in filtered
                if any(form in url.lower() for form in form_patterns)
            ]

        return filtered

    def _get_target_urls(self, request: IngestionRequest) -> list[str]:
        if request.url_file:
            with open(request.url_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            return self._filter_urls(urls, request)[:request.max_pages]

        sitemap_fetcher = SitemapFetcher()
        urls = sitemap_fetcher.get_seed_urls(request.seed_url, max_urls=request.max_pages * 2)

        if request.include_seed and request.seed_url not in urls:
            urls.insert(0, request.seed_url)

        filtered_urls = self._filter_urls(urls, request)
        return filtered_urls[:request.max_pages]

    def handle_ingestion(self, request: IngestionRequest) -> dict:
        crawler = WebCrawler(
            base_url=request.seed_url, rate_limit_rps=RATE_LIMIT_RPS
        )
        crawler._check_robots_txt()

        ensure_collection(
            self.qdrant_client,
            self.collection_name,
            self.embedding_provider.vector_size,
        )

        target_urls = self._get_target_urls(request)

        processed = 0
        total_chunks = 0

        with ThreadPoolExecutor(max_workers=request.concurrency) as executor:
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
            "seed_url": request.seed_url,
            "max_pages": request.max_pages,
            "concurrency": request.concurrency,
            "filters": {
                "allow_prefix": request.allow_prefix,
                "only_html": request.only_html,
                "only_pdf": request.only_pdf,
                "forms": request.forms,
            },
            "pages_processed": processed,
            "total_chunks": total_chunks,
            "target_urls_found": len(target_urls),
        }


_ingestion_handler_instance = None


def get_ingestion_handler() -> IngestionHandler:
    global _ingestion_handler_instance
    if _ingestion_handler_instance is None:
        _ingestion_handler_instance = IngestionHandler()
    return _ingestion_handler_instance
