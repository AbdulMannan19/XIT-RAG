from concurrent.futures import ThreadPoolExecutor, as_completed

from services.core_services import settings
from services.rag_services.embedding_service import get_embedding_provider, ensure_collection
from services.rag_services.ingestion_service import process_page
from helpers.rag_helpers import WebCrawler, SitemapFetcher, StorageManager


def handle_ingestion(seed_url: str, max_pages: int, concurrency: int, qdrant_client):
    crawler = WebCrawler(base_url=seed_url, rate_limit_rps=settings.rate_limit_rps)
    crawler._check_robots_txt()
    storage = StorageManager()
    collection_name = settings.collection_name

    embedding_provider = get_embedding_provider()
    ensure_collection(qdrant_client, collection_name, embedding_provider.vector_size)

    target_urls = SitemapFetcher().get_seed_urls(seed_url, max_urls=max_pages)
    target_urls = target_urls[:max_pages]

    processed = 0
    total_chunks = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_page, url, crawler, storage, collection_name): url
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
