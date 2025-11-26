"""Simple ingestion runner that bypasses typer CLI issues."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from app.core.config import settings
from app.core.logging import setup_logging
from app.ingestion.crawler import create_crawler
from app.ingestion.sitemap import get_seed_urls
from app.ingestion.storage import StorageManager
from app.vector.embeddings import get_embedding_provider
from app.vector.qdrant_client import ensure_collection, get_client

# Import the process_page function from ingest script
import sys
sys.path.insert(0, 'app/scripts')
from app.scripts.ingest import process_page

setup_logging()
logger = logging.getLogger(__name__)


def run_ingestion(seed_url: str, max_pages: int, concurrency: int):
    """Run ingestion with specified parameters."""
    logger.info(f"Starting ingestion: seed={seed_url}, max_pages={max_pages}, concurrency={concurrency}")
    
    # Setup
    crawler = create_crawler(seed_url)
    storage = StorageManager()
    client = get_client()
    collection_name = settings.collection_name
    
    # Ensure collection exists
    embedding_provider = get_embedding_provider()
    ensure_collection(client, collection_name, embedding_provider.vector_size)
    logger.info(f"Collection {collection_name} ready with vector size {embedding_provider.vector_size}")
    
    # Get seed URLs from sitemap
    logger.info("Discovering URLs from sitemaps...")
    target_urls = get_seed_urls(seed_url, max_urls=max_pages)
    logger.info(f"Found {len(target_urls)} URLs")
    
    # Limit to max_pages
    target_urls = target_urls[:max_pages]
    
    # Process pages
    processed = 0
    total_chunks = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_page, url, crawler, storage, collection_name): url 
            for url in target_urls
        }
        
        with tqdm(total=len(target_urls), desc="Processing pages") as pbar:
            for future in as_completed(futures):
                url = futures[future]
                try:
                    chunks_count = future.result()
                    if chunks_count:
                        processed += 1
                        total_chunks += chunks_count
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    pbar.update(1)
    
    crawler.close()
    logger.info(f"Ingestion complete: {processed} pages, {total_chunks} chunks")
    return processed, total_chunks


if __name__ == "__main__":
    # Configuration
    SEED_URL = "https://www.irs.gov/"
    MAX_PAGES = 100
    CONCURRENCY = 2
    
    print(f"Starting ingestion with:")
    print(f"  Seed: {SEED_URL}")
    print(f"  Max pages: {MAX_PAGES}")
    print(f"  Concurrency: {CONCURRENCY}")
    print()
    
    processed, chunks = run_ingestion(SEED_URL, MAX_PAGES, CONCURRENCY)
    
    print()
    print(f"✅ Ingestion complete!")
    print(f"   Pages processed: {processed}")
    print(f"   Total chunks: {chunks}")
