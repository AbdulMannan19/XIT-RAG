from concurrent.futures import ThreadPoolExecutor, as_completed

from models import IngestionRequest
from helpers.rag_helpers.crawlers import WebCrawler, SitemapFetcher
from helpers.rag_helpers.storage import StorageManager
from helpers.rag_helpers.parsers import HtmlParser, PdfParser
from helpers.rag_helpers.chunkers import chunk_page
from utils import compute_content_hash

COLLECTION_NAME = "irs_rag_v1"
RATE_LIMIT_RPS = 0.5


class IngestionHandler:
    def __init__(self, embedding_service, qdrant_service, ingestion_service):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.ingestion_service = ingestion_service
        self.collection_name = COLLECTION_NAME
        self.storage = StorageManager()
        self.html_parser = HtmlParser()
        self.pdf_parser = PdfParser()

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

    def _process_page(self, url: str, crawler, collection_name: str):
        
        try:
            page = crawler.fetch(url)
            if not page:
                return None

            page.content_hash = compute_content_hash(page.raw_content)
            self.storage.save_raw_page(page)

            if page.content_type.value == "pdf":
                page = self.pdf_parser.parse(page)
            else:
                page = self.html_parser.parse(page)

            self.storage.save_cleaned_page(page)

            chunks = chunk_page(page)
            if not chunks:
                return None

            self.storage.save_chunks(chunks, str(page.url))

            chunk_texts = [chunk.chunk_text for chunk in chunks]
            embeddings = self.embedding_service.get_embedding(chunk_texts)

            self.ingestion_service.upsert_chunks(chunks, embeddings, collection_name)

            return len(chunks)

        except Exception as e:
            return None

    def handle_ingestion(self, request: IngestionRequest) -> dict:
        crawler = WebCrawler(
            base_url=request.seed_url, rate_limit_rps=RATE_LIMIT_RPS
        )
        crawler._check_robots_txt()

        self.qdrant_service.ensure_collection(
            self.collection_name,
            self.embedding_service.vector_size,
        )

        target_urls = self._get_target_urls(request)

        processed = 0
        total_chunks = 0

        with ThreadPoolExecutor(max_workers=request.concurrency) as executor:
            futures = {
                executor.submit(
                    self._process_page, url, crawler, self.collection_name
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

