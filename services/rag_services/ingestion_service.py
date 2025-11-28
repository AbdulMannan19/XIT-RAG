from qdrant_client.models import PointStruct

from services.core_services import settings, compute_content_hash
from services.rag_services.embedding_service import get_embedding_provider
from services.rag_services.qdrant_service import get_client as get_qdrant_client
from helpers.rag_helpers import TextChunker, HtmlParser, PdfParser, StorageManager


def upsert_chunks_to_qdrant(chunks: list, embeddings: list, collection_name: str):
    client = get_qdrant_client()

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

    client.upsert(collection_name=collection_name, points=points)


def process_page(url: str, crawler, storage: StorageManager, collection_name: str):
    try:
        page = crawler.fetch(url)
        if not page:
            return None

        page.content_hash = compute_content_hash(page.raw_content)
        storage.save_raw_page(page)

        if page.content_type.value == "pdf":
            page = PdfParser().parse(page)
        else:
            page = HtmlParser().parse(page)

        storage.save_cleaned_page(page)

        chunks = TextChunker().chunk(page)
        if not chunks:
            return None

        storage.save_chunks(chunks, str(page.url))

        embedding_provider = get_embedding_provider()
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        embeddings = embedding_provider.get_embeddings(chunk_texts)

        upsert_chunks_to_qdrant(chunks, embeddings, collection_name)

        return len(chunks)

    except Exception as e:
        return None
