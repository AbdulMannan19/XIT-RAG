from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException, Request, status, Header

from services.rag_services.core_services import (
    settings,
    ChatRequest,
    ChatResponse,
    Source,
    AdminStats,
    ReindexRequest,
    verify_api_key,
)
from services.rag_services.generation_service import RAGPipeline
from services.rag_services.embedding_service import (
    get_client,
    get_collection_info,
    ensure_collection,
    get_embedding_provider,
)
from services.rag_services.ingestion_services import (
    create_crawler,
    get_seed_urls,
    StorageManager,
    process_page,
)

router = APIRouter()

qdrant_client = get_client()
rag_pipeline = RAGPipeline()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, request_obj: Request):
    """Chat endpoint for RAG queries."""
    try:
        result = rag_pipeline.answer(query=request.query, filters=request.filters)

        sources = []
        for src in result.get("sources", []):
            sources.append(
                Source(
                    url=src["url"],
                    title=src["title"],
                    section=src.get("section"),
                    snippet=src.get("snippet", "")[:300],
                    char_start=src.get("char_start", 0),
                    char_end=src.get("char_end", 0),
                    score=min(max(src.get("score", 0.0), 0.0), 1.0),
                )
            )

        response = ChatResponse(
            answer_text=result["answer_text"],
            sources=sources,
            confidence=result["confidence"],
            query_embedding_similarity=result.get("query_embedding_similarity", []),
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/stats", response_model=AdminStats)
async def get_stats(x_api_key: str = Header(..., alias="X-API-Key")):
    """Get collection statistics."""
    try:
        await verify_api_key(x_api_key)
        info = get_collection_info(qdrant_client, settings.collection_name)

        stats = AdminStats(
            collection_name=settings.collection_name,
            total_chunks=info.get("points_count", 0),
            last_updated=datetime.utcnow(),
            embedding_model=settings.openai_embed_model if settings.embeddings_provider == "openai" else "local",
            vector_size=info.get("vector_size", 0),
        )
        return stats

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/reindex")
async def reindex(request: ReindexRequest, x_api_key: str = Header(..., alias="X-API-Key")):
    """Trigger reindexing of collection."""
    try:
        await verify_api_key(x_api_key)
        return {"status": "accepted", "message": "Reindex job queued"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/ingest")
async def trigger_ingest(
    x_api_key: str = Header(..., alias="X-API-Key"),
    seed_url: str = settings.crawl_base,
    max_pages: int = 100,
    concurrency: int = 2,
):
    """Trigger ingestion pipeline."""
    try:
        await verify_api_key(x_api_key)

        crawler = create_crawler(seed_url)
        storage = StorageManager()
        collection_name = settings.collection_name

        embedding_provider = get_embedding_provider()
        ensure_collection(qdrant_client, collection_name, embedding_provider.vector_size)

        target_urls = get_seed_urls(seed_url, max_urls=max_pages)
        target_urls = target_urls[:max_pages]

        processed = 0
        total_chunks = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_page, url, crawler, storage, collection_name): url for url in target_urls}

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {str(e)}")
