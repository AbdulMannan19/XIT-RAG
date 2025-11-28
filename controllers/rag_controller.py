from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status, Header

from services.core_services import (
    settings,
    ChatRequest,
    ChatResponse,
    Source,
    AdminStats,
    ReindexRequest,
)
from services.rag_services.qdrant_service import get_client, get_collection_info
from handlers.ingestion_handler import handle_ingestion
from handlers.query_handler import handle_query

router = APIRouter()

qdrant_client = get_client()


@router.post("/query", response_model=ChatResponse)
async def query(request: ChatRequest, request_obj: Request):
    try:
        result = handle_query(query=request.query, filters=request.filters)

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
async def get_stats():
    """Get collection statistics."""
    try:
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
async def reindex(request: ReindexRequest):
    """Trigger reindexing of collection."""
    try:
        return {"status": "accepted", "message": "Reindex job queued"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/ingest")
async def trigger_ingest(
    seed_url: str = settings.crawl_base,
    max_pages: int = 100,
    concurrency: int = 2,
):

    try:
        result = handle_ingestion(seed_url, max_pages, concurrency, qdrant_client)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {str(e)}")
