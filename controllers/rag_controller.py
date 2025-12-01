from fastapi import APIRouter, HTTPException, status

from models import ChatRequest, ChatResponse, Source, AdminStats
from handlers.rag_handlers import (
    get_query_handler,
    get_ingestion_handler,
    get_stats_handler,
)

# Configuration
CRAWL_BASE = "https://www.irs.gov"

router = APIRouter()

@router.post("/query", response_model=ChatResponse)
async def query(request: ChatRequest):
    """Handle RAG query requests."""
    try:
        handler = get_query_handler()
        result = handler.handle_query(query=request.query, filters=request.filters)

        sources = [
            Source(
                url=src["url"],
                title=src["title"],
                section=src.get("section"),
                snippet=src.get("snippet", "")[:300],
                char_start=src.get("char_start", 0),
                char_end=src.get("char_end", 0),
                score=min(max(src.get("score", 0.0), 0.0), 1.0),
            )
            for src in result.get("sources", [])
        ]

        return ChatResponse(
            answer_text=result["answer_text"],
            sources=sources,
            confidence=result["confidence"],
            query_embedding_similarity=result.get("query_embedding_similarity", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/stats", response_model=AdminStats)
async def get_stats():
    """Get collection statistics."""
    try:
        handler = get_stats_handler()
        return handler.handle_stats()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/ingest")
async def trigger_ingest(
    seed_url: str = CRAWL_BASE,
    max_pages: int = 100,
    concurrency: int = 2,
):
    """Trigger ingestion of pages from seed URL."""
    try:
        handler = get_ingestion_handler()
        return handler.handle_ingestion(seed_url, max_pages, concurrency)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
