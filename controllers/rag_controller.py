from fastapi import APIRouter, HTTPException, status, Depends

from models import ChatRequest, ChatResponse, AdminStats, IngestionRequest
from handlers import QueryHandler, IngestionHandler, StatsHandler
from dependencies import get_query_handler, get_ingestion_handler, get_stats_handler

router = APIRouter()

@router.post("/query", response_model=ChatResponse)
async def query(
    request: ChatRequest,
    handler: QueryHandler = Depends(get_query_handler)
):
    try:
        return handler.handle_query(query=request.query, filters=request.filters)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/stats", response_model=AdminStats)
async def get_stats(handler: StatsHandler = Depends(get_stats_handler)):
    try:
        return handler.handle_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/ingest")
async def trigger_ingest(
    request: IngestionRequest,
    handler: IngestionHandler = Depends(get_ingestion_handler)
):
    try:
        return handler.handle_ingestion(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
