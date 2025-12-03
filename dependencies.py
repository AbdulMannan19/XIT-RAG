import os
from functools import lru_cache
from dotenv import load_dotenv

from services import EmbeddingService, LLMService, QdrantService, RetrievalService, IngestionService
from handlers import QueryHandler, IngestionHandler, StatsHandler

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Handlers

@lru_cache()
def get_query_handler() -> QueryHandler:
    return QueryHandler(
        embedding_service=get_embedding_service(),
        llm_service=get_llm_service(),
        retrieval_service=get_retrieval_service(),
    )

@lru_cache()
def get_ingestion_handler() -> IngestionHandler:
    return IngestionHandler(
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
        ingestion_service=get_ingestion_service(),
    )

@lru_cache()
def get_stats_handler() -> StatsHandler:
    return StatsHandler(qdrant_service=get_qdrant_service())

# Services

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService(ollama_host=OLLAMA_HOST)

@lru_cache()
def get_qdrant_service() -> QdrantService:
    return QdrantService(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@lru_cache()
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_qdrant_service())

@lru_cache()
def get_ingestion_service() -> IngestionService:
    return IngestionService(vector_db_service=get_qdrant_service())
