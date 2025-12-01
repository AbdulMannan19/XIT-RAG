import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

# Qdrant Configuration (secrets from environment)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Vector DB defaults
HNSW_M = 64
HNSW_EF_CONSTRUCTION = 128


class QdrantService:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY or None
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection not in collection_names:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
            )

        try:
            self.client.update_collection(
                collection_name=collection,
                hnsw_config=HnswConfigDiff(
                    m=HNSW_M,
                    ef_construct=HNSW_EF_CONSTRUCTION,
                    full_scan_threshold=10000,
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                ),
            )
        except Exception as e:
            pass

    def get_collection_info(self, collection: str) -> dict:
        info = self.client.get_collection(collection)
        return {
            "name": collection,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count,
            "status": info.status,
        }

    def delete_collection(self, collection: str) -> None:
        try:
            self.client.delete_collection(collection)
        except Exception as e:
            raise


_qdrant_service_cache = None


def get_client() -> QdrantClient:
    """Get Qdrant client for backward compatibility."""
    service = get_qdrant_service()
    return service.client


def get_qdrant_service() -> QdrantService:
    global _qdrant_service_cache
    if _qdrant_service_cache is None:
        _qdrant_service_cache = QdrantService()
    return _qdrant_service_cache


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Backward compatibility wrapper."""
    service = get_qdrant_service()
    service.ensure_collection(collection, vector_size)


def get_collection_info(client: QdrantClient, collection: str) -> dict:
    """Backward compatibility wrapper."""
    service = get_qdrant_service()
    return service.get_collection_info(collection)
