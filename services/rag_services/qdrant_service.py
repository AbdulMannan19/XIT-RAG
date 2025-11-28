from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from services.core_services import settings, HNSW_EF_CONSTRUCTION, HNSW_M


def get_client(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    url = url or settings.qdrant_url
    api_key = api_key or settings.qdrant_api_key or None
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection not in collection_names:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )

    try:
        client.update_collection(
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


def get_collection_info(client: QdrantClient, collection: str) -> dict:
    try:
        info = client.get_collection(collection)
        return {
            "name": collection,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        return {}


def delete_collection(client: QdrantClient, collection: str) -> None:
    try:
        client.delete_collection(collection)
    except Exception as e:
        raise
