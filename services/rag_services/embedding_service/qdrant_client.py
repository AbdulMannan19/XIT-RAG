"""Qdrant client and collection management."""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from services.core_services import settings, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M

logger = logging.getLogger(__name__)


def get_client(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    """Get Qdrant client instance."""
    url = url or settings.qdrant_url
    api_key = api_key or settings.qdrant_api_key or None

    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Ensure Qdrant collection exists with proper configuration."""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection not in collection_names:
        logger.info(f"Creating collection: {collection}")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        logger.info(f"Collection {collection} created with vector size {vector_size}")
    else:
        logger.info(f"Collection {collection} already exists")

    # Update HNSW configuration
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
        logger.info(f"Updated HNSW config for {collection}: M={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCTION}")
    except Exception as e:
        logger.warning(f"Could not update HNSW config: {e}")


def get_collection_info(client: QdrantClient, collection: str) -> dict:
    """Get collection information."""
    try:
        info = client.get_collection(collection)
        return {
            "name": collection,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {}


def delete_collection(client: QdrantClient, collection: str) -> None:
    """Delete a collection."""
    try:
        client.delete_collection(collection)
        logger.info(f"Deleted collection: {collection}")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise

