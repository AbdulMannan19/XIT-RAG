from typing import Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
    Filter,
)

HNSW_M = 64
HNSW_EF_CONSTRUCTION = 128

class QdrantService:
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
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

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        score_threshold: float,
        query_filter: Optional[Filter] = None,
    ) -> list[Any]:
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

