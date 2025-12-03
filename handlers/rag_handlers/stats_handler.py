from datetime import datetime
from models import AdminStats

COLLECTION_NAME = "irs_rag_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class StatsHandler:
    def __init__(self, qdrant_service):
        self.qdrant_service = qdrant_service
        self.collection_name = COLLECTION_NAME

    def handle_stats(self) -> AdminStats:
        info = self.qdrant_service.get_collection_info(self.collection_name)
        embedding_model = EMBEDDING_MODEL

        stats = AdminStats(
            collection_name=self.collection_name,
            total_chunks=info.get("points_count", 0),
            last_updated=datetime.utcnow(),
            embedding_model=embedding_model,
            vector_size=info.get("vector_size", 0),
        )
        return stats

