from datetime import datetime

from services.core_services import settings, AdminStats
from services.rag_services.qdrant_service import get_qdrant_service


class StatsHandler:
    def __init__(self):
        self.qdrant_service = get_qdrant_service()
        self.collection_name = settings.collection_name

    def handle_stats(self) -> AdminStats:
        info = self.qdrant_service.get_collection_info(self.collection_name)

        embedding_model = (
            settings.openai_embed_model
            if settings.embeddings_provider == "openai"
            else "local"
        )

        stats = AdminStats(
            collection_name=self.collection_name,
            total_chunks=info.get("points_count", 0),
            last_updated=datetime.utcnow(),
            embedding_model=embedding_model,
            vector_size=info.get("vector_size", 0),
        )
        return stats


_stats_handler_instance = None


def get_stats_handler() -> StatsHandler:
    global _stats_handler_instance
    if _stats_handler_instance is None:
        _stats_handler_instance = StatsHandler()
    return _stats_handler_instance
