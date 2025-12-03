from qdrant_client.models import PointStruct

from utils import compute_content_hash

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IngestionService:
    def __init__(self, vector_db_service, embedding_model: str = EMBEDDING_MODEL):
        self.vector_db_service = vector_db_service
        self.embedding_model = embedding_model

    def upsert_chunks(self, chunks: list, embeddings: list, collection_name: str):

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding.tolist(),
                payload={
                    "url": str(chunk.page_url),
                    "title": chunk.chunk_text[:100] if chunk.chunk_text else "",
                    "section_heading": chunk.section_heading,
                    "text": chunk.chunk_text,
                    "char_start": chunk.char_offset_start,
                    "char_end": chunk.char_offset_end,
                    "content_type": chunk.content_type.value,
                    "crawl_ts": chunk.crawl_timestamp.isoformat(),
                    "language": "en",
                    "embedding_model": self.embedding_model,
                    "tokens": len(chunk.chunk_text) // 4,
                    "hash": compute_content_hash(chunk.chunk_text),
                },
            )
            points.append(point)

        self.vector_db_service.client.upsert(collection_name=collection_name, points=points)
