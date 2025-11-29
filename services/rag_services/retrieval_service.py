from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.core_services import settings


class RetrievalService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def retrieve(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: int = 30,
        cutoff: float = 0.22,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        try:
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key == "content_type":
                        conditions.append(
                            FieldCondition(key="content_type", match=MatchValue(value=value))
                        )
                if conditions:
                    query_filter = Filter(must=conditions)

            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
                score_threshold=cutoff,
            )

            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "url": payload.get("url", ""),
                        "title": payload.get("title", ""),
                        "section_heading": payload.get("section_heading"),
                        "text": payload.get("text", ""),
                        "char_start": payload.get("char_start", 0),
                        "char_end": payload.get("char_end", 0),
                        "content_type": payload.get("content_type", "html"),
                        "crawl_ts": payload.get("crawl_ts", ""),
                        "last_modified": payload.get("last_modified"),
                        "embedding_model": payload.get("embedding_model", ""),
                    }
                )

            return results

        except Exception as e:
            return []

    def retrieve_with_cutoff(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: Optional[int] = None,
        cutoff: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if top_k is None:
            top_k = settings.top_k
        if cutoff is None:
            cutoff = settings.similarity_cutoff

        return self.retrieve(collection, query_vec, top_k=top_k, cutoff=cutoff, filters=filters)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.model = None
            self.tokenizer = None

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not self.model or not self.tokenizer:
            return chunks[:top_n]

        try:
            pairs = [(query, chunk.get("text", "")) for chunk in chunks]

            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs).logits.squeeze().cpu()
                scores = logits.tolist() if len(chunks) > 1 else [logits.item()]

            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for chunk, score in scored_chunks[:top_n]:
                chunk["rerank_score"] = float(score)
                chunk["score"] = float(score)
                reranked.append(chunk)

            return reranked

        except Exception as e:
            return chunks[:top_n]


class ParallelCrossEncoderReranker:
    def __init__(
        self,
        base_reranker: CrossEncoderReranker,
        batch_size: int = 8,
        max_workers: int = 3,
    ):
        self.base_reranker = base_reranker
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _process_batch(self, query: str, batch: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
        if not self.base_reranker.model or not self.base_reranker.tokenizer:
            return [(chunk, 0.0) for chunk in batch]

        try:
            pairs = [(query, chunk.get("text", "")) for chunk in batch]

            with torch.no_grad():
                inputs = self.base_reranker.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.base_reranker.device)

                logits = self.base_reranker.model(**inputs).logits.squeeze().cpu()
                scores = logits.tolist() if len(batch) > 1 else [logits.item()]

            return list(zip(batch, scores))

        except Exception as e:
            return [(chunk, 0.0) for chunk in batch]

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not self.base_reranker.model or not self.base_reranker.tokenizer:
            return chunks[:top_n]

        try:
            batches = [
                chunks[i : i + self.batch_size]
                for i in range(0, len(chunks), self.batch_size)
            ]

            all_scored = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_batch, query, batch)
                    for batch in batches
                ]
                for future in futures:
                    batch_results = future.result()
                    all_scored.extend(batch_results)

            all_scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for chunk, score in all_scored[:top_n]:
                chunk["rerank_score"] = float(score)
                chunk["score"] = float(score)
                reranked.append(chunk)

            return reranked

        except Exception as e:
            return chunks[:top_n]


_retrieval_service_cache = None
_reranker_cache: Optional[CrossEncoderReranker] = None
_parallel_reranker_cache: Optional[ParallelCrossEncoderReranker] = None


def get_retrieval_service(client: Optional[QdrantClient] = None) -> RetrievalService:
    global _retrieval_service_cache
    if _retrieval_service_cache is None:
        from services.rag_services.qdrant_service import get_client
        client = client or get_client()
        _retrieval_service_cache = RetrievalService(client)
    return _retrieval_service_cache


def retrieve_with_cutoff(
    client: QdrantClient,
    collection: str,
    query_vec: np.ndarray,
    top_k: Optional[int] = None,
    cutoff: Optional[float] = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Backward compatibility wrapper."""
    service = get_retrieval_service(client)
    return service.retrieve_with_cutoff(collection, query_vec, top_k, cutoff, filters)


def get_reranker(model_name: Optional[str] = None) -> Optional[CrossEncoderReranker]:
    global _reranker_cache
    
    if not settings.enable_reranking:
        return None
    
    if _reranker_cache is not None:
        return _reranker_cache
    
    if model_name is None:
        model_name = settings.reranker_model

    try:
        _reranker_cache = CrossEncoderReranker(model_name)
        return _reranker_cache
    except Exception as e:
        return None


def get_parallel_reranker(
    base_reranker: Optional[CrossEncoderReranker] = None,
    batch_size: int = 8,
    max_workers: int = 3,
) -> Optional[ParallelCrossEncoderReranker]:
    global _parallel_reranker_cache
    
    if not settings.enable_reranking:
        return None
    
    if _parallel_reranker_cache is not None:
        return _parallel_reranker_cache
    
    if base_reranker is None:
        base_reranker = get_reranker()
    
    if base_reranker is None:
        return None
    
    try:
        _parallel_reranker_cache = ParallelCrossEncoderReranker(
            base_reranker, batch_size=batch_size, max_workers=max_workers
        )
        return _parallel_reranker_cache
    except Exception as e:
        return None
