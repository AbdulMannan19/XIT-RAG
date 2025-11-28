from typing import Any, Optional

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.core_services import settings


def retrieve(
    client: QdrantClient,
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

        hits = client.search(
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
    client: QdrantClient,
    collection: str,
    query_vec: np.ndarray,
    top_k: int = None,
    cutoff: float = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if top_k is None:
        top_k = settings.top_k
    if cutoff is None:
        cutoff = settings.similarity_cutoff

    candidates = retrieve(client, collection, query_vec, top_k=top_k * 2, cutoff=0.0, filters=filters)
    filtered = [c for c in candidates if c["score"] >= cutoff]
    filtered = filtered[:top_k]

    return filtered


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
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

                scores = self.model(**inputs).logits.squeeze().cpu().tolist()

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


def get_reranker(model_name: Optional[str] = None) -> Optional[CrossEncoderReranker]:
    if model_name is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    try:
        return CrossEncoderReranker(model_name)
    except Exception as e:
        return None
