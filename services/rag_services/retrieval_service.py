from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SIMILARITY_CUTOFF = 0.22
TOP_K = 20
TOP_N = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalService:
    def __init__(
        self,
        qdrant_service,
        reranker_model: str = RERANKER_MODEL,
        batch_size: int = 8,
        max_workers: int = 3,
    ):
        self.qdrant_service = qdrant_service
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
        self.model.to(self.device)
        self.model.eval()

    def retrieve(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: int,
        cutoff: float,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
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

        hits = self.qdrant_service.search(
            collection_name=collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
            score_threshold=cutoff,
            query_filter=query_filter,
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

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        all_scored = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for batch in batches:
                pairs = [(query, chunk.get("text", "")) for chunk in batch]
                
                def score_batch(pairs_batch, batch_chunks):
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            pairs_batch,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(self.device)

                        logits = self.model(**inputs).logits.squeeze().cpu()
                        scores = logits.tolist() if len(batch_chunks) > 1 else [logits.item()]

                    return list(zip(batch_chunks, scores))
                
                future = executor.submit(score_batch, pairs, batch)
                futures.append(future)
            
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



