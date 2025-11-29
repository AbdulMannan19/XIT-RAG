import time
from typing import Optional

import numpy as np

from services.core_services import settings, NO_KB_MSG, build_rag_prompt
from services.rag_services.llm_service import get_llm
from services.rag_services.embedding_service import get_embedding_provider
from services.rag_services.qdrant_service import get_client
from services.rag_services.retrieval_service import get_parallel_reranker, retrieve_with_cutoff


class QueryHandler:
    def __init__(self, cache_ttl: int = 3600):
        self.embedding_provider = get_embedding_provider()
        self.llm = get_llm()
        self.qdrant_client = get_client()
        self.reranker = get_parallel_reranker(batch_size=8, max_workers=3)
        self.collection_name = settings.collection_name
        self._response_cache = {}
        self._cache_ttl = cache_ttl

    def _get_cache_key(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> str:
        filters_str = str(sorted(filters.items())) if filters else "none"
        return f"{query}|{filters_str}|{top_k}|{top_n}|{cutoff}"

    def handle_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> dict:
        cache_key = self._get_cache_key(query, filters, top_k, top_n, cutoff)
        if cache_key in self._response_cache:
            cached_response, cached_time = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                print(f"✓ Cache hit for query: {query[:50]}...")
                return cached_response

        try:
            query_embedding_tuple = self.embedding_provider.get_embedding_cached(query)
            query_embedding = np.array(query_embedding_tuple)

            top_k = top_k or settings.top_k
            top_n = top_n or settings.top_n
            cutoff = cutoff or settings.similarity_cutoff

            chunks = retrieve_with_cutoff(
                self.qdrant_client,
                self.collection_name,
                query_embedding,
                top_k=top_k,
                cutoff=cutoff,
                filters=filters,
            )

            if not chunks:
                print(f"❌ No chunks retrieved for query: {query[:50]}...")
                return {
                    "answer_text": NO_KB_MSG,
                    "sources": [],
                    "confidence": "low",
                    "query_embedding_similarity": [],
                }

            scores = [f"{c.get('score', 0):.3f}" for c in chunks[:3]]
            print(f"✓ Retrieved {len(chunks)} chunks (scores: {scores}...)")

            if self.reranker and len(chunks) >= 10 and len(chunks) > top_n:
                print(f"⟳ Reranking {len(chunks)} → {top_n} chunks...")
                chunks = self.reranker.rerank(query, chunks, top_n=top_n)
                reranked_scores = [f"{c.get('score', 0):.3f}" for c in chunks[:3]]
                print(f"✓ Reranked (new scores: {reranked_scores}...)")
            else:
                chunks = chunks[:top_n]
                print(f"→ Using top {len(chunks)} chunks (no reranking)")

            prompt = build_rag_prompt(chunks, query)
            print(f"→ Generating answer with {len(chunks)} chunks...")

            system_prompt = "You are a factual assistant that answers only from the provided IRS.gov knowledge snippets."
            try:
                answer_text = self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=200,
                )
                print(f"✓ Generated answer ({len(answer_text)} chars)")
            except Exception as gen_error:
                print(f"❌ Generation failed: {gen_error}")
                raise

            sources = []
            similarities = []
            for chunk in chunks:
                sources.append(
                    {
                        "url": chunk.get("url", ""),
                        "title": chunk.get("title", ""),
                        "section": chunk.get("section_heading"),
                        "snippet": chunk.get("text", "")[:300],
                        "char_start": chunk.get("char_start", 0),
                        "char_end": chunk.get("char_end", 0),
                        "score": chunk.get("score", 0.0),
                    }
                )
                similarities.append(chunk.get("score", 0.0))

            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity >= 0.8:
                confidence = "high"
            elif avg_similarity >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"

            result = {
                "answer_text": answer_text,
                "sources": sources,
                "confidence": confidence,
                "query_embedding_similarity": similarities,
            }

            self._response_cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            print(f"❌ Query handler error: {e}")
            import traceback

            traceback.print_exc()
            return {
                "answer_text": NO_KB_MSG,
                "sources": [],
                "confidence": "low",
                "query_embedding_similarity": [],
            }


_query_handler_instance = None


def get_query_handler() -> QueryHandler:
    global _query_handler_instance
    if _query_handler_instance is None:
        _query_handler_instance = QueryHandler()
    return _query_handler_instance
