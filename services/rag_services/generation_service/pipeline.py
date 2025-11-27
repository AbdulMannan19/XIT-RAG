"""RAG pipeline orchestration."""

from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient

from services.core_services import settings, NO_KB_MSG, build_no_results_prompt, build_rag_prompt, estimate_tokens
from services.rag_services.generation_service import get_llm_provider
from services.rag_services.embedding_service import get_embedding_provider, get_client, get_reranker, retrieve_with_cutoff


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.llm_provider = get_llm_provider()
        self.qdrant_client = get_client()
        self.reranker = get_reranker()
        self.collection_name = settings.collection_name

    def answer(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> dict[str, Any]:
        """Answer a query using RAG pipeline."""
        try:
            # Step 1: Embed query
            query_embedding = self.embedding_provider.get_embedding(query)

            # Step 2: Retrieve chunks
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

            # Step 3: Optional reranking
            if self.reranker and len(chunks) > top_n:
                print(f"⟳ Reranking {len(chunks)} → {top_n} chunks...")
                chunks = self.reranker.rerank(query, chunks, top_n=top_n)
                reranked_scores = [f"{c.get('score', 0):.3f}" for c in chunks[:3]]
                print(f"✓ Reranked (new scores: {reranked_scores}...)")
            else:
                chunks = chunks[:top_n]
                print(f"→ Using top {len(chunks)} chunks (no reranking)")

            # Step 4: Build prompt
            prompt = build_rag_prompt(chunks, query)
            print(f"→ Generating answer with {len(chunks)} chunks...")

            # Step 5: Generate answer
            system_prompt = "You are a factual assistant that answers only from the provided IRS.gov knowledge snippets."
            try:
                answer_text = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=500,
                )
                print(f"✓ Generated answer ({len(answer_text)} chars)")
            except Exception as gen_error:
                print(f"❌ Generation failed: {gen_error}")
                raise

            # Step 6: Format sources
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

            # Step 7: Determine confidence
            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity >= 0.8:
                confidence = "high"
            elif avg_similarity >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "answer_text": answer_text,
                "sources": sources,
                "confidence": confidence,
                "query_embedding_similarity": similarities,
            }

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer_text": NO_KB_MSG,
                "sources": [],
                "confidence": "low",
                "query_embedding_similarity": [],
            }


def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance."""
    return RAGPipeline()

