from services.core_services import settings, NO_KB_MSG, build_rag_prompt
from services.rag_services.llm_service import get_llm
from services.rag_services.embedding_service import get_embedding_provider
from services.rag_services.qdrant_service import get_client
from services.rag_services.retrieval_service import get_reranker, retrieve_with_cutoff
import numpy as np


def handle_query(query: str, filters: dict = None, top_k: int = None, top_n: int = None, cutoff: float = None) -> dict:
    try:
        embedding_provider = get_embedding_provider()
        llm = get_llm()
        qdrant_client = get_client()
        reranker = get_reranker()
        collection_name = settings.collection_name

        query_embedding = embedding_provider.get_embedding(query)

        top_k = top_k or settings.top_k
        top_n = top_n or settings.top_n
        cutoff = cutoff or settings.similarity_cutoff

        chunks = retrieve_with_cutoff(
            qdrant_client,
            collection_name,
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

        if reranker and len(chunks) > top_n:
            print(f"⟳ Reranking {len(chunks)} → {top_n} chunks...")
            chunks = reranker.rerank(query, chunks, top_n=top_n)
            reranked_scores = [f"{c.get('score', 0):.3f}" for c in chunks[:3]]
            print(f"✓ Reranked (new scores: {reranked_scores}...)")
        else:
            chunks = chunks[:top_n]
            print(f"→ Using top {len(chunks)} chunks (no reranking)")

        prompt = build_rag_prompt(chunks, query)
        print(f"→ Generating answer with {len(chunks)} chunks...")

        system_prompt = "You are a factual assistant that answers only from the provided IRS.gov knowledge snippets."
        try:
            answer_text = llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500,
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

        return {
            "answer_text": answer_text,
            "sources": sources,
            "confidence": confidence,
            "query_embedding_similarity": similarities,
        }

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
