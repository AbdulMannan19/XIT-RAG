from typing import Optional

import numpy as np

from models import ChatResponse, Source
from services.rag_services.retrieval_service import TOP_K, TOP_N, SIMILARITY_CUTOFF

COLLECTION_NAME = "irs_rag_v1"
NO_KB_MSG = "I don't have verifiable information in the knowledge base for that query."


class QueryHandler:
    def __init__(
        self,
        embedding_service,
        llm_service,
        retrieval_service,
    ):
        self.embedding_provider = embedding_service
        self.llm = llm_service
        self.retrieval_service = retrieval_service
        self.collection_name = COLLECTION_NAME

    def handle_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        cutoff: Optional[float] = None,
    ):
        try:
            query_embedding = self.embedding_provider.get_embedding(query)

            top_k = top_k or TOP_K
            top_n = top_n or TOP_N
            cutoff = cutoff or SIMILARITY_CUTOFF

            chunks = self.retrieval_service.retrieve(
                self.collection_name,
                query_embedding,
                top_k,
                cutoff,
                filters,
            )

            if not chunks:
                return ChatResponse(
                    answer_text=NO_KB_MSG,
                    sources=[],
                    confidence="low",
                    query_embedding_similarity=[],
                )

            if len(chunks) > top_n:
                chunks = self.retrieval_service.rerank(query, chunks, top_n)
            else:
                chunks = chunks[:top_n]

            prompt = self.llm.build_rag_prompt(chunks, query)
            answer_text = self.llm.generate(prompt, temperature=0.0, max_tokens=200)

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

            source_models = [
                Source(
                    url=src["url"],
                    title=src["title"],
                    section=src.get("section"),
                    snippet=src.get("snippet", "")[:300],
                    char_start=src.get("char_start", 0),
                    char_end=src.get("char_end", 0),
                    score=min(max(src.get("score", 0.0), 0.0), 1.0),
                )
                for src in sources
            ]

            response = ChatResponse(
                answer_text=answer_text,
                sources=source_models,
                confidence=confidence,
                query_embedding_similarity=similarities,
            )

            return response

        except Exception as e:
            return ChatResponse(
                answer_text=NO_KB_MSG,
                sources=[],
                confidence="low",
                query_embedding_similarity=[],
            )



