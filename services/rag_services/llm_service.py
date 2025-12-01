import httpx
import orjson
from typing import Any, Optional

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"


class LLMService:
    def __init__(self):
        self.client = httpx.Client(base_url=OLLAMA_HOST, timeout=120.0)
        self.model_name = OLLAMA_MODEL
        self._warmup()

    def _warmup(self):
        """Warm up the model with a tiny generation to avoid cold start."""
        try:
            self.generate(
                prompt="Say: OK",
                system_prompt="Respond briefly.",
                max_tokens=3,
            )
        except Exception:
            pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.0),
                        "num_predict": kwargs.get("max_tokens", 500),
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            raise


def build_rag_prompt(chunks: list[dict[str, Any]], user_query: str) -> str:
    """Build RAG prompt with context chunks."""
    # Format chunks as JSONL
    ctx_lines = []
    for chunk in chunks:
        ctx_obj = {
            "url": chunk.get("url", ""),
            "title": chunk.get("title", ""),
            "section_heading": chunk.get("section_heading"),
            "char_start": chunk.get("char_start", 0),
            "char_end": chunk.get("char_end", 0),
            "excerpt": chunk.get("text", "")[:300],
        }
        ctx_lines.append(orjson.dumps(ctx_obj).decode("utf-8"))

    ctx_block = "\n".join(ctx_lines)

    prompt = f"""SYSTEM:
You are a factual assistant that answers only from the provided IRS.gov knowledge snippets. You must cite sources and never invent facts.

CONTEXT:
{ctx_block}

USER:
{user_query}

ASSISTANT INSTRUCTIONS:
- Use only the provided context. If it does not support an answer, say: "I don't have verifiable information in the knowledge base for that query." Offer top similar sources with excerpts.

- Respond in GitHub-Flavored Markdown (GFM).
- Begin your answer with a level-3 heading containing the user question exactly:
  "### {user_query}"
- Use concise paragraphs and bullet lists for steps and key points.
- Bold key labels or terms (e.g., **Eligibility**, **Amount**, **Deadline**).

- After the answer, include a section titled "### Sources" with bullet items in this format:
  - [<page_title>] — <section_heading if available> — <url> — excerpt: "..." (char_start–char_end)

- If sources conflict, present both and mark uncertainty.
- Include relevant IRS form numbers if present in context.

- If the question asks for legal/tax filing advice, prepend:

  "I am not a lawyer; for legal or tax-filing advice consult a qualified tax professional or the IRS."
"""

    return prompt


_llm_cache = None


def get_llm() -> LLMService:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMService()
    return _llm_cache
