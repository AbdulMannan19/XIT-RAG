import httpx
from typing import Any, Optional

from services.core_services import settings


class OllamaLLM:
    def __init__(self):
        self.client = httpx.Client(base_url=settings.ollama_host, timeout=120.0)
        self.model_name = settings.ollama_model

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


def get_llm() -> OllamaLLM:
    return OllamaLLM()
