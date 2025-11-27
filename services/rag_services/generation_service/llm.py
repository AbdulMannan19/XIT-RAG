"""LLM provider abstraction."""

from typing import Any, Optional

from openai import OpenAI

from services.core_services import settings


class LLMProvider:
    """Abstract LLM provider."""

    def __init__(self):
        self.model_name = ""

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate text from prompt."""
        raise NotImplementedError


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.openai_chat_model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate text using OpenAI API."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 500),
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider."""

    def __init__(self):
        super().__init__()
        import httpx

        self.client = httpx.Client(base_url=settings.ollama_host, timeout=120.0)
        self.model_name = settings.ollama_model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate text using Ollama API."""
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


def get_llm_provider() -> LLMProvider:
    """Get configured LLM provider."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not set")
        return OpenAILLMProvider()
    elif settings.llm_provider == "ollama":
        return OllamaLLMProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


