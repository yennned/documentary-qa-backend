"""Provider-agnostic LLM client.

Every supported backend (Ollama, Kimi, GLM, MiniMax, Groq, Gemini) speaks the
OpenAI-compatible Chat Completions API, so a single ``openai.OpenAI`` client serves all
of them — only base_url / api_key / model change, and those come from config. Switching
providers therefore needs no code change, only environment variables.
"""
from __future__ import annotations

from collections.abc import Iterator

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.settings = settings
        cfg = settings.provider_config()
        self.model = cfg["model"]
        self.client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=settings.request_timeout,
        )

    def _create_kwargs(self, messages: list[dict[str, str]]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(**self._create_kwargs(messages))
        if not resp.choices:
            return ""
        return (resp.choices[0].message.content or "").strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield answer text token-by-token (bonus: streaming responses)."""
        stream = self.client.chat.completions.create(**self._create_kwargs(messages), stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
