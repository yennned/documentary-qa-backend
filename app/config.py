"""Central configuration.

Everything that a reviewer might want to change — which LLM provider to use, which
models, retrieval knobs — is an environment variable read here. No value is hard-coded
in business logic. See ``.env.example`` for the documented list.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Provider registry -------------------------------------------------------
# Each provider speaks the OpenAI-compatible API, so the only things that change
# between them are the base URL, which env var holds the key, and the default
# model id. This table is *just defaults*: base_url / model / key can all be
# overridden by LLM_BASE_URL / LLM_MODEL / LLM_API_KEY for forward-compat, since
# hosted model ids drift over time.
PROVIDERS: dict[str, dict[str, str]] = {
    # Local, fully offline, no key required (api_key is sent but ignored by Ollama).
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": "", "model": "llama3.1:8b"},
    # Chosen open-source showcase model (open-weight, Modified MIT).
    "kimi": {"base_url": "https://api.moonshot.ai/v1", "key_env": "MOONSHOT_API_KEY", "model": "kimi-k2-0905-preview"},
    # Other open-weight options — swap in by changing LLM_PROVIDER only.
    "glm": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "key_env": "ZHIPU_API_KEY", "model": "glm-4.6"},
    "minimax": {"base_url": "https://api.minimaxi.com/v1", "key_env": "MINIMAX_API_KEY", "model": "MiniMax-M2"},
    # Free-tier hosted fallbacks.
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY", "model": "gemini-2.0-flash"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    llm_provider: str = "ollama"
    # Optional overrides (win over the PROVIDERS registry defaults).
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    # Per-provider API keys (picked up by name from the registry's key_env).
    moonshot_api_key: str | None = None
    zhipu_api_key: str | None = None
    minimax_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(700, ge=1)
    request_timeout: float = Field(25.0, gt=0.0, le=29.0)

    # --- Embeddings ---
    # "local" = sentence-transformers (offline, no Ollama needed; default so dev/tests
    # work with zero external services). "ollama" = nomic-embed-text via Ollama.
    embed_backend: str = "local"
    embed_model: str | None = None  # resolved by backend if unset
    ollama_base_url: str = "http://localhost:11434/v1"
    # Local chat model for the ollama provider. Single source of truth: the SAME
    # OLLAMA_MODEL the compose entrypoint pulls and the healthcheck waits for, so the api
    # always requests the model that was actually pulled (LLM_MODEL still overrides).
    ollama_model: str = "llama3.1:8b"

    # --- Retrieval ---
    chunk_segments: int = Field(2, ge=1)   # transcript segments grouped per chunk
    chunk_stride: int = Field(1, ge=1)     # step between chunk starts (stride<segments => overlap)
    top_k: int = Field(8, ge=2)            # candidates pulled from the index
    final_k: int = Field(3, ge=2, le=3)    # sources kept for the answer
    # Cosine cutoff; below => treated as out-of-scope. 0.20 is calibrated for the local
    # MiniLM default (clearly in-scope questions score >=0.26, clearly off-topic <0.18).
    # It is a coarse pre-filter only — the LLM abstention prompt is the primary grounding
    # guard. Re-tune if you switch embedding models (e.g. nomic produces higher cosines).
    score_threshold: float = Field(0.20, ge=0.0, le=1.0)
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    transcript_path: str = "data/transcript.txt"

    @model_validator(mode="after")
    def _validate_retrieval_knobs(self) -> "Settings":
        if self.top_k < self.final_k:
            raise ValueError("TOP_K must be >= FINAL_K so retrieval can return all requested sources")
        if self.chunk_stride > self.chunk_segments:
            raise ValueError("CHUNK_STRIDE must be <= CHUNK_SEGMENTS to avoid dropping transcript segments")
        return self

    # --- Resolved provider helpers ---
    def resolved_embed_model(self) -> str:
        if self.embed_model:
            return self.embed_model
        return "nomic-embed-text" if self.embed_backend == "ollama" else "sentence-transformers/all-MiniLM-L6-v2"

    def provider_config(self) -> dict[str, str]:
        """Return {base_url, api_key, model} for the active LLM provider."""
        if self.llm_provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={self.llm_provider!r}. Known: {', '.join(PROVIDERS)}"
            )
        entry = PROVIDERS[self.llm_provider]
        # For Ollama the endpoint and model are driven by OLLAMA_BASE_URL / OLLAMA_MODEL so
        # the same code works on localhost and inside docker-compose (host "ollama"), and the
        # api requests exactly the model the ollama service pulled. Other providers use the
        # registry defaults unless explicitly overridden.
        if self.llm_provider == "ollama":
            default_base, default_model = self.ollama_base_url, self.ollama_model
        else:
            default_base, default_model = entry["base_url"], entry["model"]
        base_url = self.llm_base_url or default_base
        model = self.llm_model or default_model
        # Resolve the API key: explicit override > the provider's named env field.
        key = self.llm_api_key
        if not key and entry["key_env"]:
            key = getattr(self, entry["key_env"].lower(), None)
        if not key:
            if entry["key_env"]:
                # A hosted provider was selected but its key is missing — fail loudly with
                # the exact variable name instead of sending a placeholder and getting an
                # opaque 401 back from the provider.
                raise ValueError(
                    f"LLM_PROVIDER={self.llm_provider!r} requires {entry['key_env']} to be set "
                    f"(or pass LLM_API_KEY). See .env.example."
                )
            # Ollama requires a non-empty (but ignored) key.
            key = "not-needed"
        return {"base_url": base_url, "api_key": key, "model": model}


@lru_cache
def get_settings() -> Settings:
    return Settings()
