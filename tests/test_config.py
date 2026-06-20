"""Configuration validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_timeout_stays_within_30_second_budget():
    settings = Settings()
    assert settings.request_timeout < 30


def test_final_k_is_limited_to_spec_range():
    with pytest.raises(ValidationError):
        Settings(final_k=1)
    with pytest.raises(ValidationError):
        Settings(final_k=4)


def test_top_k_must_cover_final_k():
    with pytest.raises(ValidationError, match="TOP_K must be >= FINAL_K"):
        Settings(top_k=2, final_k=3)


def test_chunk_stride_must_not_exceed_window():
    with pytest.raises(ValidationError, match="CHUNK_STRIDE must be <= CHUNK_SEGMENTS"):
        Settings(chunk_segments=2, chunk_stride=3)


def test_ollama_model_drives_the_requested_model():
    # OLLAMA_MODEL must be the single knob: what the api requests == what compose pulls,
    # so `OLLAMA_MODEL=llama3.2:3b docker compose up` works end-to-end.
    cfg = Settings(llm_provider="ollama", ollama_model="llama3.2:3b").provider_config()
    assert cfg["model"] == "llama3.2:3b"
    # explicit LLM_MODEL still wins
    cfg2 = Settings(llm_provider="ollama", ollama_model="llama3.2:3b", llm_model="qwen2.5:7b").provider_config()
    assert cfg2["model"] == "qwen2.5:7b"
