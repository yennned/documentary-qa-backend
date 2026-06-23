"""Retriever behavior tests around scope gating and exact-token fallback."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.embeddings import EmbeddingClient
from app.retriever import Retriever


def test_in_scope_bread_question_survives_scope_gate():
    result = Retriever(Settings()).retrieve("What did the Victorians add to bread?")
    assert result.in_scope is True
    assert 2 <= len(result.hits) <= 3


def test_out_of_scope_question_still_abstains():
    result = Retriever(Settings()).retrieve("What is the capital of Australia?")
    assert result.in_scope is False
    assert result.hits == []


@pytest.fixture
def fallback_retriever(monkeypatch):
    monkeypatch.setattr(EmbeddingClient, "_sentence_transformer", lambda self: False)
    return Retriever(Settings())


def test_exact_name_query_survives_hash_fallback_scope_gate(fallback_retriever):
    result = fallback_retriever.retrieve("Who is Annie Gray?")
    assert result.in_scope is True
    assert 2 <= len(result.hits) <= 3


def test_topical_off_topic_question_is_blocked_in_hash_fallback(fallback_retriever):
    result = fallback_retriever.retrieve("Why did Victorians eat bread in Australia?")
    assert result.in_scope is False
    assert result.hits == []
