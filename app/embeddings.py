"""Embedding client with two interchangeable backends.

- ``local``  : sentence-transformers, fully offline, no Ollama needed (default, so
               dev/tests run with zero external services).
- ``ollama`` : nomic-embed-text via Ollama's OpenAI-compatible /v1/embeddings endpoint.

Both return L2-normalized float32 vectors so that a dot product == cosine similarity.
"""
from __future__ import annotations

import numpy as np

from .config import Settings


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.embed_backend
        self.model_name = settings.resolved_embed_model()
        self._st_model = None       # lazy sentence-transformers model
        self._openai_client = None  # lazy OpenAI client for Ollama

    # --- lazy initialisers -------------------------------------------------
    def _sentence_transformer(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def _ollama(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",  # required but ignored by Ollama
                timeout=self.settings.request_timeout,
            )
        return self._openai_client

    # --- public API --------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) matrix of normalized embeddings for ``texts``."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "ollama":
            resp = self._ollama().embeddings.create(model=self.model_name, input=texts)
            vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
        else:
            model = self._sentence_transformer()
            vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors)

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single normalized (dim,) embedding vector."""
        return self.embed([text])[0]
