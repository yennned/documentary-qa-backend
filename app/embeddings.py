"""Embedding client with two interchangeable backends.

- ``local``  : a cached sentence-transformers model when available, otherwise a
               deterministic zero-download fallback, so dev/tests run with zero
               external services.
- ``ollama`` : nomic-embed-text via Ollama's OpenAI-compatible /v1/embeddings endpoint.

Both return L2-normalized float32 vectors so that a dot product == cosine similarity.
"""
from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from .config import Settings

_LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_DIMS = 1536
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    # Scrub any non-finite values first: a single NaN/inf would otherwise poison cosine
    # scores (cosine.max() -> NaN, making every query look out-of-scope).
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _hash_slot(feature: str, dims: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
    slot = int.from_bytes(digest[:8], "little") % dims
    sign = 1.0 if digest[8] & 1 else -1.0
    return slot, sign


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_ngrams(token: str) -> list[str]:
    if len(token) < 3:
        return []
    padded = f" {token} "
    grams: list[str] = []
    for n in (3, 4, 5):
        if len(padded) < n:
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def _hashing_embed(texts: list[str], dims: int = _HASH_DIMS) -> np.ndarray:
    """Deterministic fallback embedder that needs no downloads or network access."""
    if not texts:
        return np.zeros((0, dims), dtype=np.float32)

    matrix = np.zeros((len(texts), dims), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _tokenize(text)
        core_tokens = [tok for tok in tokens if tok not in _STOPWORDS] or tokens

        for token in core_tokens:
            slot, sign = _hash_slot(f"tok:{token}", dims)
            matrix[row, slot] += 2.0 * sign
            for gram in _char_ngrams(token):
                slot, sign = _hash_slot(f"chr:{gram}", dims)
                matrix[row, slot] += 0.15 * sign

        for left, right in zip(core_tokens, core_tokens[1:]):
            slot, sign = _hash_slot(f"bigram:{left}_{right}", dims)
            matrix[row, slot] += 0.75 * sign

    return _normalize(matrix)


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.embed_backend
        self.model_name = settings.resolved_embed_model()
        self._st_model = None       # lazy sentence-transformers model; False => fallback
        self._openai_client = None  # lazy OpenAI client for Ollama

    # --- lazy initialisers -------------------------------------------------
    def _sentence_transformer(self):
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # heavy import, deferred

                # Stay strictly offline here: use the cache if the model is already
                # present, otherwise fall back immediately instead of attempting a
                # network download (which hangs tests and fresh offline setups).
                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception as exc:
                _LOGGER.warning(
                    "Falling back to the built-in hashing embedder because the cached "
                    "sentence-transformers model %r is unavailable: %s",
                    self.model_name,
                    exc,
                )
                self._st_model = False
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

    def uses_hash_fallback(self) -> bool:
        """Whether local embeddings are using the built-in hashing fallback."""
        if self.backend != "local":
            return False
        return self._sentence_transformer() is False

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
            if model is False:
                vectors = _hashing_embed(texts)
            else:
                vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors)

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single normalized (dim,) embedding vector."""
        return self.embed([text])[0]
