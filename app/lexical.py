"""Lexical (BM25) index for hybrid retrieval.

Dense embeddings capture meaning but can miss exact tokens — proper names, rare
keywords — when a chunk's overall topic dominates the vector. BM25 scores exact term
overlap, so fusing it with dense retrieval recovers those cases (e.g. "Who is Annie
Gray?"). See DESIGN.md for the fusion rationale.
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; keep tokens of length >= 2."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


class BM25Index:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self._corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def scores(self, query: str) -> np.ndarray:
        """Return the (n,) array of BM25 scores for the query over all documents."""
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
