"""In-memory vector index — brute-force cosine similarity over NumPy.

At ~260-520 chunks a dedicated vector database (FAISS/Chroma) is unnecessary: the whole
search is one matrix-vector product plus an argsort, which is exact and sub-millisecond
at this scale. This keeps the system easy to reason about and explain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray):
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunks and matrix row count must match")
        self.chunks = chunks
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)  # (n, dim), normalized

    def __len__(self) -> int:
        return len(self.chunks)

    def cosine_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Return the (n,) vector of cosine similarities for every chunk.

        Vectors are pre-normalized, so the dot product equals cosine similarity.
        """
        if len(self.chunks) == 0:
            return np.zeros((0,), dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        # errstate guards against spurious FP warnings some BLAS backends (e.g. macOS
        # Accelerate) raise on matmul even though inputs are finite and results correct.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self.matrix @ query  # (n,)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` highest scores, best first.

    Works on any precomputed score array, so callers that already have the cosine
    vector (e.g. the retriever) don't recompute it.
    """
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return np.empty((0,), dtype=np.intp)
    k = min(k, n)
    # argpartition for the top-k, then sort just those k by score descending.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return top_idx[np.argsort(-scores[top_idx])]
