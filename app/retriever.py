"""Retriever: build the index once, then answer queries with ranked chunks.

Pipeline per query:
  1. embed the question and score every chunk by cosine (dense)
  2. score every chunk by BM25 (lexical) and fuse the two rankings via Reciprocal
     Rank Fusion (RRF) — dense gives semantics, BM25 recovers exact names/keywords
  3. keep the top FINAL_K fused chunks
  4. decide scope from the best *dense* cosine vs SCORE_THRESHOLD

Out-of-scope handling has two layers: this threshold (cheap pre-filter, first layer) and
the LLM abstention prompt (primary guard, second layer). See DESIGN.md.

An optional cross-encoder reranker (USE_RERANKER=true) replaces step 2's fusion with a
rerank of the dense candidates, for higher precision at some latency cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .embeddings import EmbeddingClient
from .index import SearchHit, VectorIndex
from .ingest import load_chunks
from .lexical import BM25Index

_RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class RetrievalResult:
    hits: list[SearchHit]          # final, ranked sources
    best_score: float              # top dense cosine (used for the abstain decision)
    in_scope: bool                 # True if the question looks answerable from the transcript


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = EmbeddingClient(settings)
        chunks = load_chunks(
            settings.transcript_path,
            window=settings.chunk_segments,
            stride=settings.chunk_stride,
        )
        matrix = self.embedder.embed([c.text for c in chunks])
        self.index = VectorIndex(chunks, matrix)
        self.bm25 = BM25Index([c.text for c in chunks])
        self._reranker = None

    # --- ranking strategies ------------------------------------------------
    def _rrf_fuse(self, cosine: np.ndarray, bm25: np.ndarray, top_k: int) -> list[int]:
        """Fuse dense and lexical rankings with RRF; return fused chunk indices."""
        n = cosine.shape[0]
        k = min(top_k, n)
        dense_top = np.argsort(-cosine)[:k]
        lex_top = np.argsort(-bm25)[:k]

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank)
        for rank, idx in enumerate(lex_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank)
        return sorted(rrf, key=lambda i: rrf[i], reverse=True)

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.settings.reranker_model)
        scores = self._reranker.predict([(query, h.chunk.text) for h in hits])
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)

    # --- public API --------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        if len(self.index) == 0:
            return RetrievalResult(hits=[], best_score=0.0, in_scope=False)

        query_vec = self.embedder.embed_one(question)
        cosine = self.index.cosine_scores(query_vec)
        best_score = float(cosine.max())
        # Scope is gated on the dense cosine — the semantic signal, robust to incidental
        # lexical overlap (e.g. an off-topic question sharing one common word).
        in_scope = best_score >= self.settings.score_threshold
        if not in_scope:
            return RetrievalResult(hits=[], best_score=best_score, in_scope=False)

        if self.settings.use_reranker:
            candidates = self.index.search(query_vec, top_k=self.settings.top_k)
            ranked = self._rerank(question, candidates)
            kept = ranked[: self.settings.final_k]
        else:
            bm25 = self.bm25.scores(question)
            fused_idx = self._rrf_fuse(cosine, bm25, top_k=self.settings.top_k)
            kept = [
                SearchHit(chunk=self.index.chunks[i], score=float(cosine[i]))
                for i in fused_idx[: self.settings.final_k]
            ]

        return RetrievalResult(hits=kept, best_score=best_score, in_scope=True)
