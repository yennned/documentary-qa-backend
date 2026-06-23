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
from .index import SearchHit, VectorIndex, top_k_indices
from .ingest import load_chunks
from .lexical import BM25Index, tokenize

_RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class RetrievalResult:
    hits: list[SearchHit]          # final, ranked sources (score = the value that ordered them)
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
    def _rrf_fuse(self, cosine: np.ndarray, bm25: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Fuse dense and lexical rankings with RRF.

        Returns ``(chunk_index, fused_score)`` pairs, best first, where the score is
        normalized so the top result is 1.0 — that score is what callers surface, so the
        displayed relevance always matches the ordering.

        If BM25 produced no signal at all (no query token appears in any chunk), its
        ``argsort`` would be a meaningless positional order [0,1,2,…] that injects the
        first few chunks as spurious lexical hits. In that case we fall back to dense
        ranking only.
        """
        n = cosine.shape[0]
        k = min(top_k, n)
        dense_top = top_k_indices(cosine, k)

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank)
        if float(bm25.max()) > 0.0:  # only fuse lexical when it carries real signal
            lex_top = top_k_indices(bm25, k)
            for rank, idx in enumerate(lex_top):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank)

        ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        top = max((s for _, s in ordered), default=1.0) or 1.0
        return [(idx, score / top) for idx, score in ordered]

    def _rerank(self, query: str, cosine: np.ndarray) -> list[SearchHit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.settings.reranker_model)
        # Reuse the already-computed cosine to pick candidates — no second matmul.
        cand_idx = top_k_indices(cosine, self.settings.top_k)
        chunks = [self.index.chunks[i] for i in cand_idx]
        scores = self._reranker.predict([(query, c.text) for c in chunks])
        hits = [SearchHit(chunk=c, score=float(s)) for c, s in zip(chunks, scores)]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def _lexical_scope_match(self, question: str, bm25: np.ndarray) -> bool:
        """Allow exact-token questions through even when dense scores are conservative.

        This backstops proper-name / exact-term queries that BM25 clearly matches but a
        weaker dense embedder may under-score. To avoid broad topical matches marking an
        off-topic question as answerable, we only allow this path when the top lexical
        hit contains every content token from the query.
        """
        query_tokens = tokenize(question)
        if not query_tokens or bm25.size == 0:
            return False
        top_idx = int(np.argmax(bm25))
        if float(bm25[top_idx]) <= 0.0:
            return False
        overlap = self.bm25.overlap_count(question, top_idx)
        return overlap == len(query_tokens)

    def _scope_threshold(self) -> float:
        """Use a stricter gate when we had to fall back to hashing embeddings."""
        threshold = self.settings.score_threshold
        if self.embedder.uses_hash_fallback():
            return max(threshold, 0.24)
        return threshold

    # --- public API --------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        if len(self.index) == 0:
            return RetrievalResult(hits=[], best_score=0.0, in_scope=False)

        query_vec = self.embedder.embed_one(question)
        cosine = self.index.cosine_scores(query_vec)
        best_score = float(cosine.max())
        bm25 = self.bm25.scores(question)
        # Scope is primarily gated on the dense cosine, with a lexical backstop for exact
        # names/terms that BM25 clearly matches but a weaker embedder under-scores.
        in_scope = (
            best_score >= self._scope_threshold()
            or self._lexical_scope_match(question, bm25)
        )
        if not in_scope:
            return RetrievalResult(hits=[], best_score=best_score, in_scope=False)

        if self.settings.use_reranker:
            kept = self._rerank(question, cosine)[: self.settings.final_k]
        else:
            fused = self._rrf_fuse(cosine, bm25, top_k=self.settings.top_k)
            kept = [
                SearchHit(chunk=self.index.chunks[i], score=round(score, 4))
                for i, score in fused[: self.settings.final_k]
            ]

        return RetrievalResult(hits=kept, best_score=best_score, in_scope=True)
