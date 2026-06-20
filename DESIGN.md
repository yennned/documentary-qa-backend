# Design

A retrieval-augmented Q&A service over a single documentary transcript. The transcript
(~33.5k words / ~45k tokens) would fit in a modern context window, but the system embeds,
indexes, and retrieves per query — feeding only the relevant passages to the LLM — as
required, and because that is the pattern that scales to real corpora (large codebases,
technical docs) where stuffing everything in context is impossible or wasteful.

## 1. Chunking

The transcript is already segmented by `HH:MM:SS` markers on their own lines, one roughly
every minute (260 segments; median ~135 words, but 26 are under 80 words). Those markers
are natural, semantically coherent boundaries **and** give us an accurate time code for
every citation for free.

I group **2 consecutive segments per chunk with a 1-segment stride** (`app/ingest.py`).
Reasoning:
- A lone segment averages ~180 tokens, and the short ones are too thin to retrieve well.
  Grouping two lands chunks near ~360 tokens — in the 256–512 range that works best for
  factoid Q&A.
- The 1-segment overlap means an answer that straddles a minute boundary still lives
  intact inside at least one chunk.
- Each chunk inherits its **first** segment's timestamp, which is what we cite (a "rough
  time code", exactly as the brief asks).

This yields 259 chunks. One-segment-per-chunk is supported too (`CHUNK_SEGMENTS=1`) as a
simpler alternative; grouping measurably improved retrieval on the short segments.

## 2. Retrieval

**Embeddings.** Default is `sentence-transformers/all-MiniLM-L6-v2` running locally and
offline (baked into the Docker image) — no API key, no network. `nomic-embed-text` via
Ollama is available (`EMBED_BACKEND=ollama`) as a higher-quality option. Vectors are
L2-normalized so a dot product is cosine similarity.

**Index.** Brute-force in-memory NumPy (`app/index.py`). At ~260 chunks a vector database
(FAISS/Chroma) is unjustified overhead: the entire search is one matrix-vector product
plus an argsort — exact, sub-millisecond, and trivial to reason about. A vector DB earns
its keep at 10⁵–10⁶ vectors, not 10².

**Hybrid search.** Dense retrieval alone failed an important case: "Who is Annie Gray?"
did not surface the chunk that names her, because that chunk's vector is dominated by its
bread-tasting topic. So I fuse dense cosine with **BM25 lexical** scores
(`app/lexical.py`) via Reciprocal Rank Fusion (`app/retriever.py`). BM25 matches the exact
name; dense handles paraphrase and meaning. This fixed the named-entity case (one of the
five evaluation categories) at negligible cost. An optional cross-encoder reranker
(`bge-reranker-base`, `USE_RERANKER=true`) is wired in as a higher-precision alternative.

**Out-of-scope handling is two-layered**, because retrieval alone does not prevent
hallucination:
1. A cosine **score threshold** (default `0.20`) — a cheap pre-filter. If the best chunk
   is below it, the service returns "I don't know" *without ever calling the LLM*, so an
   off-topic question cannot be answered. The value was calibrated on this transcript:
   clearly in-scope questions scored ≥0.26 (e.g. Annie Gray 0.26, bread 0.62), while
   clearly off-topic ones scored <0.18 (capital of Australia 0.17, World Cup 0.10). It is
   embedding-model-specific and must be re-tuned if you change models.
2. The **LLM abstention prompt** (below) — the primary guard for topically-adjacent
   questions that clear the threshold but aren't actually answered by the passages.

## 3. Prompt construction

`app/prompt.py` builds two messages:
- **System:** answer using *only* the numbered excerpts; if they don't contain the answer,
  reply with a fixed "I don't know — this isn't covered in the documentary"; cite used
  excerpts as `[1]`, `[2]`; be concise and factual.
- **User:** the retrieved chunks rendered as `[n] (time HH:MM:SS) <text>`, then the
  question.

Numbering the sources and requiring bracket citations keeps answers traceable to specific
timestamps, and the explicit abstention instruction is what stops the model from filling
gaps with outside knowledge. Temperature is low (0.1) for faithful, repeatable answers.

## 4. Provider abstraction

Every backend — Ollama (default, local), Kimi K2, GLM, MiniMax, Groq, Gemini — speaks the
OpenAI-compatible Chat Completions API, so one `openai.OpenAI` client serves all of them
(`app/llm.py`). A provider is just a `(base_url, key_env, model)` row in a registry
(`app/config.py`); switching providers is an environment-variable change with **no code
change**. The local default needs no key; the open-source frontier model (Kimi K2) and the
free-tier options plug in by setting `LLM_PROVIDER` and one API key.

## 5. What I'd improve with more time

- **Calibrate the threshold empirically** with a held-out set of in/out-of-scope questions
  and pick it from the score-distribution crossover, rather than a hand-tuned constant.
- **Evaluate `nomic-embed-text` and the reranker** quantitatively (retrieval hit-rate) to
  decide whether their added latency is worth it within the 30s budget.
- **Sentence-aware excerpts**: trim the cited excerpt to the sentence that actually
  supports the answer instead of the chunk head.
- **Answer-level grounding check**: verify each cited `[n]` actually appears, and
  re-prompt or down-rank if the model cites a source it didn't use.
- **Caching** of query embeddings / answers for repeated questions.
