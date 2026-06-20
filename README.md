# Documentary Q&A Backend

A retrieval-augmented (RAG) backend that answers natural-language questions about a
documentary transcript and returns grounded answers with ranked, timestamped sources.

- **Endpoint:** `POST /ask` → `{ "answer": "...", "sources": [ { "timestamp": "HH:MM:SS", "excerpt": "...", "score": 0.xx } ] }`
- Runs **fully locally and offline** by default (Ollama + an open-source model). No paid
  service required.
- LLM provider is **switchable by environment variable** — local Ollama, Kimi K2, GLM,
  MiniMax, Groq, or Gemini — with no code changes.

See [DESIGN.md](DESIGN.md) for the chunking, retrieval, and prompting rationale.

## Quick start (one command, fully local)

Requires Docker.

```bash
docker compose up
```

On first boot the `ollama` service downloads the chat model (`llama3.1:8b`, ~5 GB) — this
is a one-time download cached on a named volume. The API waits until the model is ready,
then comes up on **http://localhost:8000**. A minimal web UI is at the root URL.

Ask a question:

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What did the Victorians add to bread?"}' | jq
```

Health check: `curl http://localhost:8000/health`

### Lighter model

`llama3.1:8b` needs a reasonably capable machine. For a smaller/faster pull, set a
different model before starting:

```bash
OLLAMA_MODEL=llama3.2:3b docker compose up
```

## Running without Docker (local dev)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: local Ollama (install from https://ollama.com, then:)
ollama serve &              # in another terminal
ollama pull llama3.1:8b
uvicorn app.main:app --reload

# Embeddings run locally (all-MiniLM-L6-v2, downloaded once); no key needed.
```

## Using a hosted open-source model (e.g. Kimi K2)

The default is local Ollama. To use the open-source **Kimi K2** model (or another hosted
provider) instead, copy `.env.example` to `.env` and set the provider + key:

```bash
LLM_PROVIDER=kimi
MOONSHOT_API_KEY=sk-...        # from https://platform.moonshot.ai
```

Then start **just the API** (no local model needed) — with Docker:

```bash
docker compose up api      # starts only the api; does not wait for the Ollama pull
```

or without Docker: `uvicorn app.main:app`. Other providers work the same way —
`LLM_PROVIDER=glm|minimax|groq|gemini` with the matching key from `.env.example`. If the
selected provider's key is missing, the service fails fast with a clear message naming the
exact variable.

| Provider | `LLM_PROVIDER` | Key env var | Get a key |
|----------|----------------|-------------|-----------|
| Ollama (local, default) | `ollama` | — | install Ollama |
| Kimi K2 (open-weight)   | `kimi`    | `MOONSHOT_API_KEY` | platform.moonshot.ai |
| GLM (open-weight)       | `glm`     | `ZHIPU_API_KEY`    | open.bigmodel.cn |
| MiniMax (open-weight)   | `minimax` | `MINIMAX_API_KEY`  | minimaxi.com |
| Groq (free tier)        | `groq`    | `GROQ_API_KEY`     | console.groq.com/keys |
| Gemini (free tier)      | `gemini`  | `GEMINI_API_KEY`   | aistudio.google.com/apikey |

## API

### `POST /ask`
Request: `{ "question": "..." }`
Response: `{ "answer": "...", "sources": [ { "timestamp", "excerpt", "score" } ] }`

Out-of-scope questions (not covered by the transcript) return
`"I don't know — this isn't covered in the documentary."` with no sources.

**Streaming (bonus):** `POST /ask?stream=true` returns Server-Sent Events — `token` events
with answer deltas, then a `sources` event. The web UI at `/` uses this.

## Tests

```bash
pip install -r requirements.txt
pytest
```

## Configuration

All knobs are environment variables — see [.env.example](.env.example) for the full,
documented list (provider, models, chunking, retrieval thresholds).
