"""FastAPI application exposing the Documentary Q&A backend.

Endpoints:
  POST /ask            -> {answer, sources}            (grounded answer + ranked sources)
  POST /ask?stream=true-> text/event-stream            (bonus: token-by-token streaming)
  GET  /health         -> {status, chunks, provider}
  GET  /               -> minimal web UI                (bonus)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAIError

from .config import get_settings
from .llm import LLMClient
from .prompt import OUT_OF_SCOPE_ANSWER, build_messages
from .retriever import Retriever
from .schemas import AskRequest, AskResponse, Source

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Returned when the LLM produces no content despite in-scope context.
EMPTY_ANSWER_FALLBACK = "I couldn't generate an answer from the transcript for that question."


class QAService:
    """Builds the index once, then answers questions against it."""

    def __init__(self):
        self.settings = get_settings()
        self.retriever = Retriever(self.settings)
        self.llm = LLMClient(self.settings)

    def _sources(self, hits) -> list[Source]:
        return [
            Source(timestamp=h.chunk.timestamp, excerpt=h.chunk.excerpt, score=round(h.score, 4))
            for h in hits
        ]

    def _prepare(self, question: str):
        """Shared retrieval + scope decision for both the blocking and streaming paths.

        Returns (in_scope, messages_or_None, hits). When out-of-scope, messages is None
        and the caller returns the canned abstention without ever calling the LLM.
        """
        result = self.retriever.retrieve(question)
        if not result.in_scope:
            return False, None, []
        return True, build_messages(question, result.hits), result.hits

    def answer(self, question: str) -> AskResponse:
        in_scope, messages, hits = self._prepare(question)
        if not in_scope:
            return AskResponse(answer=OUT_OF_SCOPE_ANSWER, sources=[])
        answer = self.llm.complete(messages) or EMPTY_ANSWER_FALLBACK
        return AskResponse(answer=answer, sources=self._sources(hits))

    def stream(self, question: str):
        """Yield Server-Sent-Events: token deltas, then a final 'sources' event.

        The LLM call runs inside this generator (after the response has started), so its
        errors can't reach the route handler — we catch them here and emit an 'error'
        event followed by 'done' so the client is never left hanging on a silent stream.
        """
        in_scope, messages, hits = self._prepare(question)
        if not in_scope:
            yield _sse("token", {"text": OUT_OF_SCOPE_ANSWER})
            yield _sse("sources", {"sources": []})
            yield _sse("done", {})
            return
        emitted = False
        try:
            for delta in self.llm.stream(messages):
                emitted = True
                yield _sse("token", {"text": delta})
        except Exception as exc:  # provider/connectivity failure mid-stream
            yield _sse("error", {"detail": f"LLM backend error: {exc}"})
            yield _sse("done", {})
            return
        if not emitted:
            yield _sse("token", {"text": EMPTY_ANSWER_FALLBACK})
        sources = [s.model_dump() for s in self._sources(hits)]
        yield _sse("sources", {"sources": sources})
        yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


service: QAService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = QAService()  # builds + embeds the index at startup
    yield


app = FastAPI(title="Documentary Q&A Backend", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "status": "ok",
        "chunks": len(service.retriever.index),
        "llm_provider": service.settings.llm_provider,
        "llm_model": service.llm.model,
        "embed_backend": service.settings.embed_backend,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, stream: bool = Query(False)):
    if service is None:  # startup not complete / failed
        raise HTTPException(status_code=503, detail="Service not ready")
    if stream:
        # Streaming errors are handled inside the generator (it runs after this returns).
        return StreamingResponse(service.stream(req.question), media_type="text/event-stream")
    try:
        return service.answer(req.question)
    except OpenAIError as exc:  # only LLM/provider failures become 503
        raise HTTPException(status_code=503, detail=f"LLM backend error: {exc}") from exc


@app.get("/")
def index_page():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(html)
    return {"message": "Documentary Q&A backend. POST /ask with {\"question\": \"...\"}."}
