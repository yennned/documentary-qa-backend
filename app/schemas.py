"""Request/response models for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about the documentary.")


class Source(BaseModel):
    timestamp: str = Field(..., description="Rough time code (HH:MM:SS) where the supporting passage starts.")
    excerpt: str = Field(..., description="Short verbatim snippet from the transcript.")
    score: float = Field(..., description="Relevance score (cosine similarity) used for ranking.")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list, description="Ranked source references (most relevant first).")
