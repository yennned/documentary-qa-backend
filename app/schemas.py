"""Request/response models for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about the documentary.")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # min_length counts characters, so a whitespace-only string would slip through;
        # strip and reject it so the empty query never reaches retrieval.
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    timestamp: str = Field(..., description="Rough time code (HH:MM:SS) where the supporting passage starts.")
    excerpt: str = Field(..., description="Short verbatim snippet from the transcript.")
    score: float = Field(..., description="Relevance score used for ranking (higher = more relevant); reflects the ordering.")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list, description="Ranked source references (most relevant first).")
