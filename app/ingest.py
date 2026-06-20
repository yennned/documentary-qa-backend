"""Transcript ingestion: raw text -> timestamped segments -> retrieval chunks.

The transcript is already segmented by ``HH:MM:SS`` markers on their own lines, each
followed by ~1 minute of narration. We treat each marker + its text as one *segment*,
then group a sliding window of consecutive segments into a *chunk*. The chunk inherits
its first segment's timestamp, which is what we cite as the source time code.

Why grouping (not one-segment-per-chunk): a single segment is ~135 words (~180 tokens),
and 26 of the 260 segments are very short (<80 words). Grouping two adjacent segments
lands chunks near the 256-512 token range that suits factoid Q&A, and the 1-segment
overlap keeps an answer that straddles a minute boundary inside at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A timestamp line is exactly HH:MM:SS on its own line.
_TIMESTAMP_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})$")


@dataclass(frozen=True)
class Segment:
    timestamp: str
    text: str


@dataclass(frozen=True)
class Chunk:
    index: int
    timestamp: str          # start time code (first segment in the window)
    text: str               # combined narration of the window
    excerpt: str            # short snippet for the API "sources" field


def parse_segments(raw: str) -> list[Segment]:
    """Split raw transcript text into (timestamp, text) segments.

    Lines between one timestamp marker and the next are joined into that segment's text.
    """
    segments: list[Segment] = []
    current_ts: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_ts is not None:
            text = " ".join(part.strip() for part in buffer if part.strip())
            segments.append(Segment(timestamp=current_ts, text=text.strip()))

    for line in raw.splitlines():
        match = _TIMESTAMP_RE.match(line.strip())
        if match:
            flush()
            current_ts = match.group(1)
            buffer = []
        else:
            buffer.append(line)
    flush()
    return segments


def _make_excerpt(text: str, max_chars: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Cut on a word boundary so the snippet reads cleanly.
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"  # ellipsis


def build_chunks(segments: list[Segment], window: int = 2, stride: int = 1) -> list[Chunk]:
    """Group consecutive segments into overlapping chunks.

    window=2, stride=1 => each chunk merges 2 segments and overlaps the next by 1.
    window=1, stride=1 => one-segment-per-chunk (documented alternative).
    """
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    if stride > window:
        # A stride larger than the window leaves gaps: segments between consecutive
        # windows land in no chunk and become unsearchable/uncitable.
        raise ValueError(
            f"stride ({stride}) must be <= window ({window}); a larger stride drops transcript segments"
        )
    if not segments:
        return []

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    n = len(segments)
    while start < n:
        group = segments[start : start + window]
        text = " ".join(s.text for s in group if s.text).strip()
        if text:
            chunks.append(
                Chunk(
                    index=idx,
                    timestamp=group[0].timestamp,
                    text=text,
                    excerpt=_make_excerpt(text),
                )
            )
            idx += 1
        # Stop once the window has covered the tail, to avoid trailing duplicate chunks.
        if start + window >= n:
            break
        start += stride
    return chunks


def load_chunks(path: str | Path, window: int = 2, stride: int = 1) -> list[Chunk]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = parse_segments(raw)
    return build_chunks(segments, window=window, stride=stride)
