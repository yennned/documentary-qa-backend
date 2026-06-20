"""Tests for transcript parsing and chunking."""
from app.ingest import build_chunks, load_chunks, parse_segments

SAMPLE = """00:00:05
First segment line one. Line two.
00:01:05
Second segment text.
00:02:02
Third segment text here.
"""


def test_parse_segments_splits_on_timestamps():
    segments = parse_segments(SAMPLE)
    assert len(segments) == 3
    assert segments[0].timestamp == "00:00:05"
    assert "First segment" in segments[0].text
    assert segments[1].timestamp == "00:01:05"


def test_build_chunks_window_and_overlap():
    segments = parse_segments(SAMPLE)
    chunks = build_chunks(segments, window=2, stride=1)
    # 3 segments, window 2, stride 1 -> chunks starting at seg 0 and seg 1.
    assert len(chunks) == 2
    # First chunk merges segments 0 and 1, keeps segment 0's timestamp.
    assert chunks[0].timestamp == "00:00:05"
    assert "First segment" in chunks[0].text and "Second segment" in chunks[0].text
    # Overlap: segment 1 appears in both chunk 0 and chunk 1.
    assert "Second segment" in chunks[1].text


def test_single_segment_chunking_alternative():
    segments = parse_segments(SAMPLE)
    chunks = build_chunks(segments, window=1, stride=1)
    assert len(chunks) == 3
    assert [c.timestamp for c in chunks] == ["00:00:05", "00:01:05", "00:02:02"]


def test_real_transcript_has_expected_shape():
    chunks = load_chunks("data/transcript.txt", window=2, stride=1)
    assert len(chunks) == 259  # 260 segments, window 2 stride 1
    assert all(len(c.timestamp) == 8 and c.text for c in chunks)
    assert all(c.excerpt for c in chunks)
