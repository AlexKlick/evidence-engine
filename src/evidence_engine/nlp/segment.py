"""Text segmentation (pipeline stage `segment`)."""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SEGMENTS_PER_EVIDENCE = 12


def split_sentences(text: str, max_segments: int = MAX_SEGMENTS_PER_EVIDENCE) -> list[str]:
    """Split snippet/title text into sentence-ish segments."""
    sentences = [
        s.strip() for s in _SENTENCE_SPLIT.split(text or "") if len(s.strip()) >= 12
    ]
    return sentences[:max_segments]
