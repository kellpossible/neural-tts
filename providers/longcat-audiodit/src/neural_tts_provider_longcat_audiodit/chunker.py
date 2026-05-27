"""Intelligent text chunking for LongCat.

LongCat synthesises whole clips per forward pass. For paragraph-length input
that would make time-to-first-audio painful. We split text into sentence-
aware chunks and let the provider synth and stream each one separately.

Goals:
- Never split mid-word.
- Prefer sentence boundaries; pack short sentences together so each chunk
  is a meaningful unit (TARGET_CHARS).
- Don't let any chunk grow past HARD_CAP_CHARS (the model has a finite
  max duration per forward pass).
- Fall back gracefully on sentences that are themselves too long.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

log = logging.getLogger("neural_tts_provider_longcat_audiodit.chunker")

TARGET_CHARS = 220
HARD_CAP_CHARS = 400

# Soft break points inside a too-long sentence, in priority order.
_SOFT_BREAKS = re.compile(r"[,;:—–]\s+|\s+—\s+")


def _pysbd_segment(text: str, lang: str) -> list[str]:
    """Sentence-segment via pysbd; fall back to a regex split if pysbd chokes."""
    try:
        import pysbd  # type: ignore[import-not-found]
    except ImportError:
        log.warning("pysbd not installed, falling back to regex sentence split")
        return _regex_segment(text)

    pysbd_lang = "zh" if lang.startswith("zh") else "en"
    try:
        seg = pysbd.Segmenter(language=pysbd_lang, clean=False)
        sents = [s.strip() for s in seg.segment(text) if s and s.strip()]
        return sents or [text.strip()]
    except Exception as e:
        log.warning("pysbd failed (%s); using regex fallback", e)
        return _regex_segment(text)


def _regex_segment(text: str) -> list[str]:
    # Conservative split: sentence-ending punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_oversized(sentence: str) -> list[str]:
    """Split a single oversized sentence at soft breakpoints, never mid-word."""
    if len(sentence) <= HARD_CAP_CHARS:
        return [sentence]

    pieces: list[str] = []
    remaining = sentence
    while len(remaining) > HARD_CAP_CHARS:
        # Look for the latest soft break that keeps the head under HARD_CAP_CHARS.
        head_limit = remaining[:HARD_CAP_CHARS]
        matches = list(_SOFT_BREAKS.finditer(head_limit))
        if matches:
            cut = matches[-1].end()
        else:
            # No soft break — fall back to the last space before the cap.
            space = head_limit.rfind(" ")
            cut = space + 1 if space > 0 else HARD_CAP_CHARS
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def chunk(text: str, lang: str) -> Iterator[str]:
    """Yield chunks ready for individual model.forward() calls."""
    text = text.strip()
    if not text:
        return

    sentences = _pysbd_segment(text, lang)

    # Pre-split any oversized sentences into pieces that respect HARD_CAP_CHARS.
    expanded: list[str] = []
    for s in sentences:
        if len(s) > HARD_CAP_CHARS:
            expanded.extend(_split_oversized(s))
        else:
            expanded.append(s)

    buf: list[str] = []
    buf_len = 0
    for s in expanded:
        # +1 accounts for the joining space we'll insert between pieces.
        addition = len(s) + (1 if buf else 0)
        if buf and buf_len + addition > TARGET_CHARS:
            yield " ".join(buf)
            buf = [s]
            buf_len = len(s)
        else:
            buf.append(s)
            buf_len += addition
    if buf:
        yield " ".join(buf)
