"""Sentence-aware text chunking for qwen3-tts.

Ported from omnivoice's chunker. Used only when the user opts in via
`TTS_QWEN3_CHUNKER=1`. faster-qwen3-tts streams natively, so the default
(no chunker) is fine on GPUs that sustain RTF >= 1. On slower hardware,
chunking caps per-chunk audio duration so the adaptive jitter buffer in
provider.py can size a bounded prebuffer per piece.

Tunables (read at module load):
  TTS_QWEN3_CHUNK_TARGET_CHARS   default 120
  TTS_QWEN3_CHUNK_HARD_CAP_CHARS default 240

Defaults are deliberately smaller than omnivoice's (220/400) because the
prebuffer needed below realtime is ~chunk_duration*(1-rtf)/rtf — capping
the chunk caps TTFA.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterator

log = logging.getLogger("neural_tts_provider_qwen3_tts.chunker")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring %s=%r (not int); using %d", name, raw, default)
        return default


TARGET_CHARS = _env_int("TTS_QWEN3_CHUNK_TARGET_CHARS", 120)
HARD_CAP_CHARS = _env_int("TTS_QWEN3_CHUNK_HARD_CAP_CHARS", 240)

_SOFT_BREAKS = re.compile(r"[,;:—–]\s+|\s+—\s+")

_PYSBD_LANGS = {"en", "zh", "fr", "de", "es", "it", "ja", "ru", "pt", "nl", "pl", "ar"}


def _pysbd_segment(text: str, lang: str) -> list[str]:
    try:
        import pysbd  # type: ignore[import-not-found]
    except ImportError:
        log.warning("pysbd not installed, falling back to regex sentence split")
        return _regex_segment(text)

    primary = lang.split("-", 1)[0].lower()
    pysbd_lang = primary if primary in _PYSBD_LANGS else "en"
    try:
        seg = pysbd.Segmenter(language=pysbd_lang, clean=False)
        sents = [s.strip() for s in seg.segment(text) if s and s.strip()]
        return sents or [text.strip()]
    except Exception as e:
        log.warning("pysbd failed (%s); using regex fallback", e)
        return _regex_segment(text)


def _regex_segment(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_oversized(sentence: str) -> list[str]:
    if len(sentence) <= HARD_CAP_CHARS:
        return [sentence]
    pieces: list[str] = []
    remaining = sentence
    while len(remaining) > HARD_CAP_CHARS:
        head = remaining[:HARD_CAP_CHARS]
        matches = list(_SOFT_BREAKS.finditer(head))
        if matches:
            cut = matches[-1].end()
        else:
            space = head.rfind(" ")
            cut = space + 1 if space > 0 else HARD_CAP_CHARS
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def chunk(text: str, lang: str) -> Iterator[str]:
    text = text.strip()
    if not text:
        return
    sentences = _pysbd_segment(text, lang)
    expanded: list[str] = []
    for s in sentences:
        if len(s) > HARD_CAP_CHARS:
            expanded.extend(_split_oversized(s))
        else:
            expanded.append(s)

    buf: list[str] = []
    buf_len = 0
    for s in expanded:
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
