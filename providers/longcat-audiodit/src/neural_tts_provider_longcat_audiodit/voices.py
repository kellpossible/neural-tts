"""User-drop voice registry for LongCat.

LongCat is zero-shot: every "voice" is a pair of (reference clip, transcript
of that clip). Users drop both into:

    ~/.local/share/neural-tts-daemon/voices/longcat/

with the filename grammar:

    <voice-id>.<lang>.wav       # 5-15 s reference clip, any channel layout
    <voice-id>.<lang>.txt       # exact transcript of the WAV (UTF-8)
    <voice-id>.<lang>.toml      # optional sidecar: display_name, gender

<lang> must be `en` or `zh` (the languages LongCat was trained on).
<voice-id> is the id surfaced to speechd.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pb import neural_tts_pb2 as pb

log = logging.getLogger("neural_tts_provider_longcat_audiodit.voices")

ALLOWED_LANGS = ("en", "zh")
GENDER_FROM_STR = {
    "male": pb.MALE,
    "female": pb.FEMALE,
    "neutral": pb.NEUTRAL,
    "unspecified": pb.GENDER_UNSPECIFIED,
}


@dataclass(frozen=True)
class VoiceEntry:
    voice_id: str
    lang: str  # "en" or "zh"
    wav_path: Path
    prompt_text: str
    display_name: str
    gender: int  # pb.Gender enum value


def _voices_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon" / "voices" / "longcat"


def scan_voices() -> list[VoiceEntry]:
    """Rescan the user-drop dir; return all valid voice entries."""
    root = _voices_root()
    if not root.exists():
        log.info("voices dir does not exist yet: %s", root)
        return []

    entries: list[VoiceEntry] = []
    for wav in sorted(root.glob("*.wav")):
        stem = wav.stem  # e.g. "alice.en"
        if "." not in stem:
            log.warning("skipping %s: expected <id>.<lang>.wav, got no lang suffix", wav.name)
            continue
        voice_id, _, lang = stem.rpartition(".")
        if not voice_id:
            log.warning("skipping %s: empty voice id before lang suffix", wav.name)
            continue
        if lang not in ALLOWED_LANGS:
            log.warning(
                "skipping %s: lang %r not supported (must be one of %s)",
                wav.name,
                lang,
                ALLOWED_LANGS,
            )
            continue

        txt = wav.with_suffix(".txt")
        if not txt.exists():
            log.warning("skipping %s: missing transcript sibling %s", wav.name, txt.name)
            continue
        try:
            prompt_text = txt.read_text(encoding="utf-8").strip()
        except OSError as e:
            log.warning("skipping %s: cannot read transcript: %s", wav.name, e)
            continue
        if not prompt_text:
            log.warning("skipping %s: transcript file is empty", wav.name)
            continue

        display_name = voice_id
        gender = pb.GENDER_UNSPECIFIED
        toml_path = wav.with_suffix(".toml")
        if toml_path.exists():
            try:
                with toml_path.open("rb") as f:
                    meta = tomllib.load(f)
                display_name = str(meta.get("display_name") or display_name)
                gender_str = str(meta.get("gender") or "").lower()
                if gender_str:
                    gender = GENDER_FROM_STR.get(gender_str, pb.GENDER_UNSPECIFIED)
            except (OSError, tomllib.TOMLDecodeError) as e:
                log.warning("ignoring malformed sidecar %s: %s", toml_path.name, e)

        entries.append(
            VoiceEntry(
                voice_id=voice_id,
                lang=lang,
                wav_path=wav,
                prompt_text=prompt_text,
                display_name=display_name,
                gender=gender,
            )
        )

    if not entries:
        log.info(
            "no voices found in %s — drop <id>.<en|zh>.{wav,txt} pairs there", root
        )
    else:
        log.info("loaded %d voice(s) from %s", len(entries), root)
    return entries


def to_pb(entry: VoiceEntry) -> pb.Voice:
    # Map LongCat's "en"/"zh" to BCP-47 tags consumers like Firefox/Okular
    # expect. en→en-US (matches Kokoro's default English bucket), zh→zh-CN.
    lang_tag = {"en": "en-US", "zh": "zh-CN"}.get(entry.lang, entry.lang)
    return pb.Voice(
        id=entry.voice_id,
        language=lang_tag,
        gender=entry.gender,
        display_name=entry.display_name,
    )
