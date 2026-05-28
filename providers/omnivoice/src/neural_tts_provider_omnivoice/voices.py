"""User-drop voice registry for OmniVoice.

OmniVoice is zero-shot: each "voice" is a short reference clip. Users drop into:

    ~/.local/share/neural-tts-daemon/voices/omnivoice/

with the filename grammar:

    <voice-id>.<lang>.wav       # 3-10 s reference clip
    <voice-id>.<lang>.txt       # OPTIONAL transcript of the WAV. If absent,
                                # upstream's Whisper auto-transcribes the clip
                                # the first time the voice is used.
    <voice-id>.<lang>.toml      # OPTIONAL: display_name, gender

<lang> is a BCP-47 primary subtag (e.g. en, fr, zh, sw). OmniVoice supports
600+ languages — we pass <lang> through and let upstream decide.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pb import neural_tts_pb2 as pb

log = logging.getLogger("neural_tts_provider_omnivoice.voices")

GENDER_FROM_STR = {
    "male": pb.MALE,
    "female": pb.FEMALE,
    "neutral": pb.NEUTRAL,
    "unspecified": pb.GENDER_UNSPECIFIED,
}


@dataclass(frozen=True)
class VoiceEntry:
    voice_id: str
    lang: str  # BCP-47 primary subtag (e.g. "en", "zh", "fr")
    wav_path: Path
    prompt_text: str | None  # None → upstream auto-transcribes via Whisper
    display_name: str
    gender: int  # pb.Gender enum value


def _voices_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon" / "voices" / "omnivoice"


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
        if not lang:
            log.warning("skipping %s: empty lang after voice id", wav.name)
            continue

        prompt_text: str | None = None
        txt = wav.with_suffix(".txt")
        if txt.exists():
            try:
                prompt_text = txt.read_text(encoding="utf-8").strip() or None
            except OSError as e:
                log.warning("ignoring transcript %s: %s", txt.name, e)

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
            "no voices found in %s — drop <id>.<lang>.wav files there "
            "(optional <id>.<lang>.txt overrides Whisper auto-transcription)",
            root,
        )
    else:
        log.info("loaded %d voice(s) from %s", len(entries), root)
    return entries


# Common BCP-47 region tags to give Firefox/Okular a recognisable locale.
# Anything not in this map is passed through as the primary subtag alone,
# which most consumers accept (Qt/speechd, festival, etc.).
_LANG_TO_BCP47 = {
    "en": "en-US",
    "zh": "zh-CN",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "ar": "ar-SA",
}


def to_pb(entry: VoiceEntry) -> pb.Voice:
    lang_tag = _LANG_TO_BCP47.get(entry.lang, entry.lang)
    return pb.Voice(
        id=entry.voice_id,
        language=lang_tag,
        gender=entry.gender,
        display_name=entry.display_name,
    )
