"""User-drop voice registry for MOSS-TTS-Nano.

Nano is zero-shot from raw reference audio — no transcript is required (the
model conditions on the prompt waveform's audio tokens, not text). Users drop
wav files into:

    ~/.local/share/neural-tts-daemon/voices/moss-tts-nano/

with the filename grammar:

    <voice-id>.<lang>.wav       # 5-15 s reference clip
    <voice-id>.<lang>.toml      # optional sidecar: display_name, gender

<lang> is one of the 20 languages Nano was trained on (see ALLOWED_LANGS).
<voice-id> is the id surfaced to speech-dispatcher.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pb import neural_tts_pb2 as pb

log = logging.getLogger("neural_tts_provider_moss_tts_nano.voices")

# MOSS-TTS-Nano supports 20 languages. ALLOWED_LANGS keys are the short tags
# users put in filenames; values are the BCP-47 tags surfaced to speechd.
ALLOWED_LANGS: dict[str, str] = {
    "zh": "zh-CN",
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "ja": "ja-JP",
    "it": "it-IT",
    "hu": "hu-HU",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "fa": "fa-IR",
    "ar": "ar",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "cs": "cs-CZ",
    "da": "da-DK",
    "sv": "sv-SE",
    "el": "el-GR",
    "tr": "tr-TR",
}
GENDER_FROM_STR = {
    "male": pb.MALE,
    "female": pb.FEMALE,
    "neutral": pb.NEUTRAL,
    "unspecified": pb.GENDER_UNSPECIFIED,
}


@dataclass(frozen=True)
class VoiceEntry:
    voice_id: str
    lang: str  # short tag — key of ALLOWED_LANGS
    wav_path: Path
    display_name: str
    gender: int  # pb.Gender enum value


def _voices_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon" / "voices" / "moss-tts-nano"


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
                sorted(ALLOWED_LANGS),
            )
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
                display_name=display_name,
                gender=gender,
            )
        )

    if not entries:
        log.info(
            "no voices found in %s — drop <id>.<lang>.wav files there "
            "(supported langs: %s)",
            root, ", ".join(sorted(ALLOWED_LANGS)),
        )
    else:
        log.info("loaded %d voice(s) from %s", len(entries), root)
    return entries


def to_pb(entry: VoiceEntry) -> pb.Voice:
    lang_tag = ALLOWED_LANGS.get(entry.lang, entry.lang)
    return pb.Voice(
        id=entry.voice_id,
        language=lang_tag,
        gender=entry.gender,
        display_name=entry.display_name,
    )
