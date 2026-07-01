"""Voice registry for the pocket-tts provider.

Two voice sources share one id namespace:

1. **Preset voices** — the built-in named voices shipped by pocket-tts
   (`pocket_tts.utils.utils._ORIGINS_OF_PREDEFINED_VOICES`). We mirror that
   list here as PRESET_VOICES, annotated with a BCP-47 language tag and a
   best-effort gender for the KDE voice picker. The id we surface is the bare
   preset name (e.g. "alba"), which is exactly what
   `TTSModel.get_state_for_audio_prompt(name)` accepts.

2. **Cloned voices** — user-supplied reference clips. pocket-tts is zero-shot
   from raw audio (no transcript needed), so users just drop wav files into:

       ~/.local/share/neural-tts-daemon/voices/pocket-tts/

   with the filename grammar:

       <voice-id>.<lang>.wav       # a few seconds of clean reference speech
       <voice-id>.<lang>.toml      # optional sidecar: display_name, gender

   <lang> is one of ALLOWED_LANGS; <voice-id> is surfaced to speech-dispatcher.
   Preset names are reserved: a cloned clip that reuses a preset id is skipped
   with a warning.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pb import neural_tts_pb2 as pb

log = logging.getLogger("neural_tts_provider_pocket_tts.voices")

# Short filename tag → BCP-47 tag surfaced to speechd. pocket-tts conditions on
# the reference waveform rather than a language id, so this list is deliberately
# permissive; the tag is advisory metadata for the KDE picker.
ALLOWED_LANGS: dict[str, str] = {
    "en": "en-US",
    "it": "it-IT",
    "es": "es-ES",
    "de": "de-DE",
    "fr": "fr-FR",
    "pt": "pt-PT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "ko": "ko-KR",
    "ru": "ru-RU",
}

GENDER_FROM_STR = {
    "male": pb.MALE,
    "female": pb.FEMALE,
    "neutral": pb.NEUTRAL,
    "unspecified": pb.GENDER_UNSPECIFIED,
}

# The ~26 built-in preset voices, mirrored from pocket-tts'
# _ORIGINS_OF_PREDEFINED_VOICES. Value is (BCP-47 language, pb.Gender). Language
# is derived from each origin clip's source; gender is best-effort (based on the
# reference speaker) and only affects how the voice is labelled in KDE.
PRESET_VOICES: dict[str, tuple[str, int]] = {
    # English
    "alba": ("en-US", pb.FEMALE),
    "anna": ("en-US", pb.FEMALE),
    "vera": ("en-US", pb.FEMALE),
    "fantine": ("en-US", pb.FEMALE),
    "cosette": ("en-US", pb.FEMALE),
    "eponine": ("en-US", pb.FEMALE),
    "azelma": ("en-US", pb.FEMALE),
    "mary": ("en-US", pb.FEMALE),
    "jane": ("en-US", pb.FEMALE),
    "eve": ("en-US", pb.FEMALE),
    "caro_davy": ("en-US", pb.FEMALE),
    "marius": ("en-US", pb.MALE),
    "javert": ("en-US", pb.MALE),
    "jean": ("en-US", pb.MALE),
    "charles": ("en-US", pb.MALE),
    "paul": ("en-US", pb.MALE),
    "george": ("en-US", pb.MALE),
    "michael": ("en-US", pb.MALE),
    "bill_boerst": ("en-US", pb.MALE),
    "peter_yearsley": ("en-US", pb.MALE),
    "stuart_bell": ("en-US", pb.MALE),
    # Other languages
    "giovanni": ("it-IT", pb.MALE),
    "lola": ("es-ES", pb.FEMALE),
    "juergen": ("de-DE", pb.MALE),
    "rafael": ("pt-PT", pb.MALE),
    "estelle": ("fr-FR", pb.FEMALE),
}


@dataclass(frozen=True)
class VoiceEntry:
    """A single voice — either a preset (wav_path is None) or a cloned clip."""

    voice_id: str
    language: str  # BCP-47 tag surfaced to speechd
    display_name: str
    gender: int  # pb.Gender enum value
    wav_path: Path | None  # None for presets; the reference clip for clones


def _display(voice_id: str) -> str:
    return voice_id.replace("_", " ").title()


def preset_entries() -> list[VoiceEntry]:
    return [
        VoiceEntry(
            voice_id=name,
            language=lang,
            display_name=_display(name),
            gender=gender,
            wav_path=None,
        )
        for name, (lang, gender) in PRESET_VOICES.items()
    ]


def _voices_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon" / "voices" / "pocket-tts"


def scan_cloned_voices() -> list[VoiceEntry]:
    """Rescan the user-drop dir; return valid cloned-voice entries."""
    root = _voices_root()
    if not root.exists():
        log.info("voices dir does not exist yet: %s", root)
        return []

    entries: list[VoiceEntry] = []
    for wav in sorted(root.glob("*.wav")):
        stem = wav.stem  # e.g. "myvoice.en"
        if "." not in stem:
            log.warning("skipping %s: expected <id>.<lang>.wav, got no lang suffix", wav.name)
            continue
        voice_id, _, lang = stem.rpartition(".")
        if not voice_id:
            log.warning("skipping %s: empty voice id before lang suffix", wav.name)
            continue
        if voice_id in PRESET_VOICES:
            log.warning(
                "skipping %s: %r is a reserved preset voice id; rename the file",
                wav.name,
                voice_id,
            )
            continue
        if lang not in ALLOWED_LANGS:
            log.warning(
                "skipping %s: lang %r not supported (must be one of %s)",
                wav.name,
                lang,
                sorted(ALLOWED_LANGS),
            )
            continue

        display_name = _display(voice_id)
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
                language=ALLOWED_LANGS[lang],
                display_name=display_name,
                gender=gender,
                wav_path=wav,
            )
        )

    if entries:
        log.info("loaded %d cloned voice(s) from %s", len(entries), root)
    return entries


def to_pb(entry: VoiceEntry) -> pb.Voice:
    return pb.Voice(
        id=entry.voice_id,
        language=entry.language,
        gender=entry.gender,
        display_name=entry.display_name,
    )
