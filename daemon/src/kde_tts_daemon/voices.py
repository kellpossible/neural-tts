"""Voice metadata model."""

from __future__ import annotations

from dataclasses import dataclass


# Kokoro voice-prefix → (speechd-language, gender) reference.
# Kept here for documentation; the actual provider-side mapping lives in
# providers/kokoro-onnx/src/kde_tts_provider_kokoro_onnx/provider.py because
# venv isolation precludes importing across packages.
KOKORO_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "af": ("en-US", "FEMALE"),
    "am": ("en-US", "MALE"),
    "bf": ("en-GB", "FEMALE"),
    "bm": ("en-GB", "MALE"),
    "jf": ("ja", "FEMALE"),
    "jm": ("ja", "MALE"),
    "zf": ("zh", "FEMALE"),
    "zm": ("zh", "MALE"),
    "ef": ("es", "FEMALE"),
    "em": ("es", "MALE"),
    "ff": ("fr", "FEMALE"),
    "fm": ("fr", "MALE"),
    "hf": ("hi", "FEMALE"),
    "hm": ("hi", "MALE"),
    "if": ("it", "FEMALE"),
    "im": ("it", "MALE"),
    "pf": ("pt-BR", "FEMALE"),
    "pm": ("pt-BR", "MALE"),
}


@dataclass(frozen=True)
class Voice:
    id: str
    language: str
    gender: str  # "MALE", "FEMALE", "NEUTRAL"
    display_name: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "language": self.language,
            "gender": self.gender,
            "display_name": self.display_name,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Voice":
        return cls(
            id=raw["id"],
            language=raw["language"],
            gender=raw["gender"],
            display_name=raw.get("display_name"),
        )
