"""LongCat-AudioDiT model load + warmup.

Loads `meituan-longcat/LongCat-AudioDiT-1B` onto CUDA in fp16. CPU is not
supported — diffusion TTS without GPU acceleration is impractical for an
interactive daemon.

Upstream's `audiodit/` package is not on PyPI; it lives in the LongCat-AudioDiT
GitHub repo, vendored by scripts/install_provider.py into
`providers/longcat-audiodit/vendor/LongCat-AudioDiT/` at the commit pinned in
scripts/_longcat_pin.py. We add that path to sys.path before importing it.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("neural_tts_provider_longcat_audiodit.engine")

HF_REPO_ID = "meituan-longcat/LongCat-AudioDiT-1B"
MODEL_DIR_NAME = "longcat-audiodit-1b"
SAMPLE_RATE = 24_000

# providers/longcat-audiodit/vendor/LongCat-AudioDiT/ — three parents up from
# this file: .../src/neural_tts_provider_longcat_audiodit/engine.py → src →
# project_dir.
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "LongCat-AudioDiT"


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon"


def resolve_model_dir() -> Path:
    """Local directory containing the snapshot_download'd model files."""
    override = os.environ.get("TTS_LONGCAT_MODEL_PATH")
    if override:
        return Path(override)
    return _data_dir() / "models" / MODEL_DIR_NAME


def ensure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "LongCat-AudioDiT requires CUDA, but torch reports no GPU is available. "
            "Switch to a CUDA host or pick a different provider."
        )


def build_longcat():
    """Load the LongCat-AudioDiT model + tokenizer onto CUDA in fp16.

    Returns (model, tokenizer). Both are kept resident across requests.
    """
    ensure_cuda()

    import torch
    from transformers import AutoTokenizer

    # The `audiodit` package isn't pip-installable upstream; it lives in the
    # cloned repo next to our venv. Add it to sys.path before importing.
    if not _VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"LongCat upstream source not vendored at {_VENDOR_DIR}; "
            "run `mise run install-provider longcat-audiodit --extra gpu` to fetch it."
        )
    vendor_str = str(_VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    from audiodit import AudioDiTModel  # type: ignore[import-not-found]

    model_dir = resolve_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"LongCat model directory not found at {model_dir}; "
            "run `mise run download-models longcat-audiodit` to fetch it."
        )

    log.info("loading LongCat-AudioDiT from %s", model_dir)
    model = AudioDiTModel.from_pretrained(str(model_dir))
    model = model.to("cuda", dtype=torch.float16)
    # The VAE is loaded separately by the model class; halve it explicitly.
    if hasattr(model, "vae") and hasattr(model.vae, "to_half"):
        model.vae.to_half()
    model.eval()

    text_encoder = getattr(model.config, "text_encoder_model", None)
    if text_encoder is None:
        # Fallback: tokenizer is shipped with the snapshot.
        text_encoder = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)

    log.info("LongCat ready on CUDA (fp16, sample_rate=%d)", SAMPLE_RATE)
    return model, tokenizer
