"""Model construction for the pocket-tts provider.

pocket-tts is CPU-only (upstream reports GPU gives no speedup). `load_model()`
fetches weights from Hugging Face on first call and caches them in the standard
HF cache, so there are no explicit model-path knobs here.

The model weights and per-preset voice embeddings are pre-fetched at install
time (`scripts/download_models.py`), so at runtime we force HF into offline mode:
under the daemon's systemd sandbox the HF cache is mounted read-only, and an
online lookup would try to take a write-lock in the cache and crash with
`OSError: Read-only file system`. Offline mode reads the cached snapshot directly
with no lock. Set HF_HUB_OFFLINE=0 in the provider env to opt back into online
fetches (e.g. when adding the gated voice-cloning model).

Env overrides:
  TTS_POCKET_CPU_THREADS  torch intra-op thread count (upstream uses ~2 cores)
  TTS_POCKET_LANGUAGE     model language passed to load_model (default: english)
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Must be set before huggingface_hub is first imported (it snapshots this into a
# module constant at import time). Kept as setdefault so the provider env can
# override it. See the module docstring for why offline is the runtime default.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

log = logging.getLogger("neural_tts_provider_pocket_tts.engine")

# Mimi codec native rate. Reconciled against model.sample_rate on load; used to
# populate the WarmupResponse in lazy mode without instantiating the model.
NATIVE_SAMPLE_RATE = 24_000


def build_model() -> Any:
    """Load the pocket-tts model. Blocking; call via asyncio.to_thread."""
    threads = os.environ.get("TTS_POCKET_CPU_THREADS", "").strip()
    if threads:
        try:
            import torch

            torch.set_num_threads(int(threads))
            log.info("torch intra-op threads set to %s", threads)
        except Exception:
            log.warning("could not honor TTS_POCKET_CPU_THREADS=%r", threads, exc_info=True)

    from pocket_tts import TTSModel

    language = os.environ.get("TTS_POCKET_LANGUAGE", "").strip() or None
    log.info("loading pocket-tts model (language=%s)", language or "english[default]")
    model = TTSModel.load_model(language=language)
    log.info("pocket-tts model ready: sample_rate=%d", int(model.sample_rate))
    return model
