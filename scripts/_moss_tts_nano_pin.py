"""Single source of truth for which upstream MOSS-TTS-Nano commit we use.

Pinning the commit pins everything upstream-side: the onnx_tts_runtime module,
its helpers, AND the requirements.txt. Bumping the pin is the one-liner change
to upgrade.
"""

MOSS_TTS_NANO_UPSTREAM_REPO = "https://github.com/OpenMOSS/MOSS-TTS-Nano.git"
MOSS_TTS_NANO_UPSTREAM_COMMIT = "b09d1962a7df4d4007146ea53880c4b282d83e39"
