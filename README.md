# neural-tts

A bridge between operating system TTS interface and various newly available TTS libraries.

Currently `speech-dispatcher` on Linux is supported, with five providers:

- **kokoro-onnx** — Kokoro-82M via ONNX runtime. CPU or GPU, dozens of
  pre-baked voices, multilingual.
- **longcat-audiodit** — Meituan's [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)
  1B diffusion model. CUDA-only, zero-shot voice cloning from user-supplied
  reference clips, English + Chinese.
- **moss-tts-nano** — OpenMOSS's [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)
  100M autoregressive ONNX model. CPU-friendly (claims realtime on 4 cores),
  zero-shot voice cloning, 20 languages.
- **omnivoice** — k2-fsa's [OmniVoice](https://github.com/k2-fsa/OmniVoice)
  diffusion language model. GPU-recommended (CUDA/XPU/MPS), zero-shot voice
  cloning, 600+ languages, optional Whisper auto-transcription of reference
  clips.
- **qwen3-tts** — Alibaba's [Qwen3-TTS-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
  via [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
  (CUDA-graph capture, no flash-attn / vLLM / Triton). CUDA-only, zero-shot
  voice cloning, native streaming, 10 languages. Optional RTF-aware jitter
  buffer for sub-realtime hardware.

The daemon-↔-provider protocol is provider-agnostic so other engines can be plugged in later.

## Why

KDE Plasma 6 routes TTS through `qt6-qtspeech-speechd` → `speech-dispatcher`.
The default voices (espeak-ng) sound robotic. Kokoro produces dramatically
more natural audio but loads in seconds, so we need a resident daemon to keep
the model warm.

## Architecture

```
KDE app → QtSpeech → speech-dispatcher → sd_generic → bin/sd-neural-tts
                                                          │
                                                  AF_UNIX socket (protobuf framing)
                                                          │
                                              neural-tts.service (resident daemon)
                                                          │
                                              socketpair, NEURAL_TTS_PROVIDER_FD
                                                          │
                                              provider subprocess
                                              (its own uv venv, model warm)
```

- **Wire format**: length-prefixed protobuf (`proto/neural_tts.proto`). Audio is
  streamed as raw PCM in `AudioChunk` messages.
- **Speechd-side wire rate**: always 24 kHz s16le mono; the daemon resamples
  on the fly via `soxr` if the active provider's native rate differs.
- **Provider isolation**: each provider is its own uv project with its own
  `.venv/`. Only one provider subprocess is alive at a time.
- **Auto-routing**: the daemon keeps an in-memory voice index mapping every
  voice id to its owning provider. The first LIST VOICES from speechd populates
  it by spawning each enabled provider in *lazy* mode (no model load) to
  enumerate voices, then shutting it down. On synth, the daemon looks up the
  voice in the index and ensures the owning provider is running and warm.
- **Lazy vs eager warmup**: providers default to lazy — they enumerate voices
  in milliseconds, deferring model load until first synth. The daemon spawns
  them eagerly (`--eager-startup`) for the synth path so the model is loaded
  before any audio commitment is made to speechd. Run `bin/neural-tts-ctl
  reload-voices` to rebuild the voice index after dropping reference clips
  or installing a new provider.
- **Idle eviction**: the daemon unloads its provider after
  `supervisor.idle_timeout_seconds` (default 600). Next request re-spawns.

## Prerequisites

- Linux + speech-dispatcher 0.12+
- KDE Plasma 6 with `qt6-qtspeech-speechd`
- `pulseaudio-utils` or `pipewire-pulse` (provides `paplay`)
- `espeak-ng` (Kokoro uses it via the `misaki` G2P library)
- [`mise`](https://mise.jdx.dev/) to manage tools and run tasks

On Fedora 44:

```bash
sudo dnf install speech-dispatcher pulseaudio-utils espeak-ng \
                 qt6-qtspeech-speechd
```

## Quickstart

```bash
# 1. Create the daemon's venv (numpy, soxr, protobuf)
mise run sync-daemon

# 2. Install the Kokoro provider (creates providers/kokoro-onnx/.venv,
#    downloads ~400 MB of model files into ~/.local/share/neural-tts-daemon/models/)
mise run install-provider kokoro-onnx
# (NVIDIA GPU only) add CUDA runtime + fp16-gpu model:
#    mise run install-provider kokoro-onnx --extra gpu
#    requires the NVIDIA driver + xorg-x11-drv-nvidia-cuda-libs (or system CUDA).
#    Re-running with --extra gpu after a CPU install upgrades in place and
#    downloads the GPU-quantised fp16 model that the daemon prefers on CUDA.

# 3. Install the systemd units and speechd module config (no sudo needed)
mise run install

# 4. Verify
bin/neural-tts-ctl status
spd-say -o neural-tts-generic "Speech dispatcher route working"
```

Then in KDE: **System Settings → Accessibility → Text-to-Speech**,
select *Speech Dispatcher*, pick a kokoro voice (`af_heart`, `am_adam`, …).

## Tasks

| `mise run …` | Purpose |
|---|---|
| `sync-daemon` | Create the daemon venv |
| `install-provider <name>` | Sync a provider's venv (e.g. `kokoro-onnx`) |
| `download-models [name]` | Re-fetch model files for a provider |
| `install` | Install systemd units + speechd module config |
| `uninstall` | Remove user-scope unit files and module config |
| `run` | Run daemon in the foreground (binds sockets itself) |
| `status` | `neural-tts-ctl status` |
| `reload-voices` | Rebuild the global voice index (re-enumerate every enabled provider) |
| `voices` | List speechd-visible voices |
| `logs` | `journalctl --user -u neural-tts.service -f` |
| `test` | Run pytest |
| `gen-proto` | Regenerate `*_pb2.py` from `proto/neural_tts.proto` |

## Files installed (user scope)

```
~/.config/systemd/user/
    neural-tts.service             ← daemon
    neural-tts.socket              ← synthesis socket  ($XDG_RUNTIME_DIR/neural-tts.sock)
    neural-tts-control.socket      ← control socket    ($XDG_RUNTIME_DIR/neural-tts-control.sock)
~/.config/speech-dispatcher/
    speechd.conf                ← seeded from /etc/, with AddModule line appended
    modules/neural-tts-generic.conf
~/.config/neural-tts-daemon/
    config.toml                 ← default provider + supervisor settings
~/.local/share/neural-tts-daemon/
    models/                     ← Kokoro ONNX model files (~400 MB)
    voices/                     ← cloned voice reference clips (future)
```

## Configuration

`~/.config/neural-tts-daemon/config.toml`:

```toml
[provider]
default = "kokoro-onnx"             # vestigial: only used when `eager_startup`
                                    # below is true; daemon auto-routes by voice id otherwise.
enabled = ["kokoro-onnx"]           # allowlist; providers not listed here are
                                    # invisible to the daemon. Add other names
                                    # (e.g. "moss-tts-nano") after installing them.

[supervisor]
idle_timeout_seconds = 600   # unload provider after this many seconds idle (0 = never)
eager_startup = false        # if true, pre-spawn `provider.default` with its model
                             # loaded at daemon start. Otherwise providers spawn
                             # on demand when a synth request arrives.

# Optional per-provider settings. Section name matches the provider's
# registry name (use TOML's bracket form for hyphens, as below).
[providers.kokoro-onnx]
voices = ["bm_daniel", "bm_lewis"]   # voice-id allowlist; omit or set to []
                                     # to surface every voice the provider
                                     # reports. Filtering happens during
                                     # enumeration, so dropped voices never
                                     # reach speechd or its clients.
```

Providers are opt-in: a fresh install ships with only `kokoro-onnx` enabled.
After running `mise run install-provider <name>`, add the name to
`[provider] enabled` to make its voices visible to the daemon. The daemon
routes every synth request to the owning provider via the global voice
index — there's no manual "switch to provider X" step.

Per-provider voice allowlists (`[providers.<name>] voices = [...]`) let
you trim a provider's voice list before it reaches speechd — useful for
kokoro's 54-voice catalogue when you only want a handful surfaced in
Firefox/Okular pickers. Use `spd-say -o neural-tts --list-synthesis-voices`
to see voice ids, then list the ones you want to keep. After editing the
config, restart the daemon (`systemctl --user restart neural-tts.service`)
to re-enumerate.

Per-provider environment variables (`[providers.<name>] env = { ... }`)
get injected into the provider's subprocess on spawn. Use this to set
provider knobs without editing a systemd drop-in:

```toml
[providers.omnivoice]
env = { TTS_OMNIVOICE_NUM_STEP = "24", TTS_OMNIVOICE_DEVICE = "cuda:0" }
```

Values must be strings (TOML's typed values get coerced). The env table
overrides any same-named variable from the daemon's own environment.

Environment overrides (set in a service drop-in):

| Var | Default | Effect |
|---|---|---|
| `TTS_KOKORO_MODEL_PATH` | auto (fp16-gpu on GPU, int8 on CPU) | Pin a specific Kokoro model file |
| `TTS_KOKORO_VOICES_PATH` | `~/.local/share/neural-tts-daemon/models/voices-v1.0.bin` | Pin a voices file |
| `NEURAL_TTS_LOG_LEVEL` | `INFO` | Daemon and provider log level |

## LongCat-AudioDiT (zero-shot voice cloning)

> **Status: untested.** The LongCat provider builds and follows the same
> daemon/provider protocol as the others, but it has not been end-to-end
> verified — the 1B model needs more VRAM than the dev box has. Treat the
> install and synth paths here as unproven; expect rough edges (notably:
> the model download is not resumable after a network interruption, so
> reinstall on failure).

LongCat-AudioDiT is a diffusion TTS that clones any voice from a short
reference clip — no fine-tuning, no per-voice training. It's CUDA-only and
the 1B model needs roughly 5–6 GB VRAM at fp16.

```bash
# 1. Install the provider (creates providers/longcat-audiodit/.venv,
#    downloads ~4 GB of HuggingFace model files)
mise run install-provider longcat-audiodit --extra gpu

# 2. Drop reference clips into the user-voices dir. Each "voice" is a
#    pair of files:
#      <voice-id>.<lang>.wav   ← 5-15 s clean reference audio
#      <voice-id>.<lang>.txt   ← exact transcript of that clip (UTF-8)
#    lang must be `en` or `zh`. Optional sidecar:
#      <voice-id>.<lang>.toml  ← { display_name = "...", gender = "female" }
mkdir -p ~/.local/share/neural-tts-daemon/voices/longcat
cp my-clip.wav ~/.local/share/neural-tts-daemon/voices/longcat/alice.en.wav
echo "the exact words spoken in my clip" \
    > ~/.local/share/neural-tts-daemon/voices/longcat/alice.en.txt

# 3. Refresh the global voice index (only needed after dropping new clips)
bin/neural-tts-ctl reload-voices

# 4. Speak — the daemon auto-routes by voice id; no explicit switch needed
spd-say -o neural-tts -y alice "Hello world from LongCat."
```

Limitations: English + Chinese only; the daemon's `speed` knob is ignored
(LongCat has no native speed control); long input is sentence-chunked, so
time-to-first-audio scales with the first chunk, not the full text.

## MOSS-TTS-Nano (CPU-friendly zero-shot voice cloning)

MOSS-TTS-Nano is a 100M-parameter autoregressive TTS shipped as ONNX. No
torch GPU required at runtime — upstream claims realtime on 4 CPU cores.
Voice cloning is zero-shot from a single reference clip; no transcript is
needed (the model conditions on audio tokens, not text).

```bash
# 1. Install the provider — clones the pinned upstream into
#    providers/moss-tts-nano/vendor/, creates its .venv, and downloads
#    both ONNX repos (~few hundred MB) into ~/.local/share/neural-tts-daemon/models/.
mise run install-provider moss-tts-nano
#    The [gpu] extra exists but is NOT recommended for this model:
#    a 100M-param autoregressive ONNX session on GPU loses to CPU on most
#    hardware (per-token CUDA launches + memcpy overhead exceed the compute
#    savings). On one local test, CPU ran 2.4× realtime; GPU ran 1.0×.
#    To force the choice anyway: TTS_MOSS_TTS_NANO_EP=cpu|cuda (env var).

# 2. Drop reference clips into the user-voices dir. Each "voice" is one wav:
#      <voice-id>.<lang>.wav    ← 5-15 s clean reference audio
#    lang is the short tag for one of the 20 supported languages:
#      zh, en, de, es, fr, ja, it, hu, ko, ru, fa, ar, pl, pt, cs, da, sv, el, tr
#    Optional sidecar:
#      <voice-id>.<lang>.toml   ← { display_name = "...", gender = "female" }
mkdir -p ~/.local/share/neural-tts-daemon/voices/moss-tts-nano
cp my-clip.wav ~/.local/share/neural-tts-daemon/voices/moss-tts-nano/alice.en.wav

# 3. Refresh the global voice index (only needed after dropping new clips)
bin/neural-tts-ctl reload-voices

# 4. Speak — the daemon auto-routes by voice id; no explicit switch needed
spd-say -o neural-tts -y alice "Hello world from MOSS-TTS-Nano."
```

Limitations: the daemon's `speed` knob is ignored (no native speed control);
input is chunked by token budget so time-to-first-audio scales with the
first chunk; WeTextProcessing-based text normalisation is intentionally
disabled (it requires `pynini`, which doesn't install cleanly under `uv`),
so very heavy numeric/abbreviation input may sound less polished than via
upstream's CLI.

## OmniVoice (massively multilingual zero-shot cloning)

OmniVoice is a diffusion language model TTS from k2-fsa (Apache-2.0). It
clones voices zero-shot from a short reference clip, supports 600+
languages, and runs on CUDA, Intel Arc (XPU), Apple Silicon (MPS), or CPU
via PyTorch. The 1B+ model is too large for realtime CPU synthesis; a GPU
is strongly recommended.

```bash
# 1. Install the provider — pulls a pinned commit of upstream from GitHub
#    via uv, then snapshot-downloads the HuggingFace model weights
#    (~few GB) into ~/.local/share/neural-tts-daemon/models/omnivoice/.
mise run install-provider omnivoice --extra gpu
#    Omit --extra gpu to pull the CPU torch wheel instead (much smaller
#    download, but synthesis will be well below realtime).

# 2. Drop reference clips into the user-voices dir. Each "voice" is one wav:
#      <voice-id>.<lang>.wav    ← 3-10 s clean reference audio
#    <lang> is a BCP-47 primary subtag (en, fr, zh, sw, …) — OmniVoice
#    handles 600+ languages, we just pass it through. Optional sidecars:
#      <voice-id>.<lang>.txt    ← manual transcript (overrides Whisper)
#      <voice-id>.<lang>.toml   ← { display_name = "...", gender = "female" }
mkdir -p ~/.local/share/neural-tts-daemon/voices/omnivoice
cp my-clip.wav ~/.local/share/neural-tts-daemon/voices/omnivoice/alice.en.wav

# 3. Refresh the global voice index (only needed after dropping new clips)
bin/neural-tts-ctl reload-voices

# 4. Speak — the daemon auto-routes by voice id; no explicit switch needed
spd-say -o neural-tts -y alice "Hello world from OmniVoice."
```

You can reuse a reference clip from another cloning provider by
symlinking it into the omnivoice voices dir, e.g.:

```bash
ln -s ../moss-tts-nano/hobbits.en.wav \
      ~/.local/share/neural-tts-daemon/voices/omnivoice/hobbits.en.wav
```

Environment overrides:

| Var | Default | Effect |
|---|---|---|
| `TTS_OMNIVOICE_DEVICE` | auto (cuda → xpu → mps → cpu) | Pin a specific torch device string |
| `TTS_OMNIVOICE_MODEL_PATH` | `~/.local/share/neural-tts-daemon/models/omnivoice` | Pin a local model snapshot dir |
| `TTS_OMNIVOICE_NUM_STEP` | `16` | Diffusion steps per utterance. Quality vs latency knob — upstream default is 32 (better fidelity); 16 is the README's faster-inference value. Lower = lower TTFA + per-chunk synth time; higher = cleaner audio. |
| `TTS_OMNIVOICE_COMPILE` | `` (off) | Wrap the model in `torch.compile()` for ~20-40% faster steady-state per diffusion step. Costs 30-60 s extra on the first synth (JIT compile). Accepts `1`/`true`/`on` (= `default` mode), or one of `default`, `reduce-overhead`, `max-autotune`. `reduce-overhead` is fastest but uses CUDA graphs that pin tensor shapes — if you hit recompile storms or shape errors, drop to `default`. Compile failures fall back to eager with a warning. |

Limitations: no native streaming API upstream, so input is sentence-chunked
and emitted per chunk (time-to-first-audio scales with the first chunk).
On a fresh voice without a transcript sidecar, the first synthesis pays a
one-time Whisper transcription cost; subsequent calls reuse it from
in-memory cache.

## Qwen3-TTS (streaming zero-shot cloning, CUDA-only)

Qwen3-TTS-0.6B via [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) —
a hand-written CUDA-graph capture of Qwen3-TTS's predictor + talker that
hits ~4.8× realtime on an RTX 4090 (with ~150 ms TTFA) and well into
multi-x on smaller cards too. No flash-attn, no vLLM, no Triton; just
`torch.cuda.CUDAGraph` over a static KV cache. Streams natively, so by
default the provider passes each model chunk straight to the daemon.

Voice cloning requires both a reference WAV and a transcript sidecar (no
transcript-less mode in v1). 10 languages.

```bash
# 1. Install the provider — pulls faster-qwen3-tts from PyPI and
#    snapshot-downloads the HF model (~1.5 GB) into
#    ~/.local/share/neural-tts-daemon/models/qwen3-tts-0.6b/.
mise run install-provider qwen3-tts --extra gpu

# 2. Drop reference clips into the user-voices dir. Each "voice" is two
#    files (transcript is REQUIRED):
#      <voice-id>.<lang>.wav    ← 3-10 s clean reference audio
#      <voice-id>.<lang>.txt    ← exact transcript of the wav
#    <lang> must be one of: en, zh, ja, ko, de, fr, ru, pt, es, it.
#    Optional sidecar:
#      <voice-id>.<lang>.toml   ← { display_name = "...", gender = "female" }
mkdir -p ~/.local/share/neural-tts-daemon/voices/qwen3-tts
cp my-clip.wav ~/.local/share/neural-tts-daemon/voices/qwen3-tts/alice.en.wav
echo "exact transcript of my-clip" \
  > ~/.local/share/neural-tts-daemon/voices/qwen3-tts/alice.en.txt

# 3. Refresh the global voice index (only needed after dropping new clips)
bin/neural-tts-ctl reload-voices

# 4. Speak
spd-say -o neural-tts -y alice "Hello world from Qwen3-TTS."
```

Environment overrides:

| Var | Default | Effect |
|---|---|---|
| `TTS_QWEN3_DEVICE` | auto (`cuda` if available) | Pin device, e.g. `cuda:1`. CPU is unsupported by faster-qwen3-tts. |
| `TTS_QWEN3_MODEL_PATH` | `~/.local/share/neural-tts-daemon/models/qwen3-tts-0.6b` | Pin a local model snapshot dir |
| `TTS_QWEN3_DTYPE` | `bf16` | One of `bf16`, `fp16`, `fp32`. bf16 is the upstream sweet spot. |
| `TTS_QWEN3_ATTN` | `sdpa` | `sdpa` or `flash_attention_2`. Flash-attn isn't installed by default; sdpa is fine. |
| `TTS_QWEN3_MAX_SEQ_LEN` | `2048` | KV-cache capacity. Bigger = more VRAM. |
| `TTS_QWEN3_GREEDY` | `` (off) | `1` switches from sampled to greedy decoding: always pick the highest-probability next token. Modest speedup, fully deterministic output, slightly less prosodic variation. Usually a win for voice-clone TTS because the reference clip pins the voice character; drop it if synthesis sounds robotic. |
| `TTS_QWEN3_TEMPERATURE` | `0.9` | Sampled-decoding temperature. Ignored when `GREEDY=1`. |
| `TTS_QWEN3_TOP_K` | `50` | Sampled-decoding top-k. Ignored when `GREEDY=1`. |
| `TTS_QWEN3_TOP_P` | `1.0` | Sampled-decoding top-p. Ignored when `GREEDY=1`. |
| `TTS_QWEN3_REPETITION_PENALTY` | `1.05` | Repetition penalty applied to the predictor. |
| `TTS_QWEN3_CHUNK_SIZE` | `12` | Codec-frame batching per yield from the streaming generator. Lower = lower TTFA, slightly more overhead per yield. |
| `TTS_QWEN3_MAX_NEW_TOKENS_PER_CHAR` | `6` | Cap on `max_new_tokens` is `min(2048, 32 + per_char * len(text))`. Lower = tighter cap (faster) but risks truncating long sentences. |

**Optional RTF-aware jitter buffer.** On sub-realtime hardware (sustained
RTF < 1, e.g. an RTX 3060 Laptop on the 0.6B model) raw streaming starves
`paplay` mid-utterance and you get audible gaps. Setting
`TTS_QWEN3_CHUNKER=1` enables a sentence-aware chunker plus an adaptive
prebuffer per chunk: it measures RTF from the model's per-yield timing
dict, estimates how much audio to hold back before emitting, then drains
the buffer and passes through the rest of the chunk. Trades TTFA for
gap-free playback; off by default because faster GPUs don't need it.

| Var | Default | Effect |
|---|---|---|
| `TTS_QWEN3_CHUNKER` | `` (off) | `1` enables the chunker + jitter buffer described above. |
| `TTS_QWEN3_CHUNK_TARGET_CHARS` | `120` | Target characters per chunk. Smaller = lower TTFA + more prosody breaks. |
| `TTS_QWEN3_CHUNK_HARD_CAP_CHARS` | `240` | Hard ceiling per chunk after soft-break fallback. |
| `TTS_QWEN3_JITTER_SAFETY_MS` | `200` | Extra millis added on top of the computed prebuffer to absorb GPU contention spikes. |
| `TTS_QWEN3_JITTER_INITIAL_MS` | `500` | Prebuffer used for the very first chunk before any RTF has been observed. |
| `TTS_QWEN3_CHARS_PER_SEC_BOOTSTRAP` | `15.0` | Initial chars/sec estimate before the first chunk's audio duration calibrates the ratio. |

Math: for a chunk of audio duration D at sustained RTF r < 1, gap-free
playback requires a prebuffer ≥ D·(1−r)/r. At r = 0.5 that's D — i.e.
hold the whole chunk before emitting. Streaming-within-chunk still wins
over full pre-synth because the model keeps producing during playback;
total wall time per chunk is D/r vs D/r + D for full pre-synth, so TTFA
halves.

Limitations: CUDA-required; transcript sidecar mandatory; no
`speed`/style knob; 10 languages only (see `LANG_TO_QWEN` in
`voices.py`); first synth pays a one-time CUDA-graph capture cost
(~10–30 s).

## Adding another provider

A provider is a uv project under `providers/<name>/` with:

1. `pyproject.toml` declaring its deps (torch, ONNX runtime, whatever).
2. A Python module that:
   - Reads `NEURAL_TTS_PROVIDER_FD` env var, adopts that FD as an `AF_UNIX` socket.
   - Runs a framed-protobuf request/response loop (`proto/neural_tts.proto`).
   - Implements `Warmup`, `Synthesize`, `ListVoices`, `Shutdown`.
3. An entry in `providers/registry.toml` mapping the provider name to its
   project dir and Python module name.

Add a future cloning-capable provider's `register_cloned_voice` and
`remove_cloned_voice` ops once the upstream lib is ready.

## Troubleshooting

- **`bin/neural-tts-ctl status` says state=STOPPED for ages**: that's normal —
  the provider only spawns on the first synthesis request (or set
  `eager_startup = true` in config.toml).
- **KDE TTS settings shows no Kokoro voices**: after install you may need to
  restart any open KDE apps so they re-query speech-dispatcher. The install
  script kills running `speech-dispatcher` processes so the next client
  respawn re-reads the new config — but apps with long-lived speechd
  connections (open Okular, etc.) need to be restarted.
- **First utterance after a long pause is slow**: that's the idle-evict
  reload (default 10 min). Increase `idle_timeout_seconds` or set to `0` if
  you have RAM to spare.
- **`uv sync` is slow the first time**: kokoro-onnx pulls onnxruntime
  (~80 MB wheel) and misaki/espeak-loader. Subsequent syncs hit the uv cache.
