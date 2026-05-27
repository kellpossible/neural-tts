# neural-tts

A bridge between operating system TTS interface and various newly available TTS libraries.

Currently `speech-dispatcher` on Linux is supported, with two providers:

- **kokoro-onnx** — Kokoro-82M via ONNX runtime. CPU or GPU, dozens of
  pre-baked voices, multilingual.
- **longcat-audiodit** — Meituan's [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)
  1B diffusion model. CUDA-only, zero-shot voice cloning from user-supplied
  reference clips, English + Chinese.

The daemon-↔-provider protocol is provider-agnostic so other engines (LongCat-AudioDiT, MOSS-TTS, …) can be plugged in later.

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
  `.venv/`. Switch providers via `bin/neural-tts-ctl switch <name>`.
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
| `switch-provider <name>` | Switch active provider, regenerate AddVoice block |
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
default = "kokoro-onnx"      # which provider to spawn on first request

[supervisor]
idle_timeout_seconds = 600   # unload provider after this many seconds idle (0 = never)
eager_startup = false         # spawn provider on daemon start (don't wait for first request)
```

Environment overrides (set in a service drop-in):

| Var | Default | Effect |
|---|---|---|
| `TTS_KOKORO_MODEL_PATH` | auto (fp16-gpu on GPU, int8 on CPU) | Pin a specific Kokoro model file |
| `TTS_KOKORO_VOICES_PATH` | `~/.local/share/neural-tts-daemon/models/voices-v1.0.bin` | Pin a voices file |
| `NEURAL_TTS_LOG_LEVEL` | `INFO` | Daemon and provider log level |

## LongCat-AudioDiT (zero-shot voice cloning)

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

# 3. Switch the active provider and reload speechd's voice list
bin/neural-tts-ctl switch longcat-audiodit
bin/neural-tts-ctl reload-voices

# 4. Speak
spd-say -o neural-tts -y alice "Hello world from LongCat."
```

Limitations: English + Chinese only; the daemon's `speed` knob is ignored
(LongCat has no native speed control); long input is sentence-chunked, so
time-to-first-audio scales with the first chunk, not the full text.

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
