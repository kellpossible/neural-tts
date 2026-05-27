# kde-kokoro-tts

A user-systemd daemon that bridges **speech-dispatcher** (and therefore KDE
Plasma's QtSpeech) to neural TTS engines. Ships with **Kokoro-82M** as the
first provider; the daemon-↔-provider protocol is provider-agnostic so other
engines (LongCat-AudioDiT, MOSS-TTS, …) can be plugged in later.

## Why

KDE Plasma 6 routes TTS through `qt6-qtspeech-speechd` → `speech-dispatcher`.
The default voices (espeak-ng) sound robotic. Kokoro produces dramatically
more natural audio but loads in seconds, so we need a resident daemon to keep
the model warm.

## Architecture

```
KDE app → QtSpeech → speech-dispatcher → sd_generic → bin/kde-tts-say
                                                          │
                                                  AF_UNIX socket (protobuf framing)
                                                          │
                                              kde-tts.service (resident daemon)
                                                          │
                                              socketpair, KDE_TTS_PROVIDER_FD
                                                          │
                                              provider subprocess
                                              (its own uv venv, model warm)
```

- **Wire format**: length-prefixed protobuf (`proto/kde_tts.proto`). Audio is
  streamed as raw PCM in `AudioChunk` messages.
- **Speechd-side wire rate**: always 24 kHz s16le mono; the daemon resamples
  on the fly via `soxr` if the active provider's native rate differs.
- **Provider isolation**: each provider is its own uv project with its own
  `.venv/`. Switch providers via `bin/kde-tts-ctl switch <name>`.
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
#    downloads ~400 MB of model files into ~/.local/share/kde-tts-daemon/models/)
mise run install-provider kokoro-onnx
# (NVIDIA GPU only) add CUDA runtime + fp16-gpu model:
#    mise run install-provider kokoro-onnx --extra gpu
#    requires the NVIDIA driver + xorg-x11-drv-nvidia-cuda-libs (or system CUDA).
#    Re-running with --extra gpu after a CPU install upgrades in place and
#    downloads the GPU-quantised fp16 model that the daemon prefers on CUDA.

# 3. Install the systemd units and speechd module config (no sudo needed)
mise run install

# 4. Verify
bin/kde-tts-ctl status
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
| `status` | `kde-tts-ctl status` |
| `switch-provider <name>` | Switch active provider, regenerate AddVoice block |
| `voices` | List speechd-visible voices |
| `logs` | `journalctl --user -u kde-tts.service -f` |
| `test` | Run pytest |
| `gen-proto` | Regenerate `*_pb2.py` from `proto/kde_tts.proto` |

## Files installed (user scope)

```
~/.config/systemd/user/
    kde-tts.service             ← daemon
    kde-tts.socket              ← synthesis socket  ($XDG_RUNTIME_DIR/kde-tts.sock)
    kde-tts-control.socket      ← control socket    ($XDG_RUNTIME_DIR/kde-tts-control.sock)
~/.config/speech-dispatcher/
    speechd.conf                ← seeded from /etc/, with AddModule line appended
    modules/neural-tts-generic.conf
~/.config/kde-tts-daemon/
    config.toml                 ← default provider + supervisor settings
~/.local/share/kde-tts-daemon/
    models/                     ← Kokoro ONNX model files (~400 MB)
    voices/                     ← cloned voice reference clips (future)
```

## Configuration

`~/.config/kde-tts-daemon/config.toml`:

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
| `TTS_KOKORO_VOICES_PATH` | `~/.local/share/kde-tts-daemon/models/voices-v1.0.bin` | Pin a voices file |
| `KDE_TTS_LOG_LEVEL` | `INFO` | Daemon and provider log level |

## Adding another provider

A provider is a uv project under `providers/<name>/` with:

1. `pyproject.toml` declaring its deps (torch, ONNX runtime, whatever).
2. A Python module that:
   - Reads `KDE_TTS_PROVIDER_FD` env var, adopts that FD as an `AF_UNIX` socket.
   - Runs a framed-protobuf request/response loop (`proto/kde_tts.proto`).
   - Implements `Warmup`, `Synthesize`, `ListVoices`, `Shutdown`.
3. An entry in `providers/registry.toml` mapping the provider name to its
   project dir and Python module name.

Add a future cloning-capable provider's `register_cloned_voice` and
`remove_cloned_voice` ops once the upstream lib is ready.

## Troubleshooting

- **`bin/kde-tts-ctl status` says state=STOPPED for ages**: that's normal —
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

## License

TBD.
