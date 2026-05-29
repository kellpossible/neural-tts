"""Provider subprocess lifecycle: spawn, drain, idle-evict, restart.

The supervisor owns a single active provider at any time. It spawns the provider
as a subprocess whose Python interpreter lives in that provider's own venv, and
communicates with it over an inherited AF_UNIX socketpair (child end on fd 3,
LISTEN_FDS=1 in the environment so the child uses the same FD-adoption code as
when running under systemd socket activation).
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import socket
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from .config import providers_registry_path, repo_root
from .pb import neural_tts_pb2 as pb
from .protocol import ProtocolError, read_message, write_message
from .voice_index import VoiceIndex
from .voices import Voice

log = logging.getLogger("neural_tts_daemon.supervisor")


class ProviderState(enum.Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    FAILED = "FAILED"


class ProviderNotInstalled(Exception):
    def __init__(self, name: str):
        super().__init__(
            f"provider {name!r} venv not found; run "
            f"`mise run install-provider {name}` to install it"
        )
        self.name = name


class ProviderUnknown(Exception):
    pass


_GENDER_FROM_PB = {
    pb.GENDER_UNSPECIFIED: "NEUTRAL",
    pb.MALE: "MALE",
    pb.FEMALE: "FEMALE",
    pb.NEUTRAL: "NEUTRAL",
}


def _voice_from_pb(v: pb.Voice) -> Voice:
    return Voice(
        id=v.id,
        language=v.language,
        gender=_GENDER_FROM_PB.get(v.gender, "NEUTRAL"),
        display_name=v.display_name or None,
    )


@dataclass
class ProviderMeta:
    name: str
    display_name: str
    project_dir: Path
    python_module: str
    supports_cloning: bool
    needs_models: bool
    notes: str = ""

    @property
    def venv_python(self) -> Path:
        return self.project_dir / ".venv" / "bin" / "python"

    @property
    def installed(self) -> bool:
        return self.venv_python.exists()


def load_registry() -> dict[str, ProviderMeta]:
    path = providers_registry_path()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    out: dict[str, ProviderMeta] = {}
    repo = repo_root()
    for name, fields in raw.get("providers", {}).items():
        out[name] = ProviderMeta(
            name=name,
            display_name=fields.get("display_name", name),
            project_dir=repo / fields["project_dir"],
            python_module=fields["python_module"],
            supports_cloning=fields.get("supports_cloning", False),
            needs_models=fields.get("needs_models", False),
            notes=fields.get("notes", ""),
        )
    return out


@dataclass
class _ProviderProcess:
    meta: ProviderMeta
    process: asyncio.subprocess.Process
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    parent_sock: socket.socket
    sample_rate: int = 0
    voices: list[Voice] = field(default_factory=list)
    state: ProviderState = ProviderState.STARTING
    last_activity: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.monotonic)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set whenever a stream isn't fully drained to audio_end (e.g. consumer
    # cancellation). The *next* request drains residual frames before its own
    # exchange — this is more robust than draining inline in a finally block,
    # which can itself be cancelled and leave the buffer dirty.
    needs_drain: bool = False


class Supervisor:
    """Manages the active provider subprocess."""

    def __init__(
        self,
        default_provider: str,
        idle_timeout_seconds: int,
        enabled_providers: list[str] | None = None,
        eager_startup: bool = False,
        voice_allowlists: dict[str, list[str]] | None = None,
        provider_envs: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.default_provider = default_provider
        self.idle_timeout_seconds = idle_timeout_seconds
        self.eager_startup = eager_startup
        # provider_name → frozenset of voice ids. Empty set = no filter.
        self._voice_allowlists = {
            name: frozenset(ids) for name, ids in (voice_allowlists or {}).items() if ids
        }
        # provider_name → {ENV_VAR: value}. Merged into the subprocess env at
        # spawn so users can set provider knobs (e.g. TTS_OMNIVOICE_NUM_STEP)
        # via config.toml's `[providers.<name>] env = { ... }` table without
        # touching systemd drop-ins.
        self._provider_envs = {
            name: dict(env) for name, env in (provider_envs or {}).items() if env
        }
        full_registry = load_registry()
        allowed = set(enabled_providers or [])
        for name in allowed:
            if name not in full_registry:
                log.warning(
                    "config enables unknown provider %r — ignoring (known: %s)",
                    name, sorted(full_registry),
                )
        self._registry = {n: m for n, m in full_registry.items() if n in allowed}
        if not self._registry:
            log.warning(
                "no providers enabled — set [provider] enabled = [...] in "
                "~/.config/neural-tts-daemon/config.toml (known providers: %s)",
                sorted(full_registry),
            )
        elif default_provider not in self._registry:
            fallback = next(iter(self._registry))
            log.warning(
                "default provider %r is not in enabled list; falling back to %r",
                default_provider, fallback,
            )
            self.default_provider = fallback
        self._active: _ProviderProcess | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._closing = False
        self._voice_index = VoiceIndex(sorted(self._registry))

    # ── voice index access ─────────────────────────────────────────────

    def voice_index(self) -> VoiceIndex:
        return self._voice_index

    async def ensure_voice_index_populated(self) -> VoiceIndex:
        """Populate the in-memory voice index if it's empty. Idempotent."""
        if not self._voice_index.is_empty():
            return self._voice_index
        await self.enumerate_all_voices()
        return self._voice_index

    async def enumerate_all_voices(self) -> dict[str, list[Voice]]:
        """For each enabled provider: reuse if already active+READY, otherwise
        spawn it briefly in lazy mode, read voices from the warmup response,
        then shut it down."""
        out: dict[str, list[Voice]] = {}
        async with self._lifecycle_lock:
            for name in sorted(self._registry):
                meta = self._registry[name]
                if not meta.installed:
                    log.warning("enumerate: skipping %s (not installed)", name)
                    continue
                if (
                    self._active
                    and self._active.meta.name == name
                    and self._active.state == ProviderState.READY
                ):
                    log.info(
                        "enumerate: reusing active provider %s (%d voices)",
                        name, len(self._active.voices),
                    )
                    out[name] = list(self._active.voices)
                    continue
                log.info("enumerate: spawning %s briefly to list voices", name)
                spawned_here = False
                try:
                    await self._spawn_locked(name, eager=False)
                    spawned_here = True
                    assert self._active is not None
                    out[name] = list(self._active.voices)
                except Exception:
                    log.exception("enumerate: provider %s failed", name)
                finally:
                    if spawned_here and self._active and self._active.meta.name == name:
                        try:
                            await self._shutdown_locked()
                        except Exception:
                            log.exception("enumerate: shutdown of %s failed", name)

        # Apply per-provider voice allowlists, if configured.
        for name, voices in list(out.items()):
            allow = self._voice_allowlists.get(name)
            if allow:
                filtered = [v for v in voices if v.id in allow]
                if len(filtered) != len(voices):
                    log.info(
                        "voice allowlist for %s: %d/%d voices kept (%s)",
                        name, len(filtered), len(voices), sorted(allow),
                    )
                out[name] = filtered

        # Populate the index from what we got.
        self._voice_index.clear()
        for name, voices in out.items():
            self._voice_index.set_provider_voices(name, voices)
        return out

    def known_providers(self) -> list[ProviderMeta]:
        return list(self._registry.values())

    def active_name(self) -> str | None:
        return self._active.meta.name if self._active else None

    def active_state(self) -> ProviderState:
        return self._active.state if self._active else ProviderState.STOPPED

    def active_voices(self) -> list[Voice]:
        return list(self._active.voices) if self._active else []

    def active_sample_rate(self) -> int:
        return self._active.sample_rate if self._active else 0

    def idle_for_seconds(self) -> float:
        if not self._active:
            return 0.0
        return max(0.0, time.monotonic() - self._active.last_activity)

    def uptime_seconds(self) -> float:
        if not self._active:
            return 0.0
        return max(0.0, time.monotonic() - self._active.started_at)

    async def start_idle_loop(self) -> None:
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def ensure_ready(self, name: str | None = None) -> _ProviderProcess:
        """Spawn `name` (or the current active or default) and load its model.

        Always eager: the provider's model is fully loaded before this returns.
        Otherwise speechd's first-audio timeout fires during a lazy model load.
        For enumerate-only use cases see `enumerate_all_voices` which spawns
        explicitly lazy and tears down immediately after.
        """
        target = name or (self._active.meta.name if self._active else self.default_provider)
        async with self._lifecycle_lock:
            if (
                self._active
                and self._active.meta.name == target
                and self._active.state == ProviderState.READY
            ):
                return self._active
            if self._active and self._active.meta.name != target:
                await self._shutdown_locked()
            if not self._active or self._active.state != ProviderState.READY:
                await self._spawn_locked(target, eager=True)
            assert self._active is not None
            return self._active

    async def synthesize(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[bytes]:
        """Send a synthesize request; yields raw float32 PCM chunks at provider's native rate."""
        proc = await self.ensure_ready()
        # `voice` is the public id (potentially `<id>.<provider>` if the index
        # disambiguated a cross-provider collision). The provider only knows
        # the bare id — translate before forwarding.
        local_voice = self._voice_index.local_id_for(voice)
        if local_voice not in {v.id for v in proc.voices}:
            raise ValueError(f"unknown voice {voice!r}")
        proc.last_activity = time.monotonic()

        async with proc.request_lock:
            await self._drain_if_needed(proc)

            req = pb.Request(
                synthesize=pb.SynthesizeRequest(
                    voice=local_voice, speed=speed, lang=lang, text=text
                )
            )
            # Mark dirty: from this point on, the stream may carry frames the
            # provider hasn't sent yet. If the consumer aborts before audio_end,
            # the flag remains set so the *next* request drains first.
            proc.needs_drain = True
            write_message(proc.writer, req)
            await proc.writer.drain()

            header = await read_message(proc.reader, pb.Response)
            body = header.WhichOneof("body")
            if body == "error":
                # Provider already finished (no more frames expected).
                proc.needs_drain = False
                raise RuntimeError(f"provider error: {header.error.message}")
            if body != "synthesize_header":
                raise RuntimeError(f"unexpected response: {body}")

            while True:
                resp = await read_message(proc.reader, pb.Response)
                body = resp.WhichOneof("body")
                proc.last_activity = time.monotonic()
                if body == "audio_chunk":
                    yield resp.audio_chunk.pcm
                elif body == "audio_end":
                    proc.needs_drain = False
                    return
                elif body == "error":
                    # Mid-stream error: assume audio_end will follow.
                    log.warning("provider error mid-stream: %s", resp.error.message)
                    # Keep needs_drain=True; the *next* request drains.
                    raise RuntimeError(
                        f"provider error mid-stream: {resp.error.message}"
                    )
                else:
                    log.warning("unexpected mid-stream response: %s", body)

    async def _drain_if_needed(self, proc: "_ProviderProcess") -> None:
        """If a prior request didn't reach audio_end, read residual frames."""
        if not proc.needs_drain:
            return
        log.info("draining residual frames from provider %s", proc.meta.name)
        try:
            while True:
                resp = await asyncio.wait_for(
                    read_message(proc.reader, pb.Response), timeout=5.0
                )
                if resp.WhichOneof("body") == "audio_end":
                    break
        except (asyncio.TimeoutError, ProtocolError) as e:
            log.warning("drain didn't see audio_end (%s); marking provider FAILED", e)
            # If the buffer is so out-of-sync that drain itself fails, the
            # only safe recovery is to restart the provider on next request.
            proc.state = ProviderState.FAILED
        finally:
            proc.needs_drain = False

    async def switch(self, new_name: str) -> None:
        if new_name not in self._registry:
            raise ProviderUnknown(f"unknown provider {new_name!r}")
        meta = self._registry[new_name]
        if not meta.installed:
            raise ProviderNotInstalled(new_name)
        async with self._lifecycle_lock:
            if (
                self._active
                and self._active.meta.name == new_name
                and self._active.state == ProviderState.READY
            ):
                return
            if self._active:
                await self._shutdown_locked()
            await self._spawn_locked(new_name)
            self.default_provider = new_name

    async def reload_voices(self) -> list[Voice]:
        proc = await self.ensure_ready()
        async with proc.request_lock:
            await self._drain_if_needed(proc)
            write_message(proc.writer, pb.Request(list_voices=pb.ListVoicesRequest()))
            await proc.writer.drain()
            resp = await read_message(proc.reader, pb.Response)
            body = resp.WhichOneof("body")
            if body == "error":
                raise RuntimeError(f"provider error: {resp.error.message}")
            if body != "list_voices":
                raise RuntimeError(f"unexpected response: {body}")
            voices = [_voice_from_pb(v) for v in resp.list_voices.voices]
            proc.voices = voices
            return voices

    async def shutdown(self) -> None:
        self._closing = True
        if self._idle_task is not None:
            self._idle_task.cancel()
        async with self._lifecycle_lock:
            if self._active:
                await self._shutdown_locked()

    # ---- internal --------------------------------------------------------

    async def _spawn_locked(self, name: str, *, eager: bool | None = None) -> None:
        if name not in self._registry:
            raise ProviderUnknown(f"unknown provider {name!r}")
        meta = self._registry[name]
        if not meta.installed:
            raise ProviderNotInstalled(name)

        # If the caller didn't specify, use the daemon's default.
        if eager is None:
            eager = self.eager_startup

        log.info(
            "spawning provider %s from %s (eager=%s)",
            name, meta.venv_python, eager,
        )

        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_fd = child_sock.fileno()
        try:
            env = {
                **os.environ,
                **self._provider_envs.get(name, {}),
                "NEURAL_TTS_PROVIDER_FD": str(child_fd),
                "PYTHONUNBUFFERED": "1",
            }
            # Strip systemd-side socket-activation vars; our daemon↔provider
            # channel uses NEURAL_TTS_PROVIDER_FD rather than LISTEN_FDS so we
            # don't have to fight CPython's post-preexec _close_open_fds pass.
            env.pop("LISTEN_FDS", None)
            env.pop("LISTEN_PID", None)
            env.pop("LISTEN_FDNAMES", None)

            argv = [str(meta.venv_python), "-m", meta.python_module]
            if eager:
                argv.append("--eager-startup")
            process = await asyncio.create_subprocess_exec(
                *argv,
                pass_fds=(child_fd,),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
            )
        finally:
            child_sock.close()

        reader, writer = await asyncio.open_unix_connection(sock=parent_sock)

        proc = _ProviderProcess(
            meta=meta,
            process=process,
            reader=reader,
            writer=writer,
            parent_sock=parent_sock,
        )
        self._active = proc

        # BaseException (not Exception) so a CancelledError from the caller —
        # e.g. speechd disconnecting mid-warmup — still tears down the child.
        # Without this the subprocess is orphaned holding GPU memory, and the
        # next ensure_ready spawns a second one that CUDA-OOMs against it.
        try:
            await asyncio.wait_for(self._warmup(proc), timeout=120.0)
        except BaseException:
            log.exception("provider %s warmup failed", name)
            proc.state = ProviderState.FAILED
            # Shield kill so an in-flight cancellation can't orphan the child.
            await asyncio.shield(self._kill(proc))
            self._active = None
            raise

    async def _warmup(self, proc: _ProviderProcess) -> None:
        write_message(proc.writer, pb.Request(warmup=pb.WarmupRequest()))
        await proc.writer.drain()
        resp = await read_message(proc.reader, pb.Response)
        body = resp.WhichOneof("body")
        if body == "error":
            raise RuntimeError(f"warmup failed: {resp.error.message}")
        if body != "warmup":
            raise RuntimeError(f"unexpected warmup response: {body}")
        proc.sample_rate = int(resp.warmup.sample_rate)
        proc.voices = [_voice_from_pb(v) for v in resp.warmup.voices]
        proc.state = ProviderState.READY
        log.info(
            "provider %s ready: sample_rate=%d voices=%d",
            proc.meta.name,
            proc.sample_rate,
            len(proc.voices),
        )

    async def _shutdown_locked(self) -> None:
        proc = self._active
        if not proc:
            return
        proc.state = ProviderState.SHUTTING_DOWN
        log.info("shutting down provider %s", proc.meta.name)
        # Shield the actual teardown: if our caller is cancelled mid-shutdown
        # we still need the child reaped, otherwise it stays alive holding GPU
        # memory. Clearing _active happens in finally so the next caller never
        # sees a half-dead reference.
        try:
            await asyncio.shield(self._do_shutdown(proc))
        finally:
            self._active = None

    async def _do_shutdown(self, proc: _ProviderProcess) -> None:
        try:
            write_message(proc.writer, pb.Request(shutdown=pb.ShutdownRequest()))
            await proc.writer.drain()
            try:
                await asyncio.wait_for(read_message(proc.reader, pb.Response), timeout=2.0)
            except (asyncio.TimeoutError, ProtocolError):
                pass
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        try:
            await asyncio.wait_for(proc.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            await self._kill(proc)
        proc.state = ProviderState.STOPPED
        try:
            proc.writer.close()
        except Exception:
            pass
        try:
            proc.parent_sock.close()
        except Exception:
            pass

    async def _kill(self, proc: _ProviderProcess) -> None:
        try:
            proc.process.terminate()
            await asyncio.wait_for(proc.process.wait(), timeout=5.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                proc.process.kill()
                await proc.process.wait()
            except ProcessLookupError:
                pass

    async def _idle_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(30.0)
                if self.idle_timeout_seconds <= 0:
                    continue
                if not self._active or self._active.state != ProviderState.READY:
                    continue
                if self.idle_for_seconds() >= self.idle_timeout_seconds:
                    log.info(
                        "idle timeout reached (%.1fs >= %ds), unloading provider",
                        self.idle_for_seconds(),
                        self.idle_timeout_seconds,
                    )
                    async with self._lifecycle_lock:
                        if (
                            self._active
                            and self.idle_for_seconds() >= self.idle_timeout_seconds
                        ):
                            await self._shutdown_locked()
        except asyncio.CancelledError:
            pass
