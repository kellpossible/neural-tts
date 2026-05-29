"""Daemon main: accept systemd-passed sockets, run speechd + control servers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
from pathlib import Path

from . import sd_notify
from .config import (
    CONFIG_DIR,
    CONTROL_SOCKET_PATH,
    SPEECHD_SOCKET_PATH,
    DaemonConfig,
)
from .control import handle_control_connection
from .proxy import handle_speechd_connection
from .supervisor import Supervisor

SD_LISTEN_FDS_START = 3

log = logging.getLogger("neural_tts_daemon")


def _adopt_systemd_sockets() -> dict[str, socket.socket]:
    """Read LISTEN_FDS / LISTEN_FDNAMES, return {name: socket}.

    Returns empty dict if not running under socket activation.
    """
    listen_fds = int(os.environ.get("LISTEN_FDS", "0"))
    if listen_fds <= 0:
        return {}

    listen_pid = os.environ.get("LISTEN_PID")
    if listen_pid and int(listen_pid) != os.getpid():
        return {}

    names = os.environ.get("LISTEN_FDNAMES", "").split(":") if "LISTEN_FDNAMES" in os.environ else []
    out: dict[str, socket.socket] = {}
    for i in range(listen_fds):
        fd = SD_LISTEN_FDS_START + i
        sock = socket.socket(fileno=fd)
        sock.setblocking(False)
        name = names[i] if i < len(names) else f"fd{fd}"
        out[name] = sock
    return out


def _setup_logging() -> None:
    level = os.environ.get("NEURAL_TTS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _serve_socket(
    sock: socket.socket,
    handler,
) -> asyncio.base_events.Server:
    return await asyncio.start_unix_server(handler, sock=sock)


async def run(args: argparse.Namespace) -> int:
    _setup_logging()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = DaemonConfig.load()
    log.info(
        "starting; provider=%s idle_timeout=%ds eager=%s",
        config.provider.default,
        config.supervisor.idle_timeout_seconds,
        config.supervisor.eager_startup,
    )

    supervisor = Supervisor(
        default_provider=config.provider.default,
        idle_timeout_seconds=config.supervisor.idle_timeout_seconds,
        enabled_providers=config.provider.enabled,
        eager_startup=config.supervisor.eager_startup,
        voice_allowlists={name: s.voices for name, s in config.providers.items()},
        provider_envs={name: dict(s.env) for name, s in config.providers.items() if s.env},
        voice_locale_overrides={
            name: s.locales for name, s in config.providers.items() if s.locales
        },
    )

    adopted = _adopt_systemd_sockets()
    speechd_sock: socket.socket | None = None
    control_sock: socket.socket | None = None

    if adopted:
        speechd_sock = adopted.get("neural-tts")
        control_sock = adopted.get("neural-tts-control")
        if speechd_sock is None or control_sock is None:
            log.error("expected both 'neural-tts' and 'neural-tts-control' FDs; got %s", list(adopted))
            return 1
    elif args.foreground:
        SPEECHD_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
        for p in (SPEECHD_SOCKET_PATH, CONTROL_SOCKET_PATH):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        speechd_sock = _make_unix_listener(SPEECHD_SOCKET_PATH)
        control_sock = _make_unix_listener(CONTROL_SOCKET_PATH)
        log.info("foreground mode; listening on %s and %s", SPEECHD_SOCKET_PATH, CONTROL_SOCKET_PATH)
    else:
        log.error("no LISTEN_FDS and --foreground not set; refusing to start")
        return 2

    speechd_server = await _serve_socket(
        speechd_sock,
        lambda r, w: handle_speechd_connection(r, w, supervisor),
    )
    control_server = await _serve_socket(
        control_sock,
        lambda r, w: handle_control_connection(r, w, supervisor),
    )

    await supervisor.start_idle_loop()

    if config.supervisor.eager_startup:
        try:
            await supervisor.ensure_ready()
        except Exception:
            log.exception("eager startup failed; will retry on first request")

    sd_notify.ready()
    sd_notify.status(f"provider={config.provider.default}")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(signame: str) -> None:
        log.info("received %s; shutting down", signame)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal, sig.name)

    try:
        await stop_event.wait()
    finally:
        sd_notify.stopping()
        speechd_server.close()
        control_server.close()
        await asyncio.gather(
            speechd_server.wait_closed(),
            control_server.wait_closed(),
            return_exceptions=True,
        )
        await supervisor.shutdown()

    return 0


def _make_unix_listener(path: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.bind(str(path))
    os.chmod(path, 0o600)
    sock.listen(8)
    return sock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neural-tts-daemon")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Bind sockets ourselves (no systemd socket activation)",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
