"""Minimal sd_notify(3) implementation — no systemd-python dependency."""

from __future__ import annotations

import os
import socket


def notify(state: str) -> bool:
    """Send a notification to systemd. Returns True if delivered, False if not running under systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(state.encode("utf-8"))
        return True
    except OSError:
        return False


def ready() -> None:
    notify("READY=1")


def stopping() -> None:
    notify("STOPPING=1")


def status(msg: str) -> None:
    notify(f"STATUS={msg}")
