"""Control socket handler — length-prefixed JSON (not protobuf).

The synth socket uses protobuf for the performance-critical PCM path; the
control socket stays JSON so neural-tts-ctl can be a stdlib-only system-python
script with no protobuf dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct

from .supervisor import ProviderNotInstalled, ProviderUnknown, Supervisor

log = logging.getLogger("neural_tts_daemon.control")

MAX_CONTROL_FRAME = 64 * 1024


async def handle_control_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    supervisor: Supervisor,
) -> None:
    try:
        try:
            request = await _read_json(reader)
        except Exception as e:
            log.warning("malformed control request: %s", e)
            return

        op = request.get("op")
        try:
            if op == "status":
                reply = _status(supervisor)
            elif op == "list-voices":
                reply = await _list_voices(supervisor)
            elif op == "reload-voices":
                reply = await _reload_voices(supervisor)
            elif op == "list-providers":
                reply = _list_providers(supervisor)
            else:
                reply = {"ok": False, "error": f"unknown op {op!r}"}
        except ProviderNotInstalled as e:
            reply = {"ok": False, "error": str(e)}
        except ProviderUnknown as e:
            reply = {"ok": False, "error": str(e)}
        except Exception as e:
            log.exception("control op %s failed", op)
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        try:
            _write_json(writer, reply)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _read_json(reader: asyncio.StreamReader) -> dict:
    (length,) = struct.unpack(">I", await reader.readexactly(4))
    if length == 0 or length > MAX_CONTROL_FRAME:
        raise ValueError(f"bad frame length: {length}")
    raw = await reader.readexactly(length)
    return json.loads(raw.decode("utf-8"))


def _write_json(writer: asyncio.StreamWriter, obj: dict) -> None:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack(">I", len(raw)))
    writer.write(raw)


def _status(s: Supervisor) -> dict:
    idx = s.voice_index()
    return {
        "ok": True,
        "active": s.active_name(),
        "state": s.active_state().value,
        "active_voices_count": len(s.active_voices()),
        "sample_rate": s.active_sample_rate(),
        "uptime_seconds": round(s.uptime_seconds(), 2),
        "idle_for_seconds": round(s.idle_for_seconds(), 2),
        "idle_timeout_seconds": s.idle_timeout_seconds,
        "indexed_voices_count": len(idx.all_voices()),
        "indexed_providers": idx.known_providers(),
    }


async def _list_voices(s: Supervisor) -> dict:
    """Return the union of voices across enabled providers.

    Populates the index on first call by enumerating each provider; subsequent
    calls just read the cache. Use `reload-voices` to force a refresh.
    """
    idx = await s.ensure_voice_index_populated()
    voices = idx.all_voices()
    return {
        "ok": True,
        "voices_count": len(voices),
        "voices": [
            {**v.to_json(), "provider": idx.provider_for(v.id)}
            for v in voices
        ],
    }


async def _reload_voices(s: Supervisor) -> dict:
    """Force a fresh enumeration; the on-disk cache is overwritten."""
    await s.enumerate_all_voices()
    return await _list_voices(s)


def _list_providers(s: Supervisor) -> dict:
    active = s.active_name()
    return {
        "ok": True,
        "providers": [
            {
                "name": p.name,
                "display_name": p.display_name,
                "installed": p.installed,
                "active": p.name == active,
                "supports_cloning": p.supports_cloning,
                "needs_models": p.needs_models,
                "notes": p.notes,
            }
            for p in s.known_providers()
        ],
    }
