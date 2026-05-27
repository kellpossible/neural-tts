"""End-to-end supervisor test using a fake provider subprocess.

Spawns a Python subprocess that speaks the framed-protobuf protocol on FD 3,
exactly like a real provider would. Verifies warmup, synth streaming, shutdown
all work over the socketpair + LISTEN_FDS path.
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# A minimal fake-provider script. It speaks the same wire protocol as the real
# provider but returns deterministic data and uses no ML libs.
FAKE_PROVIDER = textwrap.dedent(
    """
    import asyncio, os, socket, struct, sys
    sys.path.insert(0, str(__import__('pathlib').Path(r'{repo}') / 'daemon' / 'src'))
    from neural_tts_daemon.pb import neural_tts_pb2 as pb

    async def serve(sock):
        loop = asyncio.get_running_loop()
        reader, writer = await asyncio.open_unix_connection(sock=sock)
        try:
            while True:
                try:
                    head = await reader.readexactly(4)
                except asyncio.IncompleteReadError:
                    return
                (length,) = struct.unpack('>I', head)
                raw = await reader.readexactly(length)
                req = pb.Request(); req.ParseFromString(raw)
                op = req.WhichOneof('op')
                if op == 'warmup':
                    resp = pb.Response(warmup=pb.WarmupResponse(
                        sample_rate=24000,
                        voices=[pb.Voice(id='fake_one', language='en-US', gender=pb.FEMALE, display_name='fake_one')],
                    ))
                    _send(writer, resp)
                elif op == 'synthesize':
                    _send(writer, pb.Response(synthesize_header=pb.SynthesizeResponseHeader(sample_rate=24000)))
                    import numpy as np
                    for i in range(3):
                        pcm = np.full(4, float(i) / 10.0, dtype=np.float32).tobytes()
                        _send(writer, pb.Response(audio_chunk=pb.AudioChunk(pcm=pcm)))
                    _send(writer, pb.Response(audio_end=pb.AudioEnd()))
                elif op == 'list_voices':
                    _send(writer, pb.Response(list_voices=pb.ListVoicesResponse(
                        voices=[pb.Voice(id='fake_one', language='en-US', gender=pb.FEMALE)],
                    )))
                elif op == 'shutdown':
                    _send(writer, pb.Response(shutdown=pb.ShutdownResponse()))
                    await writer.drain()
                    return
                else:
                    _send(writer, pb.Response(error=pb.Error(message=f'unknown op {{op}}')))
                await writer.drain()
        finally:
            writer.close()

    def _send(writer, msg):
        raw = msg.SerializeToString()
        writer.write(struct.pack('>I', len(raw)))
        writer.write(raw)

    fd = int(os.environ['NEURAL_TTS_PROVIDER_FD'])
    sock = socket.socket(fileno=fd)
    sock.setblocking(False)
    asyncio.run(serve(sock))
    """
).strip()


@pytest.fixture
def fake_provider_dir(tmp_path):
    """Build a fake provider matching the layout supervisor expects."""
    import stat
    proj = tmp_path / "fake"
    venv_bin = proj / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    # A symlink to the daemon's venv python causes Python to resolve through
    # the symlink and lose the venv context. Use a wrapper shell script that
    # forwards to the real interpreter so site-packages (with protobuf) is
    # found.
    daemon_py = REPO / "daemon" / ".venv" / "bin" / "python"
    wrapper = venv_bin / "python"
    wrapper.write_text(f'#!/bin/sh\nexec {daemon_py} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    pkg = proj / "src" / "fake_provider"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_PROVIDER.format(repo=str(REPO)))
    return proj


@pytest.fixture
def registry_override(tmp_path, fake_provider_dir, monkeypatch):
    reg = tmp_path / "registry.toml"
    reg.write_text(
        f'[providers.fake]\n'
        f'display_name = "Fake"\n'
        f'project_dir = "{fake_provider_dir.relative_to(tmp_path) if False else fake_provider_dir}"\n'
        f'python_module = "fake_provider"\n'
        f'supports_cloning = false\n'
        f'needs_models = false\n'
    )
    # Patch config.providers_registry_path to point at our fake registry.
    from neural_tts_daemon import config as cfg
    monkeypatch.setattr(cfg, "providers_registry_path", lambda: reg)
    # Also need to override repo_root so that registry's "project_dir" resolves to the absolute path
    monkeypatch.setattr(cfg, "repo_root", lambda: tmp_path)
    # Re-import the supervisor module-state... not necessary if we monkeypatch
    # before importing.
    return reg


def test_supervisor_spawn_warmup_synth_shutdown(registry_override, fake_provider_dir):
    """Spawn the fake provider, warmup, run one synth, get audio bytes, shutdown."""
    # Make sure the fake provider's source is on sys.path for the subprocess.
    pythonpath = str(fake_provider_dir / "src")
    os.environ["PYTHONPATH"] = pythonpath + ":" + os.environ.get("PYTHONPATH", "")
    try:
        from neural_tts_daemon.supervisor import Supervisor, ProviderState

        async def run():
            sup = Supervisor(default_provider="fake", idle_timeout_seconds=0)
            try:
                proc = await sup.ensure_ready()
                assert proc.state == ProviderState.READY
                assert proc.sample_rate == 24000
                assert [v.id for v in proc.voices] == ["fake_one"]

                chunks = []
                async for chunk in sup.synthesize(
                    voice="fake_one", speed=1.0, lang="en-us", text="hi"
                ):
                    chunks.append(chunk)
                assert len(chunks) == 3
                # Each chunk is 4 float32 samples = 16 bytes
                assert all(len(c) == 16 for c in chunks)
            finally:
                await sup.shutdown()

        asyncio.run(run())
    finally:
        os.environ.pop("PYTHONPATH", None)
