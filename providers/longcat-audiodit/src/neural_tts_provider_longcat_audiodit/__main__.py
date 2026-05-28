"""Provider entry point.

Adopts the inherited socketpair on FD given by NEURAL_TTS_PROVIDER_FD,
runs a framed protobuf request/response loop until shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import struct
import sys

import numpy as np
from google.protobuf.message import DecodeError

from .pb import neural_tts_pb2 as pb
from .provider import LongCatProvider

MAX_FRAME_BYTES = 16 * 1024 * 1024

log = logging.getLogger("neural_tts_provider_longcat_audiodit")


class _ProtocolError(Exception):
    pass


async def _read_request(reader: asyncio.StreamReader) -> pb.Request:
    try:
        (length,) = struct.unpack(">I", await reader.readexactly(4))
    except asyncio.IncompleteReadError as e:
        raise _ProtocolError("eof before frame") from e
    if length == 0 or length > MAX_FRAME_BYTES:
        raise _ProtocolError(f"bad frame length: {length}")
    raw = await reader.readexactly(length)
    msg = pb.Request()
    try:
        msg.ParseFromString(raw)
    except DecodeError as e:
        raise _ProtocolError(f"protobuf decode: {e}") from e
    return msg


def _write(writer: asyncio.StreamWriter, msg: pb.Response) -> None:
    raw = msg.SerializeToString()
    if len(raw) > MAX_FRAME_BYTES:
        raise _ProtocolError(f"frame too large: {len(raw)}")
    writer.write(struct.pack(">I", len(raw)))
    writer.write(raw)


def _adopt_socket() -> socket.socket:
    fd_str = os.environ.get("NEURAL_TTS_PROVIDER_FD")
    if not fd_str:
        raise SystemExit(
            "provider must be launched by neural-tts-daemon (NEURAL_TTS_PROVIDER_FD expected)"
        )
    try:
        fd = int(fd_str)
    except ValueError:
        raise SystemExit(f"NEURAL_TTS_PROVIDER_FD must be an integer, got {fd_str!r}")
    sock = socket.socket(fileno=fd)
    if sock.family != socket.AF_UNIX or sock.type != socket.SOCK_STREAM:
        raise SystemExit(f"inherited FD {fd} is not a SOCK_STREAM AF_UNIX socket")
    sock.setblocking(False)
    return sock


def _setup_logging() -> None:
    level = os.environ.get("NEURAL_TTS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, eager: bool) -> None:
    provider = LongCatProvider(eager_startup=eager)
    try:
        while True:
            try:
                req = await _read_request(reader)
            except _ProtocolError as e:
                log.info("connection ended: %s", e)
                return

            op = req.WhichOneof("op")
            try:
                if op == "warmup":
                    resp = await _handle_warmup(provider)
                elif op == "synthesize":
                    await _handle_synthesize(provider, req.synthesize, writer)
                    continue
                elif op == "list_voices":
                    resp = pb.Response(
                        list_voices=pb.ListVoicesResponse(voices=provider.list_voices_pb())
                    )
                elif op == "shutdown":
                    _write(writer, pb.Response(shutdown=pb.ShutdownResponse()))
                    await writer.drain()
                    return
                else:
                    resp = pb.Response(error=pb.Error(message=f"unsupported op {op!r}"))
            except Exception as e:
                log.exception("op %s failed", op)
                resp = pb.Response(error=pb.Error(message=f"{type(e).__name__}: {e}"))

            _write(writer, resp)
            await writer.drain()
    finally:
        await provider.shutdown()
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_warmup(provider: LongCatProvider) -> pb.Response:
    sample_rate, voices = await provider.warmup()
    return pb.Response(warmup=pb.WarmupResponse(sample_rate=sample_rate, voices=voices))


async def _handle_synthesize(
    provider: LongCatProvider,
    req: pb.SynthesizeRequest,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        _write(
            writer,
            pb.Response(
                synthesize_header=pb.SynthesizeResponseHeader(sample_rate=provider.sample_rate)
            ),
        )
        await writer.drain()

        async for samples in provider.synthesize_stream(
            voice=req.voice,
            speed=req.speed if req.speed else 1.0,
            lang=req.lang or "en",
            text=req.text,
        ):
            arr = np.asarray(samples, dtype=np.float32, copy=False).reshape(-1)
            _write(writer, pb.Response(audio_chunk=pb.AudioChunk(pcm=arr.tobytes())))
            try:
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        _write(writer, pb.Response(audio_end=pb.AudioEnd()))
        await writer.drain()
    except Exception as e:
        log.exception("synthesize failed")
        try:
            _write(writer, pb.Response(error=pb.Error(message=f"{type(e).__name__}: {e}")))
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


async def run(*, eager: bool) -> int:
    _setup_logging()
    sock = _adopt_socket()
    reader, writer = await asyncio.open_unix_connection(sock=sock)
    try:
        await _serve(reader, writer, eager=eager)
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="neural-tts-provider-longcat-audiodit")
    parser.add_argument(
        "--eager-startup",
        action="store_true",
        help="Load the torch model + warm CUDA graphs at process start "
             "(default: defer until the first synthesize request)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(eager=args.eager_startup))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
