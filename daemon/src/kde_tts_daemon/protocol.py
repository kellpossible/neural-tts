"""Length-prefixed framing for protobuf messages.

Wire format on every socket in the system:
    [4 bytes big-endian uint32 N]
    [N bytes: a serialized protobuf Message]

The proto schema is at proto/kde_tts.proto. Generated code lives in
this package's `pb/` subpackage.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Type, TypeVar

from google.protobuf.message import DecodeError, Message

from .pb import kde_tts_pb2 as pb

MAX_FRAME_BYTES = 16 * 1024 * 1024

M = TypeVar("M", bound=Message)


class ProtocolError(Exception):
    """Raised when a framed message is malformed or oversized."""


async def read_message(reader: asyncio.StreamReader, message_cls: Type[M]) -> M:
    try:
        (length,) = struct.unpack(">I", await reader.readexactly(4))
    except asyncio.IncompleteReadError as e:
        raise ProtocolError("connection closed before frame") from e
    if length == 0 or length > MAX_FRAME_BYTES:
        raise ProtocolError(f"bad frame length: {length}")
    try:
        raw = await reader.readexactly(length)
    except asyncio.IncompleteReadError as e:
        raise ProtocolError("connection closed mid-frame") from e
    msg = message_cls()
    try:
        msg.ParseFromString(raw)
    except DecodeError as e:
        raise ProtocolError(f"protobuf decode failed: {e}") from e
    return msg


def write_message(writer: asyncio.StreamWriter, msg: Message) -> None:
    raw = msg.SerializeToString()
    if len(raw) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame too large to send: {len(raw)} > {MAX_FRAME_BYTES}")
    writer.write(struct.pack(">I", len(raw)))
    writer.write(raw)


# Convenience builders so callers don't reference pb directly all over the place.

def error_response(message: str) -> pb.Response:
    return pb.Response(error=pb.Error(message=message))


def audio_chunk_response(payload: bytes) -> pb.Response:
    return pb.Response(audio_chunk=pb.AudioChunk(pcm=payload))


def audio_end_response() -> pb.Response:
    return pb.Response(audio_end=pb.AudioEnd())
