"""Round-trip framing for protobuf-over-Unix-socket."""

from __future__ import annotations

import asyncio
import io
import struct

import pytest

from neural_tts_daemon.pb import neural_tts_pb2 as pb
from neural_tts_daemon.protocol import (
    ProtocolError,
    audio_chunk_response,
    audio_end_response,
    error_response,
    read_message,
    write_message,
)


def _drain_writer_bytes(writer) -> bytes:
    """Pull anything write_message wrote into our fake transport buffer."""
    return bytes(writer.transport._buf)  # type: ignore[attr-defined]


class _BufferTransport:
    def __init__(self):
        self._buf = bytearray()

    def write(self, data):
        self._buf.extend(data)

    def get_extra_info(self, name, default=None):
        return default


class _FakeWriter:
    def __init__(self):
        self.transport = _BufferTransport()

    def write(self, data):
        self.transport.write(data)

    async def drain(self):
        return None


def _reader_from_bytes(data: bytes) -> asyncio.StreamReader:
    loop = asyncio.new_event_loop()
    reader = asyncio.StreamReader(loop=loop)
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def test_write_then_read_request_roundtrip():
    writer = _FakeWriter()
    req = pb.Request(
        synthesize=pb.SynthesizeRequest(voice="af_heart", speed=1.0, lang="en-us", text="Hello.")
    )
    write_message(writer, req)
    data = _drain_writer_bytes(writer)
    # 4-byte length prefix + body
    (length,) = struct.unpack(">I", data[:4])
    assert length > 0
    assert len(data) == 4 + length

    reader = _reader_from_bytes(data)
    loop = asyncio.new_event_loop()
    back = loop.run_until_complete(read_message(reader, pb.Request))
    loop.close()
    assert back.WhichOneof("op") == "synthesize"
    assert back.synthesize.voice == "af_heart"
    assert back.synthesize.text == "Hello."


def test_audio_chunk_and_end_responses():
    writer = _FakeWriter()
    write_message(writer, audio_chunk_response(b"\x00\x01\x02\x03"))
    write_message(writer, audio_end_response())
    data = _drain_writer_bytes(writer)
    reader = _reader_from_bytes(data)
    loop = asyncio.new_event_loop()
    a = loop.run_until_complete(read_message(reader, pb.Response))
    b = loop.run_until_complete(read_message(reader, pb.Response))
    loop.close()
    assert a.WhichOneof("body") == "audio_chunk"
    assert a.audio_chunk.pcm == b"\x00\x01\x02\x03"
    assert b.WhichOneof("body") == "audio_end"


def test_error_response_carries_message():
    writer = _FakeWriter()
    write_message(writer, error_response("bang"))
    reader = _reader_from_bytes(_drain_writer_bytes(writer))
    loop = asyncio.new_event_loop()
    resp = loop.run_until_complete(read_message(reader, pb.Response))
    loop.close()
    assert resp.WhichOneof("body") == "error"
    assert resp.error.message == "bang"


def test_oversized_header_rejected():
    # craft a frame with a too-large length prefix
    bad = struct.pack(">I", 100 * 1024 * 1024)
    reader = _reader_from_bytes(bad + b"\x00" * 8)
    loop = asyncio.new_event_loop()
    with pytest.raises(ProtocolError):
        loop.run_until_complete(read_message(reader, pb.Request))
    loop.close()


def test_truncated_frame_raises_protocol_error():
    # length prefix says 100 bytes, body has only 5
    bad = struct.pack(">I", 100) + b"hello"
    reader = _reader_from_bytes(bad)
    loop = asyncio.new_event_loop()
    with pytest.raises(ProtocolError):
        loop.run_until_complete(read_message(reader, pb.Request))
    loop.close()
