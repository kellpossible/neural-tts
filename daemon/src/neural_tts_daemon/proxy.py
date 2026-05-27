"""Speechd-side request handler: read pb.Request, drive the supervisor, write pb.Response stream."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .config import WIRE_SAMPLE_RATE
from .pb import neural_tts_pb2 as pb
from .protocol import (
    ProtocolError,
    audio_chunk_response,
    audio_end_response,
    error_response,
    read_message,
    write_message,
)
from .resample import Resampler, float32_to_s16le_bytes
from .supervisor import ProviderNotInstalled, Supervisor

log = logging.getLogger("neural_tts_daemon.proxy")


async def handle_speechd_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    supervisor: Supervisor,
) -> None:
    try:
        try:
            request = await read_message(reader, pb.Request)
        except ProtocolError as e:
            log.warning("malformed request: %s", e)
            return

        op = request.WhichOneof("op")
        if op != "synthesize":
            await _send_and_close(writer, error_response(f"unsupported op {op!r}"))
            return

        synth = request.synthesize
        if not synth.voice:
            await _send_and_close(writer, error_response("missing 'voice'"))
            return

        # Auto-route via the global voice index: which provider owns this voice?
        # If the index isn't populated yet, this triggers a one-time enumeration.
        idx = await supervisor.ensure_voice_index_populated()
        target_provider = idx.provider_for(synth.voice)
        if target_provider is None:
            await _send_and_close(writer, error_response(
                f"unknown voice {synth.voice!r} (not found in any enabled provider; "
                f"try `bin/neural-tts-ctl reload-voices`)"
            ))
            return

        try:
            proc = await supervisor.ensure_ready(target_provider)
        except ProviderNotInstalled as e:
            await _send_and_close(writer, error_response(str(e)))
            return
        except Exception as e:
            log.exception("provider start failed")
            await _send_and_close(writer, error_response(f"provider unavailable: {e}"))
            return

        if synth.voice not in {v.id for v in proc.voices}:
            # The index said this voice was here but the actual provider disagrees —
            # most likely a reference-clip was removed since the last enumeration.
            log.warning(
                "voice %r in index for %s but provider doesn't have it; "
                "consider running reload-voices", synth.voice, target_provider,
            )
            await _send_and_close(writer, error_response(
                f"voice {synth.voice!r} no longer available in {target_provider!r}"
            ))
            return

        # Header: commit to streaming reply at the wire sample rate.
        write_message(
            writer,
            pb.Response(synthesize_header=pb.SynthesizeResponseHeader(sample_rate=WIRE_SAMPLE_RATE)),
        )

        resampler = Resampler(in_rate=proc.sample_rate, out_rate=WIRE_SAMPLE_RATE)
        chunks = supervisor.synthesize(
            voice=synth.voice,
            speed=synth.speed if synth.speed else 1.0,
            lang=synth.lang or "en-us",
            text=synth.text,
        )

        log.info(
            "synth start: voice=%s speed=%s lang=%s text_len=%d",
            synth.voice, synth.speed, synth.lang, len(synth.text),
        )
        try:
            await _pump(writer, resampler, chunks)
        except Exception:
            log.exception("synthesis pump failed")
        finally:
            # Critical: the supervisor.synthesize generator holds the per-provider
            # request_lock while suspended at `yield`. Without an explicit aclose,
            # the lock can stay held until garbage collection, deadlocking the
            # next request that touches the provider socket.
            try:
                await chunks.aclose()
            except Exception:
                log.exception("error closing synth generator")
        log.info("synth complete")

        try:
            write_message(writer, audio_end_response())
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
    except Exception:
        log.exception("unhandled error in speechd connection")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _pump(
    writer: asyncio.StreamWriter,
    resampler: Resampler,
    chunks,
) -> None:
    n_chunks = 0
    n_bytes = 0
    try:
        async for chunk in chunks:
            n_chunks += 1
            samples = np.frombuffer(chunk, dtype=np.float32)
            resampled = resampler.process(samples)
            if resampled.size:
                pcm = float32_to_s16le_bytes(resampled)
                n_bytes += len(pcm)
                write_message(writer, audio_chunk_response(pcm))
                try:
                    await writer.drain()
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    log.warning("pump: client closed after %d chunks (%s)", n_chunks, e)
                    return
        tail = resampler.flush()
        if tail.size:
            pcm = float32_to_s16le_bytes(tail)
            n_bytes += len(pcm)
            write_message(writer, audio_chunk_response(pcm))
            try:
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                log.warning("pump: client closed during tail (%s)", e)
                return
        log.info("pump: wrote %d chunks, %d bytes", n_chunks, n_bytes)
    except asyncio.CancelledError:
        log.warning("pump cancelled after %d chunks", n_chunks)
        raise
    except Exception:
        log.exception("pump failed after %d chunks", n_chunks)


async def _send_and_close(writer: asyncio.StreamWriter, resp: pb.Response) -> None:
    try:
        write_message(writer, resp)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
