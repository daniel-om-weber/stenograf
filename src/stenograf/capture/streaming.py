"""Pipe readers shared by every subprocess capture transport.

All that remains of the queue-streaming provider machinery: the
arrival-anchored ``SessionClock`` and the per-channel pump base class lived
here while Linux captured through ``parec``, and left with it (2026-08, step 5
of PLAN-CAPTURE-HELPER.md) — every platform now streams device-stamped frames
from a ``stenocap`` helper (:mod:`stenograf.capture.helper`), so no provider
stamps frames on arrival any more. A transport that would need arrival
stamping again should be handing over real timestamps instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import IO


def read_exact(stream: IO[bytes], n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or ``None`` at end of stream.

    A partial tail is discarded — it means the writer died mid-record, and no
    caller can use half a frame."""
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    return data if len(data) == n else None


def relay_lines(stream: IO[bytes], sink: Callable[[str], None]) -> None:
    """Forward a transport's piped stderr to ``sink``, one decoded line at a time.

    Runs on a daemon thread and owns the stream: it reads until EOF (the
    process exiting closes the write end) and closes the read end on the way
    out. Decoding is permissive and a raising sink is swallowed per line —
    diagnostics must never stall the transport (an undrained pipe would
    eventually block the helper's own writes).
    """
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            with contextlib.suppress(Exception):  # a broken sink must not kill the drain
                sink(line)
    finally:
        stream.close()
