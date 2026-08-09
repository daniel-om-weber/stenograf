"""Warm the notes backend during the meeting, on the thread that will generate.

The notes model otherwise loads lazily inside the first ``complete()`` call —
strictly after diarization, though the load depends on nothing from the
transcript — so the end-of-meeting wait starts with a multi-GB cold load.
:class:`NotesWarmer` owns one dedicated thread that (after a grace delay, and
only past a free-memory gate) builds the backend and calls its ``warm()``
hook, then waits for the meeting's transcript and runs the real generation
job on that same thread. One thread for warm-up *and* generation is the
mlx-lm 0.29 constraint (:mod:`stenograf.notes.mlx`: generation streams are
per-thread); backends without a ``warm()`` hook still get built early, which
is cheap and harmless. Ollama is deliberately not warmed: its server unloads
idle models (``keep_alive`` defaults to minutes), so a start-of-meeting warm
evaporates before a normal meeting ends.

If the worker never warmed (short meeting, low memory, warm failure) the job
simply builds its own backend — the cold path is the fallback by
construction, and warming can only ever move work earlier, never change it.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from stenograf.notes.backend import NotesBackend

MIN_AVAILABLE_BYTES = 12 * 1024**3
"""Warm only with this much reclaimable memory. The default mac model wires
~4.35 GB for the rest of the meeting, and buying a head start on notes must
not push the live pass or the desktop into swap — a 16 GB box stays on the
cold path, the 48 GB dev box warms."""

WARM_DELAY_S = 45.0
"""Grace before warming: meeting start is when the ASR loads and the live
pass finds its steady state — the 4-GB read and the GPU warm-up token must
not compete with that. A meeting shorter than this just takes the cold
path."""


def _available_bytes() -> int | None:
    """Reclaimable memory, or None where unknown (unknown ⇒ allow: the only
    in-process backend is mlx on macOS, and Linux has /proc/meminfo)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        page = 16384
        pages = 0
        for line in out.splitlines():
            if "page size of" in line:
                digits = [w for w in line.split() if w.isdigit()]
                if digits:
                    page = int(digits[0])
            for key in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
                if line.startswith(key + ":"):
                    pages += int(line.split(":")[1].strip().rstrip("."))
        return pages * page if pages else None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    return None


class NotesWarmer:
    """One thread that warms the notes backend mid-meeting, then generates.

    ``build`` constructs the backend exactly as the notes tail would
    (:func:`stenograf.notes.run.resolve_backend` with the run's own
    arguments) — the warmed backend must be the one generation uses, or the
    warm-up heated the wrong model.
    """

    def __init__(
        self,
        build: Callable[[], NotesBackend],
        *,
        delay_s: float = WARM_DELAY_S,
        min_available_bytes: int = MIN_AVAILABLE_BYTES,
    ) -> None:
        self._build = build
        self._delay_s = delay_s
        self._min_available = min_available_bytes
        self._backend: NotesBackend | None = None
        self._job: Callable[[NotesBackend | None], bool] | None = None
        self._result = False
        self._error: BaseException | None = None
        self._job_ready = threading.Event()
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self.warmed = False
        """Whether the backend was built (and its ``warm()`` hook ran) before
        the job arrived — the observable for tests and status lines."""
        self._thread = threading.Thread(target=self._run, name="notes-warm", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """No job will come (abandon, no transcript, a dying run): release the
        thread — and with it the warmed weights. Idempotent; harmless after a
        completed job."""
        self._cancelled.set()
        self._job_ready.set()

    def run_notes(
        self,
        job: Callable[[NotesBackend | None], bool],
        *,
        on_first_interrupt: Callable[[], None] | None = None,
    ) -> bool | None:
        """Run ``job(warmed_backend_or_None)`` on the warmer thread; block.

        Mirrors the notes tail's Ctrl-C contract from the *waiting* thread
        (signal handlers can't reach the worker): first KeyboardInterrupt
        warns via ``on_first_interrupt``, the second abandons the wait and
        returns None — the daemon thread may still finish writing, which
        only ever means the notes land anyway."""
        if self._cancelled.is_set():
            return job(None)  # thread already released — degrade to inline
        self._job = job
        self._job_ready.set()
        warned = False
        while True:
            try:
                if self._done.wait(0.2):
                    break
            except KeyboardInterrupt:
                if warned or on_first_interrupt is None:
                    return None
                warned = True
                on_first_interrupt()
        if self._error is not None:
            raise self._error
        return self._result

    def _run(self) -> None:
        job_arrived = self._job_ready.wait(self._delay_s)
        if not job_arrived and not self._cancelled.is_set():
            try:
                available = _available_bytes()
                if available is None or available >= self._min_available:
                    backend = self._build()
                    warm = getattr(backend, "warm", None)
                    if warm is not None:
                        warm()
                    self._backend = backend
                    self.warmed = True
            except Exception:
                # A failed warm-up must never cost the notes: drop the half
                # -built backend, generation rebuilds cold.
                self._backend = None
                self.warmed = False
        self._job_ready.wait()
        job = self._job
        if job is None:  # cancelled
            self._done.set()
            return
        try:
            self._result = job(self._backend)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the waiting thread
            self._error = exc
        self._done.set()
