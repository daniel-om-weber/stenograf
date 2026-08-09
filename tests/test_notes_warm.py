"""NotesWarmer: warm-on-the-generation-thread semantics, gates, fallbacks."""

from __future__ import annotations

import threading
import time

from stenograf.notes import warm as warm_module
from stenograf.notes.warm import NotesWarmer


class _FakeBackend:
    def __init__(self) -> None:
        self.warm_thread: int | None = None

    def warm(self) -> None:
        self.warm_thread = threading.get_ident()


def test_warm_and_generate_share_one_thread(monkeypatch):
    # The mlx constraint: whichever thread warmed must also generate.
    monkeypatch.setattr(warm_module, "_available_bytes", lambda: None)  # unknown ⇒ allow
    backend = _FakeBackend()
    warmer = NotesWarmer(lambda: backend, delay_s=0.01)
    warmer.start()
    for _ in range(200):  # wait for the warm-up to happen
        if warmer.warmed:
            break
        time.sleep(0.01)
    assert warmer.warmed and backend.warm_thread is not None

    seen: dict[str, object] = {}

    def job(handed) -> bool:
        seen["backend"] = handed
        seen["thread"] = threading.get_ident()
        return True

    assert warmer.run_notes(job) is True
    assert seen["backend"] is backend
    assert seen["thread"] == backend.warm_thread  # same thread, by construction


def test_job_before_delay_skips_warm():
    calls: list[str] = []

    def build():
        calls.append("built")
        return _FakeBackend()

    warmer = NotesWarmer(build, delay_s=30.0)
    warmer.start()
    assert warmer.run_notes(lambda backend: backend is None) is True
    assert not warmer.warmed and calls == []  # short meeting: cold path, no build


def test_memory_gate_blocks_warm(monkeypatch):
    monkeypatch.setattr(warm_module, "_available_bytes", lambda: 1)  # nothing free
    backend = _FakeBackend()
    warmer = NotesWarmer(lambda: backend, delay_s=0.01)
    warmer.start()
    time.sleep(0.2)
    assert not warmer.warmed
    assert warmer.run_notes(lambda handed: handed is None) is True  # cold fallback


def test_warm_failure_falls_back_cold(monkeypatch):
    monkeypatch.setattr(warm_module, "_available_bytes", lambda: None)

    def build():
        raise RuntimeError("no such model")

    warmer = NotesWarmer(build, delay_s=0.01)
    warmer.start()
    time.sleep(0.2)
    assert not warmer.warmed
    assert warmer.run_notes(lambda handed: handed is None) is True


def test_cancel_releases_thread_and_degrades_inline():
    warmer = NotesWarmer(_FakeBackend, delay_s=30.0)
    warmer.start()
    warmer.cancel()
    warmer._thread.join(timeout=5)
    assert not warmer._thread.is_alive()
    # A late job (should not happen in flow, but must not deadlock) runs inline.
    ident: list[int] = []

    def job(handed) -> bool:
        ident.append(threading.get_ident())
        return handed is None

    assert warmer.run_notes(job) is True
    assert ident == [threading.get_ident()]


def test_job_error_reraises_on_waiting_thread(monkeypatch):
    monkeypatch.setattr(warm_module, "_available_bytes", lambda: None)
    warmer = NotesWarmer(_FakeBackend, delay_s=30.0)
    warmer.start()

    def job(handed) -> bool:
        raise ValueError("boom")

    try:
        warmer.run_notes(job)
    except ValueError as exc:
        assert str(exc) == "boom"
    else:  # pragma: no cover
        raise AssertionError("job error must propagate")
