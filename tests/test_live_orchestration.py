"""Phase 2, Task 3: the live orchestration spine (AudioBus + CaptureLoop + LiveWorker).

These exercise the threaded plumbing, not the LiveDecoder's caption quality
(that is Task 1's own suite / eval harness). The load-bearing guarantees:

- the worker reconciles a backlog to the latest watermark (one catch-up decode,
  not one per intermediate step), and
- capture + worker together never drop or double-feed audio — the fed windows
  tile the whole buffer exactly — regardless of thread interleaving.
"""

import threading

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel
from stenograf.config import MeetingProfile
from stenograf.live import StreamingUpdate
from stenograf.session import (
    AudioBus,
    CaptureLoop,
    LiveWorker,
    MeetingRecorder,
    SessionStore,
)


class StubDecoder:
    """Records what it was fed so tests can assert coverage; emits nothing."""

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []
        self.offsets: list[float] = []
        self.flushed = 0

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate:
        self.received.append(np.asarray(samples).copy())
        self.offsets.append(t_offset)
        return StreamingUpdate((), "")

    def flush(self) -> StreamingUpdate:
        self.flushed += 1
        return StreamingUpdate((), "")


class FakeASR(ASRBackend):
    """One word per transcribed window (same stub the session tests use)."""

    name = "fake"

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        return [Segment(text="wort", start=0.1, end=0.5, words=(Word("wort", 0.1, 0.5),))]

    def unload(self) -> None:
        pass


class ListProvider(CaptureProvider):
    """Yields a preset list of frames; stops cleanly when told."""

    def __init__(self, frames: list[AudioFrame]):
        self._frames = frames
        self.started_channels: set[Channel] | None = None
        self.stopped = False

    def start(self, channels: set[Channel]) -> None:
        self.started_channels = channels

    def frames(self):
        for f in self._frames:
            if self.stopped:
                return
            yield f

    def stop(self) -> None:
        self.stopped = True


def _frames(channel: Channel, pcm: np.ndarray, frame_len: int) -> list[AudioFrame]:
    return [
        AudioFrame(channel, start / SAMPLE_RATE, pcm[start : start + frame_len])
        for start in range(0, len(pcm), frame_len)
    ]


def _one_second_frames(seconds: int) -> list[AudioFrame]:
    """`seconds` one-second mic frames, one per whole second (t = 0, 1, 2, …)."""
    pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
    return [AudioFrame(Channel.MIC, float(t), pcm) for t in range(seconds)]


class TestAudioBus:
    def test_wait_returns_the_latest_watermark(self):
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, 1.0)
        bus.advance(Channel.MIC, 2.5)  # newest wins — the worker jumps straight here
        marks, closed = bus.wait({Channel.MIC: 0.0})
        assert marks[Channel.MIC] == 2.5
        assert closed is False

    def test_advance_ignores_a_stale_watermark(self):
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, 5.0)
        bus.advance(Channel.MIC, 3.0)  # backward → ignored
        marks, _ = bus.wait({Channel.MIC: 0.0})
        assert marks[Channel.MIC] == 5.0

    def test_close_wakes_a_blocked_waiter(self):
        bus = AudioBus([Channel.MIC])
        result: dict[str, object] = {}

        def waiter() -> None:
            result["marks"], result["closed"] = bus.wait({Channel.MIC: 0.0})

        t = threading.Thread(target=waiter)
        t.start()
        t.join(timeout=0.1)  # no audio, not closed → genuinely blocked
        assert t.is_alive()
        bus.close()
        t.join(timeout=5)
        assert not t.is_alive()
        assert result["closed"] is True


class TestLiveWorker:
    def test_reconciles_a_backlog_into_a_single_feed(self):
        # All audio is present and the bus is closed before the worker even looks,
        # so it must collapse the whole backlog into one catch-up decode.
        store = SessionStore({Channel.MIC})
        pcm = np.arange(1000, dtype=np.int16)
        for f in _frames(Channel.MIC, pcm, 200):
            store.append(f)
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()

        stub = StubDecoder()
        worker = LiveWorker(
            store, bus, {Channel.MIC: stub}, threading.Lock(), channels=[Channel.MIC]
        )
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert worker.error is None
        assert len(stub.received) == 1  # reconciled, not one feed per frame
        assert np.array_equal(stub.received[0], store.samples(Channel.MIC))
        assert stub.offsets == [0.0]
        assert stub.flushed == 1

    def test_capture_and_worker_cover_all_audio_without_loss(self):
        store = SessionStore({Channel.MIC})
        bus = AudioBus([Channel.MIC])
        pcm = (np.arange(50 * 320) % 1000).astype(np.int16)  # 1 s, 50 frames
        provider = ListProvider(_frames(Channel.MIC, pcm, 320))

        stub = StubDecoder()
        worker = LiveWorker(
            store, bus, {Channel.MIC: stub}, threading.Lock(), channels=[Channel.MIC]
        )
        capture = CaptureLoop(provider, store, bus, channels=[Channel.MIC])
        worker.start()
        capture.start()
        capture.join(timeout=5)
        worker.join(timeout=5)

        assert not capture.is_alive() and not worker.is_alive()
        assert capture.error is None and worker.error is None
        # The fed windows tile the whole buffer exactly: nothing lost, nothing
        # decoded twice — true for any interleaving of the two threads.
        assert np.array_equal(np.concatenate(stub.received), store.samples(Channel.MIC))
        assert stub.flushed == 1

    def test_flushes_committed_text_on_the_interval(self):
        # 3 s of audio, all present before the worker looks → one reconciled feed,
        # so the interval flush (Option B checkpoint) coalesces to a single call.
        store = SessionStore({Channel.MIC})
        pcm = np.ones(3 * SAMPLE_RATE, dtype=np.int16)
        for f in _frames(Channel.MIC, pcm, SAMPLE_RATE):
            store.append(f)
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()

        flushes: list[int] = []
        worker = LiveWorker(
            store,
            bus,
            {Channel.MIC: StubDecoder()},
            threading.Lock(),
            channels=[Channel.MIC],
            on_flush=lambda: flushes.append(1),
            flush_interval=1.0,
        )
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert worker.error is None
        assert flushes  # committed text flushed at least once (reconciled → once here)

    def test_no_flush_without_a_callback(self):
        store = SessionStore({Channel.MIC})
        for f in _frames(Channel.MIC, np.ones(2 * SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE):
            store.append(f)
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()
        worker = LiveWorker(
            store, bus, {Channel.MIC: StubDecoder()}, threading.Lock(), channels=[Channel.MIC]
        )
        worker.start()
        worker.join(timeout=5)
        assert worker.error is None  # flush_interval defaults to 0 → no flushing path


class TestMeetingRecorderLive:
    def _recorder(self) -> MeetingRecorder:
        return MeetingRecorder(MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR())

    def test_live_run_streams_commits_and_still_finalizes(self):
        provider = ListProvider(_one_second_frames(4))
        updates: list[tuple[Channel, StreamingUpdate]] = []
        transcript = self._recorder().run(
            provider, live=True, on_update=lambda ch, u: updates.append((ch, u))
        )
        # The finalize pass still produced the authoritative transcript.
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        assert provider.stopped
        # The live worker committed at least one word (force-committed at flush).
        committed = [w for _, u in updates for w in u.committed]
        assert committed

    def test_live_run_stops_at_max_seconds(self):
        provider = ListProvider(_one_second_frames(10))
        transcript = self._recorder().run(provider, live=True, max_seconds=3.0)
        assert provider.stopped
        assert [e.speaker for e in transcript.entries] == ["Local-1"]

    def test_live_run_checkpoints_are_coarse_and_the_finalize_wins(self):
        provider = ListProvider(_one_second_frames(4))
        checkpoints: list[object] = []
        transcript = self._recorder().run(
            provider,
            live=True,
            on_checkpoint=checkpoints.append,
            checkpoint_interval=1.0,
        )
        # The on-stop finalize is still the authoritative, diarized transcript.
        assert provider.stopped
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        # Option B: any live checkpoint is channel-coarse and never empty (the
        # empty-flush guard means a `.partial` only appears once text exists).
        assert all(c.entries for c in checkpoints)
        assert all(e.speaker == "Local" for c in checkpoints for e in c.entries)
