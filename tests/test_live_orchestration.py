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
from stenograf.vad import SpeechSegment
from stenograf.view import LiveView


class StubDecoder:
    """Records what it was fed so tests can assert coverage; emits nothing.

    ``window_cap`` defaults to infinity so the worker's load-shedding branch never
    fires unless a test opts into it — the coverage tests feed tiny buffers and must
    tile exactly. ``drop_window`` records each load-shed abandon."""

    def __init__(self, window_cap: float = float("inf")) -> None:
        self.received: list[np.ndarray] = []
        self.offsets: list[float] = []
        self.flushed = 0
        self.dropped = 0
        self.window_cap = window_cap

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate:
        self.received.append(np.asarray(samples).copy())
        self.offsets.append(t_offset)
        return StreamingUpdate((), "")

    def flush(self) -> StreamingUpdate:
        self.flushed += 1
        return StreamingUpdate((), "")

    def drop_window(self) -> None:
        self.dropped += 1


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

    def test_load_sheds_an_over_long_backlog(self):
        # Inference fell far behind: 40 s of audio are already present and the bus
        # is closed before the worker looks. Feeding all 40 s into one decode would
        # spiral, so the worker abandons the window and feeds only the last
        # window_cap (15) seconds — the 0–25 s span is a caption gap the finalize
        # pass fills on stop.
        store = SessionStore({Channel.MIC})
        for f in _frames(Channel.MIC, np.ones(40 * SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE):
            store.append(f)
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()

        stub = StubDecoder(window_cap=15.0)
        worker = LiveWorker(
            store, bus, {Channel.MIC: stub}, threading.Lock(), channels=[Channel.MIC]
        )
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive() and worker.error is None
        assert stub.dropped == 1  # the window was abandoned, not grown
        assert stub.offsets == [25.0]  # restarted at mark - window_cap
        assert len(stub.received[0]) == 15 * SAMPLE_RATE  # only a full window fed
        assert worker.shed_seconds == 25.0  # 0–25 s skipped

    def test_normal_backlog_is_not_shed(self):
        # A backlog within window_cap feeds whole — no gap, no drop.
        store = SessionStore({Channel.MIC})
        for f in _frames(Channel.MIC, np.ones(5 * SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE):
            store.append(f)
        bus = AudioBus([Channel.MIC])
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()

        stub = StubDecoder(window_cap=15.0)
        worker = LiveWorker(
            store, bus, {Channel.MIC: stub}, threading.Lock(), channels=[Channel.MIC]
        )
        worker.start()
        worker.join(timeout=5)

        assert stub.dropped == 0 and worker.shed_seconds == 0.0
        assert stub.offsets == [0.0]

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

    def test_live_capture_error_is_non_fatal_to_finalize(self):
        # A stream desync (a backward frame) makes the capture thread's store.append
        # raise mid-capture. Every frame that did arrive is already in the store, so
        # the meeting must still finalize it — surfacing the error, not discarding
        # the transcript (the audit's "finalize is authoritative" resilience).
        good = AudioFrame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16))
        backward = AudioFrame(Channel.MIC, 0.0, np.ones(10, dtype=np.int16))  # goes backwards
        provider = ListProvider([good, backward])
        errors: list[str] = []
        transcript = self._recorder().run(provider, live=True, on_status=errors.append)
        assert [e.speaker for e in transcript.entries] == ["Local-1"]  # the good second survived
        assert provider.stopped
        assert any("capture stopped early" in m for m in errors)


class CountingASR(ASRBackend):
    """Counts decodes (the CPU proxy) and returns one stable word per elapsed second.

    The word list grows with the window length and each word keeps a fixed
    relative position, so LocalAgreement confirms a stable, distinct prefix that
    commits in order — a realistic committed stream to check for rewrites.
    """

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, samples, language):
        self.calls += 1
        n = max(1, int(len(samples) / SAMPLE_RATE))
        words = tuple(Word(f"w{i}", i + 0.1, i + 0.6) for i in range(n))
        return [Segment(" ".join(x.text for x in words), words[0].start, words[-1].end, words)]


class _SilentVAD:
    """Never reports speech — the live pass's gate must then run zero ASR."""

    def speech_segments(self, samples):
        return []


class _AlwaysSpeechVAD:
    """Reports the whole window as speech, so the gate never suppresses a decode."""

    def speech_segments(self, samples):
        return [SpeechSegment(0.0, len(samples) / SAMPLE_RATE)]


class _SpyView(LiveView):
    """Records the live-pass decode count at the finalize hand-off and every commit."""

    def __init__(self, asr: CountingASR) -> None:
        self._asr = asr
        self.decodes_at_finalizing: int | None = None
        self.committed: list[Word] = []

    def update(self, channel, update):
        self.committed.extend(update.committed)

    def finalizing(self):
        self.decodes_at_finalizing = self._asr.calls


class TestLivePassCpuProxy:
    """PLAN.md Task 7: the CPU-proxy regression — a decode counter and monotonicity,
    asserted through the wired ``MeetingRecorder.run(live=True)`` path (not just the
    decoder in isolation), so a future orchestration change can't quietly re-decode
    in silence or rewrite committed captions."""

    def _recorder(self, asr, vad) -> MeetingRecorder:
        return MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=asr, vad=vad
        )

    def test_zero_window_decodes_during_silence(self):
        # The VAD reports no speech, so the whole live pass must run no ASR — the
        # ~0% accelerator-in-silence budget (PLAN.md §5). We snapshot the decode
        # count at the finalize hand-off, before the on-stop finalize decodes.
        asr = CountingASR()
        spy = _SpyView(asr)
        provider = ListProvider(_one_second_frames(5))
        self._recorder(asr, _SilentVAD()).run(provider, live=True, view=spy)
        assert spy.decodes_at_finalizing == 0  # no window decode while silent
        assert spy.committed == []  # and nothing was committed

    def test_committed_captions_are_never_rewritten(self):
        # With speech present the live pass decodes and commits; every committed
        # word must be append-only — monotonic start times, each word emitted once,
        # never contradicted by a later decode.
        asr = CountingASR()
        spy = _SpyView(asr)
        provider = ListProvider(_one_second_frames(4))
        self._recorder(asr, _AlwaysSpeechVAD()).run(provider, live=True, view=spy)
        assert spy.committed, "speech should have produced committed captions"
        starts = [w.start for w in spy.committed]
        assert starts == sorted(starts)  # a committed word never moves back in time
        texts = [w.text for w in spy.committed]
        assert len(texts) == len(set(texts))  # each word committed exactly once
