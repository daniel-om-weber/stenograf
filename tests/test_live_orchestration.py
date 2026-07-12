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
from conftest import CallbackView, FakeASR

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import to_float32
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel
from stenograf.config import MeetingProfile
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.live import StreamingUpdate
from stenograf.session import (
    AudioBus,
    CaptureLoop,
    CheckpointConfig,
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

    def test_flushes_once_more_after_the_final_decode(self):
        # 1 s of audio but a huge interval: no periodic flush ever fires, yet the
        # worker still checkpoints once after its closing decoder flush — so every
        # committed word is on disk before the (crash-prone) finalize starts.
        store = SessionStore({Channel.MIC})
        pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
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
            flush_interval=1000.0,
        )
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert worker.error is None
        assert len(flushes) == 1  # the close-time flush, not an interval one

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
        # Recorded per channel too: this channel's live words now have a gap, so
        # the finalize pass must not reuse them.
        assert worker.shed_by_channel == {Channel.MIC: 25.0}

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
            provider, live=True, view=CallbackView(on_update=lambda ch, u: updates.append((ch, u)))
        ).transcript
        # The finalize pass still produced the authoritative transcript.
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        assert provider.stopped
        # The live worker committed at least one word (force-committed at flush).
        committed = [w for _, u in updates for w in u.committed]
        assert committed

    def test_live_run_stops_at_max_seconds(self):
        provider = ListProvider(_one_second_frames(10))
        transcript = self._recorder().run(provider, live=True, max_seconds=3.0).transcript
        assert provider.stopped
        assert [e.speaker for e in transcript.entries] == ["Local-1"]

    def test_live_run_checkpoints_are_coarse_and_the_finalize_wins(self):
        provider = ListProvider(_one_second_frames(4))
        checkpoints: list[object] = []
        transcript = self._recorder().run(
            provider, live=True, checkpoint=CheckpointConfig(checkpoints.append, 1.0)
        ).transcript
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
        transcript = self._recorder().run(
            provider, live=True, view=CallbackView(on_status=errors.append)
        ).transcript
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


class _OneRunStream:
    """A streaming VAD stream that reports one speech run once 2 s were pushed."""

    def __init__(self):
        self.fed = 0
        self.emitted = False

    def push(self, samples):
        self.fed += len(samples)

    def take_completed(self):
        if not self.emitted and self.fed >= 2 * SAMPLE_RATE:
            self.emitted = True
            return [SpeechSegment(0.2, 1.2)]
        return []

    def open_segment(self):
        return None


class _StreamingSpeechVAD:
    """Stream-capable VAD (selects the window pass) with a batch scan to match."""

    def stream(self, origin: float) -> _OneRunStream:
        return _OneRunStream()

    def speech_segments(self, samples):
        return [SpeechSegment(0.2, 1.2)]  # what a classic finalize re-scan sees


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


class TestFinalizeReuse:
    """The on-stop finalize reuses the window pass's decodes (PLAN: halve total ASR)."""

    def _recorder(self, asr) -> MeetingRecorder:
        return MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0),
            asr=asr,
            vad=_StreamingSpeechVAD(),
        )

    def test_finalize_runs_zero_asr_when_reusing(self):
        asr = CountingASR()
        spy = _SpyView(asr)
        provider = ListProvider(_one_second_frames(4))
        transcript = self._recorder(asr).run(provider, live=True, view=spy).transcript
        # Every decode happened in the live pass; the finalize added none.
        assert spy.decodes_at_finalizing is not None
        assert asr.calls == spy.decodes_at_finalizing
        assert asr.calls >= 1  # ...and the live pass did decode the speech run
        # The reused words still came out as a normal labeled transcript.
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        assert transcript.entries[0].text

    def test_full_finalize_opt_out_re_decodes(self):
        asr = CountingASR()
        spy = _SpyView(asr)
        recorder = self._recorder(asr)
        recorder.reuse_live_finalize = False  # the --full-finalize escape hatch
        result = recorder.run(ListProvider(_one_second_frames(4)), live=True, view=spy)
        transcript = result.transcript
        assert asr.calls > spy.decodes_at_finalizing  # finalize decoded again
        assert [e.speaker for e in transcript.entries] == ["Local-1"]


# -- Two-channel live fixtures -------------------------------------------------
#
# The hybrid scenario (local and remote speakers, overlapping or in turn) needs
# fakes whose output depends on the *audio*, so that a cross-channel mix-up or a
# live/batch window drift changes the transcript and fails the test. FakeASR and
# CountingASR cannot see either.

_AMPLITUDE_STEM = {1000: "foxtrot", 3000: "quebec"}
"""Peak sample value → word stem. The stems share no letters, so one channel's
text can never read as (an echo of) the other's unless the audio really matches."""


class AmplitudeASR(ASRBackend):
    """Deterministic, content-dependent decode: the window's peak amplitude picks
    the word stem and the window length fixes the word count (two per second).
    Byte-identical windows therefore decode to identical words — the property the
    live-reuse parity tests pin down."""

    name = "amplitude"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, samples, language):
        self.calls += 1
        pcm = to_float32(np.asarray(samples))
        peak = round(float(np.abs(pcm).max()) * 32768)
        if peak == 0:
            return []
        stem = _AMPLITUDE_STEM[peak]
        count = max(1, round(2 * len(pcm) / SAMPLE_RATE))
        words = tuple(Word(f"{stem}{i}", 0.5 * i + 0.1, 0.5 * i + 0.4) for i in range(count))
        return [Segment(" ".join(w.text for w in words), words[0].start, words[-1].end, words)]


def _nonzero_runs(samples, origin: float) -> list[SpeechSegment]:
    idx = np.flatnonzero(np.asarray(samples) != 0)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = [idx[0], *idx[breaks + 1]]
    ends = [*idx[breaks], idx[-1]]
    return [
        SpeechSegment(origin + s / SAMPLE_RATE, origin + (e + 1) / SAMPLE_RATE)
        for s, e in zip(starts, ends, strict=True)
    ]


class _EnergyStream:
    """Streaming half of :class:`EnergyVAD`: a run completes once ``CLOSE`` s of
    silence follow it (or at ``finish()``); the trailing run still touching the
    live edge is the open segment."""

    CLOSE = 0.5

    def __init__(self, origin: float) -> None:
        self._origin = origin
        self._buf = np.zeros(0, dtype=np.float32)
        self._emitted_to = origin
        self._finished = False

    def push(self, samples) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])

    def _edge(self) -> float:
        return self._origin + len(self._buf) / SAMPLE_RATE

    def take_completed(self) -> list[SpeechSegment]:
        done = []
        for run in _nonzero_runs(self._buf, self._origin):
            if run.end <= self._emitted_to:
                continue
            if self._finished or run.end <= self._edge() - self.CLOSE:
                done.append(run)
                self._emitted_to = run.end
        return done

    def open_segment(self) -> SpeechSegment | None:
        runs = _nonzero_runs(self._buf, self._origin)
        if self._finished or not runs:
            return None
        last = runs[-1]
        if last.end <= self._emitted_to or last.end <= self._edge() - self.CLOSE:
            return None
        return last

    def finish(self) -> None:
        self._finished = True


class EnergyVAD:
    """Content-defined fake VAD: speech = maximal runs of nonzero samples.

    The batch scan and the stream report identical runs over the same timeline,
    regardless of how the worker chunks its feeds — the agreement the
    finalize-reuse guarantee rests on."""

    def speech_segments(self, samples) -> list[SpeechSegment]:
        return _nonzero_runs(samples, 0.0)

    def stream(self, origin: float) -> _EnergyStream:
        return _EnergyStream(origin)


class RecordingDiarizer(Diarizer):
    """Returns preset turns and records what audio/count it was asked about."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns
        self.samples_len: int | None = None
        self.num_speakers: int | None = None

    def diarize(self, samples, num_speakers=None):
        self.samples_len = len(samples)
        self.num_speakers = num_speakers
        return list(self._turns)


def _pcm_with_speech(seconds: int, spans: list[tuple[float, float]], amplitude: int) -> np.ndarray:
    pcm = np.zeros(seconds * SAMPLE_RATE, dtype=np.int16)
    for start, end in spans:
        pcm[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = amplitude
    return pcm


def _interleaved_frames(mic_pcm: np.ndarray, system_pcm: np.ndarray) -> list[AudioFrame]:
    """One-second frames of both channels, arriving turn about — the two streams
    land in the store concurrently, as a real capture delivers them."""
    frames = []
    for t in range(len(mic_pcm) // SAMPLE_RATE):
        lo, hi = t * SAMPLE_RATE, (t + 1) * SAMPLE_RATE
        frames.append(AudioFrame(Channel.MIC, float(t), mic_pcm[lo:hi]))
        frames.append(AudioFrame(Channel.SYSTEM, float(t), system_pcm[lo:hi]))
    return frames


class TestTwoChannelLive:
    """The hybrid scenario: local and remote speakers, overlapping and in turn.

    Channels are independent end-to-end — each is decoded and diarized on its
    own, and only the finished entries interleave — so the live window pass must
    yield the same transcript as the offline passes, per channel, with no
    cross-channel bleed."""

    SECONDS = 7
    MIC_SPANS = [(1.0, 2.0), (3.0, 4.0)]
    SYSTEM_SPANS = [(1.5, 2.5), (5.0, 5.5)]  # overlaps the mic, then speaks alone

    def _provider(self) -> ListProvider:
        mic = _pcm_with_speech(self.SECONDS, self.MIC_SPANS, 1000)
        system = _pcm_with_speech(self.SECONDS, self.SYSTEM_SPANS, 3000)
        return ListProvider(_interleaved_frames(mic, system))

    def _recorder(self, asr, **kwargs) -> MeetingRecorder:
        return MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=1),
            asr=asr,
            vad=EnergyVAD(),
            **kwargs,
        )

    def test_live_reuse_matches_batch_and_full_finalize(self):
        # The headline parity: online with reuse, online with --full-finalize,
        # and pure offline batch must produce the same transcript for the same
        # two-channel audio, overlapping speech included.
        reuse_asr = AmplitudeASR()
        reused = self._recorder(reuse_asr).run(self._provider(), live=True).transcript

        full_asr = AmplitudeASR()
        recorder = self._recorder(full_asr)
        recorder.reuse_live_finalize = False
        full = recorder.run(self._provider(), live=True).transcript

        batch_asr = AmplitudeASR()
        batch = self._recorder(batch_asr).run(self._provider()).transcript

        assert reused.entries == full.entries == batch.entries
        # Reuse decoded each channel's one window exactly once, in the live pass;
        # the finalize added no ASR. --full-finalize re-decoded both channels.
        assert reuse_asr.calls == 2
        assert batch_asr.calls == 2
        assert full_asr.calls == 4

        local, remote = reused.entries
        assert [local.speaker, remote.speaker] == ["Local-1", "Remote-1"]
        # The simultaneous stretch survived on both channels, in time order...
        assert local.start < remote.start < local.end
        # ...and neither channel's text bled into the other.
        assert "foxtrot" in local.text and "quebec" not in local.text
        assert "quebec" in remote.text and "foxtrot" not in remote.text

    def test_live_commits_are_channel_tagged_and_checkpoints_cover_both(self):
        # Long trailing silence (> the packer's max_gap) closes both windows
        # mid-meeting, so commits and a checkpoint exist before the finalize.
        mic = _pcm_with_speech(9, [(1.0, 2.0)], 1000)
        system = _pcm_with_speech(9, [(1.5, 2.5)], 3000)
        provider = ListProvider(_interleaved_frames(mic, system))
        updates: list[tuple[Channel, StreamingUpdate]] = []
        checkpoints: list[object] = []
        transcript = self._recorder(AmplitudeASR()).run(
            provider,
            live=True,
            view=CallbackView(on_update=lambda ch, u: updates.append((ch, u))),
            checkpoint=CheckpointConfig(checkpoints.append, 1.0),
        ).transcript
        mic_words = [w.text for ch, u in updates if ch is Channel.MIC for w in u.committed]
        sys_words = [w.text for ch, u in updates if ch is Channel.SYSTEM for w in u.committed]
        # Both channels streamed commits, each carrying its own channel's audio.
        assert mic_words and all(w.startswith("foxtrot") for w in mic_words)
        assert sys_words and all(w.startswith("quebec") for w in sys_words)
        # The live checkpoint is channel-coarse and covers both channels.
        assert {e.speaker for e in checkpoints[-1].entries} == {"Local", "Remote"}
        assert [e.speaker for e in transcript.entries] == ["Local-1", "Remote-1"]

    def test_reused_live_words_still_diarize_the_full_channel(self):
        # Two remote speakers back to back: the finalize reuses the live ASR
        # words but must still run diarization over the channel's ENTIRE audio —
        # never just the live windows — and split the reused words by the turns.
        mic = _pcm_with_speech(6, [(0.5, 1.5)], 1000)
        system = _pcm_with_speech(6, [(1.0, 2.0), (2.2, 3.2)], 3000)
        provider = ListProvider(_interleaved_frames(mic, system))
        asr = AmplitudeASR()
        diarizer = RecordingDiarizer([SpeakerTurn("S0", 0.0, 2.0), SpeakerTurn("S1", 2.0, 4.0)])
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=2),
            asr=asr,
            vad=EnergyVAD(),
            diarizer=diarizer,
        )
        transcript = recorder.run(provider, live=True).transcript
        assert asr.calls == 2  # one live decode per channel window; finalize added none
        assert diarizer.samples_len == 6 * SAMPLE_RATE  # the whole channel, not a window
        assert diarizer.num_speakers == 2
        assert [e.speaker for e in transcript.entries] == ["Local-1", "Remote-1", "Remote-2"]

    def test_echo_backstop_still_applies_to_reused_live_words(self):
        # A textual echo (identical words at the same moment on both channels)
        # must still be dropped when the finalize reuses live decodes; --no-aec
        # (dedup off) must still keep both lines.
        def run(dedup: bool):
            pcm = _pcm_with_speech(5, [(1.0, 3.0)], 1000)
            provider = ListProvider(_interleaved_frames(pcm, pcm.copy()))
            recorder = self._recorder(AmplitudeASR(), dedup_echo=dedup)
            return recorder.run(provider, live=True)

        armed = run(dedup=True)
        assert [e.speaker for e in armed.transcript.entries] == ["Remote-1"]
        assert armed.dropped_echo_lines == 1
        off = run(dedup=False)
        assert sorted(e.speaker for e in off.transcript.entries) == ["Local-1", "Remote-1"]
