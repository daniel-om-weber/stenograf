import signal
import threading

import numpy as np
import pytest

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import to_float32
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel
from stenograf.config import Language, MeetingProfile
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.profiles import ProfileStore, SpeakerProfile, SpeakerReID
from stenograf.session import (
    AudioBus,
    ChannelPlan,
    MeetingRecorder,
    SessionStore,
    _shield_interrupt,
    _TailCheckpointer,
    interleave,
    plan_channels,
)
from stenograf.transcript import Transcript, TranscriptEntry


def frame(channel: Channel, timestamp: float, samples: np.ndarray) -> AudioFrame:
    return AudioFrame(channel=channel, timestamp=timestamp, samples=samples)


class TestSessionStore:
    def test_appends_contiguous_frames(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
        store.append(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([3, 4], dtype=np.int16)))
        expected = np.array([1, 2, 3, 4], dtype=np.int16).astype(np.float32) / 32768.0
        assert np.allclose(store.samples(Channel.MIC), expected)
        assert store.duration(Channel.MIC) == 4 / SAMPLE_RATE

    def test_pads_silence_across_a_gap(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 0.0, np.array([5], dtype=np.int16)))
        # Next frame starts one second later — 16000 samples of silence between.
        store.append(frame(Channel.MIC, 1.0, np.array([6], dtype=np.int16)))
        samples = store.samples(Channel.MIC)
        assert len(samples) == SAMPLE_RATE + 1
        assert samples[0] != 0.0
        assert np.all(samples[1:SAMPLE_RATE] == 0.0)
        assert samples[SAMPLE_RATE] != 0.0

    def test_ignores_unrecorded_channel(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.SYSTEM, 0.0, np.array([9], dtype=np.int16)))
        assert store.channels() == [Channel.MIC]
        assert len(store.samples(Channel.MIC)) == 0

    def test_empty_channel_yields_no_samples(self):
        store = SessionStore({Channel.SYSTEM})
        assert len(store.samples(Channel.SYSTEM)) == 0
        assert store.duration(Channel.SYSTEM) == 0.0

    def test_backward_frame_raises_instead_of_misaligning(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 1.0, np.ones(SAMPLE_RATE, dtype=np.int16)))
        with pytest.raises(ValueError, match="backwards"):
            store.append(frame(Channel.MIC, 0.0, np.ones(10, dtype=np.int16)))

    def test_minor_overlap_within_tolerance_is_clamped(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16)))
        # 5 ms behind the tail (< 10 ms tolerance): appended contiguously, no raise.
        store.append(frame(Channel.MIC, 1.0 - 0.005, np.ones(100, dtype=np.int16)))
        assert store.duration(Channel.MIC) == (SAMPLE_RATE + 100) / SAMPLE_RATE

    def test_view_extracts_an_interior_window(self):
        store = SessionStore({Channel.MIC})
        data = np.arange(1, SAMPLE_RATE + 1, dtype=np.int16)  # 1 s, value == index+1
        store.append(frame(Channel.MIC, 0.0, data))
        win = store.view(Channel.MIC, 0.25, 0.75)  # samples [4000, 12000)
        assert np.array_equal(win, to_float32(data[4000:12000]))

    def test_view_defaults_end_to_the_current_buffer(self):
        store = SessionStore({Channel.MIC})
        data = np.arange(1, SAMPLE_RATE + 1, dtype=np.int16)
        store.append(frame(Channel.MIC, 0.0, data))
        assert np.array_equal(store.view(Channel.MIC, 0.5), to_float32(data[8000:]))

    def test_view_matches_full_buffer_slice_across_a_gap(self):
        # A silence gap makes the buffer span three chunks (samples, pad, samples);
        # view must agree with slicing the whole float32 buffer at any window.
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 0.0, np.array([7, 8], dtype=np.int16)))
        store.append(frame(Channel.MIC, 1.0, np.array([9, 10, 11], dtype=np.int16)))
        full = store.samples(Channel.MIC)
        assert len(full) == SAMPLE_RATE + 3
        windows = [(0, None), (1, 5), (0, 2), (SAMPLE_RATE, None), (3, SAMPLE_RATE + 2)]
        for start, end in windows:
            end_s = None if end is None else end / SAMPLE_RATE
            win = store.view(Channel.MIC, start / SAMPLE_RATE, end_s)
            assert np.array_equal(win, full[start : len(full) if end is None else end])

    def test_view_clamps_a_window_past_the_end(self):
        store = SessionStore({Channel.MIC})
        store.append(frame(Channel.MIC, 0.0, np.array([1, 2, 3], dtype=np.int16)))
        # end far past the tail → clamped to the full buffer, not an error.
        assert np.array_equal(store.view(Channel.MIC, 0.0, 100.0), store.samples(Channel.MIC))

    def test_view_returns_empty_for_empty_or_out_of_range_windows(self):
        store = SessionStore({Channel.MIC})
        assert len(store.view(Channel.MIC, 0.0)) == 0  # nothing captured yet
        store.append(frame(Channel.MIC, 0.0, np.array([1, 2, 3], dtype=np.int16)))
        assert len(store.view(Channel.MIC, 10.0, 20.0)) == 0  # start past the tail
        assert len(store.view(Channel.MIC, 0.5, 0.1)) == 0  # inverted range
        assert len(store.view(Channel.MIC, 0.0, 0.0)) == 0  # zero-width

    def test_view_and_samples_stay_consistent_under_concurrent_append(self):
        # Single writer + reader. Each sample's value is a function of its global
        # index, so any prefix a reader observes must match exactly — a torn read
        # (chunks/length disagreeing) would corrupt it. Exercises the lock.
        store = SessionStore({Channel.MIC})
        n_frames, frame_len = 400, 500
        all_samples = ((np.arange(n_frames * frame_len) % 20000) - 10000).astype(np.int16)
        expected = to_float32(all_samples)

        errors: list[Exception] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    v = store.view(Channel.MIC, 0.0)  # prefix-immortal → must match
                    assert np.array_equal(v, expected[: len(v)])
                    s = store.samples(Channel.MIC)
                    assert np.array_equal(s, expected[: len(s)])
            except Exception as exc:  # noqa: BLE001 — surface it on the main thread
                errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        for k in range(n_frames):
            chunk = all_samples[k * frame_len : (k + 1) * frame_len]
            store.append(frame(Channel.MIC, k * frame_len / SAMPLE_RATE, chunk))
        stop.set()
        t.join(timeout=5)

        assert not t.is_alive()
        assert not errors, errors[0]
        assert np.array_equal(store.samples(Channel.MIC), expected)


class TestPlanChannels:
    def test_online_records_mic_single_and_system_diarized(self):
        plans = plan_channels(MeetingProfile(local_speakers=1, remote_speakers=2))
        assert plans == [
            ChannelPlan(Channel.MIC, 1, "Local-{n}"),
            ChannelPlan(Channel.SYSTEM, 2, "Remote-{n}"),
        ]

    def test_in_room_records_only_the_mic(self):
        plans = plan_channels(MeetingProfile(local_speakers=3, remote_speakers=0))
        assert [p.channel for p in plans] == [Channel.MIC]
        assert plans[0].num_speakers == 3

    def test_unknown_local_defaults_to_single_speaker(self):
        plans = plan_channels(MeetingProfile(remote_speakers=2))
        assert plans[0] == ChannelPlan(Channel.MIC, 1, "Local-{n}")

    def test_unknown_remote_records_system_and_estimates(self):
        plans = plan_channels(MeetingProfile(local_speakers=1))
        system = next(p for p in plans if p.channel is Channel.SYSTEM)
        assert system.num_speakers is None  # estimate

    def test_no_local_speaker_records_only_the_system(self):
        # Listen-only: --local 0 --remote 2. No mic; never a num_speakers=0 plan.
        plans = plan_channels(MeetingProfile(local_speakers=0, remote_speakers=2))
        assert [p.channel for p in plans] == [Channel.SYSTEM]
        assert plans[0].num_speakers == 2


def test_interleave_orders_channels_by_start():
    entries = [
        TranscriptEntry(speaker="Remote-1", text="b", start=1.0, end=1.5),
        TranscriptEntry(speaker="Local-1", text="a", start=0.0, end=0.5),
        TranscriptEntry(speaker="Remote-1", text="c", start=2.0, end=2.5),
    ]
    assert [e.text for e in interleave(entries)] == ["a", "b", "c"]


class FakeASR(ASRBackend):
    """One word per transcribed window, at a fixed offset within it."""

    name = "fake"

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        return [Segment(text="wort", start=0.1, end=0.5, words=(Word("wort", 0.1, 0.5),))]

    def unload(self) -> None:
        pass


class GermanASR(ASRBackend):
    """Transcribes to German text, for language-detection tests."""

    name = "german"

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        text = "und das ist wirklich eine gute idee für uns"
        return [Segment(text=text, start=0.1, end=1.0, words=(Word(text, 0.1, 1.0),))]

    def unload(self) -> None:
        pass


class RecordingASR(ASRBackend):
    """Records the length of every buffer it transcribes (proves tail exactly-once)."""

    name = "recording"

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.lengths.append(len(samples))
        return [Segment(text="w", start=0.0, end=0.1, words=(Word("w", 0.0, 0.1),))]

    def unload(self) -> None:
        pass


class CommittedWords:
    """A stand-in decoder exposing only the committed words a checkpoint reads."""

    def __init__(self, words: list[Word]) -> None:
        self._words = tuple(words)

    @property
    def committed_words(self) -> tuple[Word, ...]:
        return self._words


class FakeDiarizer(Diarizer):
    def __init__(self, turns: list[SpeakerTurn]):
        self.turns = turns
        self.seen_num_speakers: object = "unset"

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns


class EmbeddingDiarizer(Diarizer):
    """FakeDiarizer that also carries a per-cluster embedding (the re-ID surface)."""

    def __init__(self, turns: list[SpeakerTurn], embeddings: dict[str, np.ndarray]):
        self.turns = turns
        self.embeddings = embeddings

    def diarize(self, samples, num_speakers=None):
        return self.turns

    def diarize_with_embeddings(self, samples, num_speakers=None):
        return DiarizationResult(turns=self.turns, embeddings=self.embeddings)


class RaisingDiarizer(Diarizer):
    """Throws on every diarize call — stands in for a mid-meeting backend fault."""

    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, samples, num_speakers=None):
        self.calls += 1
        raise RuntimeError("diarizer exploded")


class ListProvider(CaptureProvider):
    """Yields a preset list of frames — an in-process stand-in for a device."""

    def __init__(self, frames: list[AudioFrame]):
        self._frames = frames
        self.started_channels: set[Channel] | None = None
        self.stopped = False

    def start(self, channels: set[Channel]) -> None:
        self.started_channels = channels

    def frames(self):
        yield from self._frames

    def stop(self) -> None:
        self.stopped = True


class TestMeetingRecorder:
    def test_runs_both_channels_and_labels_local_and_remote(self):
        pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider([frame(Channel.MIC, 0.0, pcm), frame(Channel.SYSTEM, 0.0, pcm)])
        diarizer = FakeDiarizer([SpeakerTurn("S0", 0.0, 2.0)])
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=2),
            asr=FakeASR(),
            diarizer=diarizer,
        )
        transcript = recorder.run(provider)

        assert provider.started_channels == {Channel.MIC, Channel.SYSTEM}
        assert provider.stopped
        speakers = {e.speaker for e in transcript.entries}
        assert speakers == {"Local-1", "Remote-1"}
        # System channel was diarized with the known remote count; mic was not.
        assert diarizer.seen_num_speakers == 2

    def test_reid_names_matched_cluster_and_survives_relabel(self):
        # A cluster matching a stored profile is named by re-ID and must NOT be
        # renumbered into the channel template (Remote-1) by finalize's relabel;
        # an unmatched cluster still gets the template label.
        model = "eres2net-voxceleb-16k.onnx"
        store = ProfileStore(
            profiles=[SpeakerProfile("Daniel", model, np.array([1.0, 0.0], np.float32))]
        )
        reid = SpeakerReID(store, model)
        pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider([frame(Channel.SYSTEM, 0.0, pcm)])
        diarizer = EmbeddingDiarizer(
            [SpeakerTurn("S0", 0.0, 2.0)], {"S0": np.array([1.0, 0.0], np.float32)}
        )
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=0, remote_speakers=2),
            asr=FakeASR(),
            diarizer=diarizer,
            reid=reid,
        )
        transcript = recorder.run(provider)
        assert {e.speaker for e in transcript.entries} == {"Daniel"}

    def test_no_reid_configured_keeps_channel_labels(self):
        # Default (no re-ID) path is unchanged: raw clusters template to Remote-N.
        pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider([frame(Channel.SYSTEM, 0.0, pcm)])
        diarizer = EmbeddingDiarizer(
            [SpeakerTurn("S0", 0.0, 2.0)], {"S0": np.array([1.0, 0.0], np.float32)}
        )
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=0, remote_speakers=2),
            asr=FakeASR(),
            diarizer=diarizer,
        )
        transcript = recorder.run(provider)
        assert {e.speaker for e in transcript.entries} == {"Remote-1"}

    def test_channel_diarizer_failure_keeps_both_channels(self):
        # One channel's diarizer throwing must not lose the other channel's
        # transcript. The failing channel falls back to un-diarized text (kept),
        # and the meeting still finalizes rather than crashing.
        pcm = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider([frame(Channel.MIC, 0.0, pcm), frame(Channel.SYSTEM, 0.0, pcm)])
        diarizer = RaisingDiarizer()
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=2),
            asr=FakeASR(),
            diarizer=diarizer,
        )
        transcript = recorder.run(provider)  # must not raise

        assert diarizer.calls == 1  # the system channel actually attempted to diarize
        # Mic (Local-1, no diarization) and system (Remote-1, un-diarized fallback)
        # both survive; neither channel's text is dropped.
        assert {e.speaker for e in transcript.entries} == {"Local-1", "Remote-1"}
        assert {e.text for e in transcript.entries} == {"wort"}

    def test_mic_single_speaker_skips_the_diarizer(self):
        provider = ListProvider([frame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16))])
        diarizer = FakeDiarizer([SpeakerTurn("S0", 0.0, 2.0)])
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0),
            asr=FakeASR(),
            diarizer=diarizer,
        )
        transcript = recorder.run(provider)
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        assert diarizer.seen_num_speakers == "unset"  # never called

    def test_interrupt_still_finalizes(self):
        def interrupting():
            yield frame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16))
            raise KeyboardInterrupt

        provider = ListProvider([])
        provider.frames = interrupting  # type: ignore[method-assign]
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        transcript = recorder.run(provider)
        assert provider.stopped
        assert [e.speaker for e in transcript.entries] == ["Local-1"]

    def test_batch_checkpoints_accumulate_with_coarse_labels(self):
        one_second = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider(
            [frame(Channel.MIC, float(t), one_second) for t in range(3)]  # 3 s of audio
        )
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        checkpoints: list[Transcript] = []
        transcript = recorder.run(
            provider, on_checkpoint=checkpoints.append, checkpoint_interval=1.0
        )
        # The tail checkpoint runs off-thread and coalesces, so the *count* is
        # timing-dependent — but at least one always fires for 3 s at interval 1.
        assert checkpoints
        # Checkpoints are channel-coarse (un-diarized); only the final transcript
        # carries the diarized ``Local-1`` label.
        assert all(e.speaker == "Local" for c in checkpoints for e in c.entries)
        assert [e.speaker for e in transcript.entries] == ["Local-1"]
        # Each checkpoint holds the full transcript-so-far: entries only ever grow.
        counts = [len(c.entries) for c in checkpoints]
        assert counts == sorted(counts)

    def test_tail_entries_shift_onto_the_session_clock_with_a_coarse_label(self):
        store = SessionStore({Channel.MIC})
        for t in range(3):
            store.append(frame(Channel.MIC, float(t), np.ones(SAMPLE_RATE, dtype=np.int16)))
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        plan = plan_channels(recorder.profile)[0]
        entries = recorder._tail_entries(store, plan, 1.0, 2.0)
        assert entries
        assert all(e.speaker == "Local" for e in entries)  # coarse, not diarized Local-1
        assert all(e.start >= 1.0 for e in entries)  # word times shifted into the tail

    def test_live_checkpoint_groups_committed_words_by_channel(self):
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=1), asr=FakeASR()
        )
        decoders = {
            Channel.MIC: CommittedWords([Word("hallo", 0.1, 0.5), Word("welt", 0.6, 0.9)]),
            Channel.SYSTEM: CommittedWords([Word("guten", 0.2, 0.6)]),
        }
        transcript = recorder._live_checkpoint(decoders)
        by_speaker = {e.speaker: e.text for e in transcript.entries}
        assert by_speaker == {"Local": "hallo welt", "Remote": "guten"}
        # Interleaved by start time across channels (mic 0.1 before system 0.2).
        assert [e.speaker for e in transcript.entries] == ["Local", "Remote"]

    def test_live_checkpoint_is_empty_before_anything_commits(self):
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        transcript = recorder._live_checkpoint({Channel.MIC: CommittedWords([])})
        assert transcript.entries == []

    def test_checkpointing_disabled_by_default_interval_zero(self):
        provider = ListProvider(
            [frame(Channel.MIC, float(t), np.ones(SAMPLE_RATE, dtype=np.int16)) for t in range(3)]
        )
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        checkpoints = []
        recorder.run(provider, on_checkpoint=lambda t: checkpoints.append(t), checkpoint_interval=0)
        assert checkpoints == []

    def test_language_is_auto_detected_from_the_transcript(self):
        provider = ListProvider([frame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16))])
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=GermanASR()
        )
        transcript = recorder.run(provider)
        assert transcript.language == Language.GERMAN

    def test_explicit_language_is_never_overridden_by_detection(self):
        provider = ListProvider([frame(Channel.MIC, 0.0, np.ones(SAMPLE_RATE, dtype=np.int16))])
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0, language=Language.ENGLISH),
            asr=GermanASR(),  # German text, but the user forced English → English wins
        )
        transcript = recorder.run(provider)
        assert transcript.language == Language.ENGLISH


class TestTailCheckpointer:
    """The batch (--no-live) crash checkpoint: tail-only finalize, off capture."""

    def test_finalizes_each_second_of_audio_exactly_once(self):
        store = SessionStore({Channel.MIC})
        for t in range(3):  # 3 s of audio
            store.append(frame(Channel.MIC, float(t), np.ones(SAMPLE_RATE, dtype=np.int16)))
        bus = AudioBus([Channel.MIC])
        asr = RecordingASR()
        recorder = MeetingRecorder(MeetingProfile(local_speakers=1, remote_speakers=0), asr=asr)
        plans = plan_channels(recorder.profile)
        writes: list[Transcript] = []
        checkpointer = _TailCheckpointer(recorder, store, plans, bus, writes.append, 1.0)
        checkpointer.start()
        bus.advance(Channel.MIC, store.duration(Channel.MIC))
        bus.close()
        checkpointer.join(timeout=5)

        assert not checkpointer.is_alive()
        assert checkpointer.error is None
        # The old whole-buffer re-finalize would total 1+2+3 = 6 s of ASR; the
        # tail-only path finalizes each second exactly once → 3 s, whatever the
        # thread interleaving (one catch-up tail or three).
        assert sum(asr.lengths) == 3 * SAMPLE_RATE
        assert writes  # at least one checkpoint written
        assert all(e.speaker == "Local" for t in writes for e in t.entries)


class TestShieldInterrupt:
    """The on-stop finalize must not be lost to a second Ctrl-C (0b)."""

    def test_ignores_sigint_within_the_block_then_restores(self):
        original = signal.getsignal(signal.SIGINT)
        try:
            with _shield_interrupt():
                assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
                signal.raise_signal(signal.SIGINT)  # swallowed, must not raise
            # A finalize wrapped in the shield runs to completion, then the
            # previous handler is restored so normal Ctrl-C works again.
            assert signal.getsignal(signal.SIGINT) is original
        finally:
            signal.signal(signal.SIGINT, original)

    def test_is_a_noop_off_the_main_thread(self):
        # Only the main thread can set signal handlers; off it the shield must be
        # a silent no-op (the TUI runs the meeting on a background thread).
        errors: list[BaseException] = []
        ran = threading.Event()

        def worker() -> None:
            try:
                with _shield_interrupt():
                    ran.set()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        assert ran.is_set()
        assert errors == []
