import threading

import numpy as np
import pytest

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import to_float32
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel
from stenograf.config import Language, MeetingProfile
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.session import (
    ChannelPlan,
    MeetingRecorder,
    SessionStore,
    interleave,
    plan_channels,
)
from stenograf.transcript import TranscriptEntry


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


class FakeDiarizer(Diarizer):
    def __init__(self, turns: list[SpeakerTurn]):
        self.turns = turns
        self.seen_num_speakers: object = "unset"

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns


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

    def test_checkpoints_fire_on_the_interval(self):
        one_second = np.ones(SAMPLE_RATE, dtype=np.int16)
        provider = ListProvider(
            [frame(Channel.MIC, float(t), one_second) for t in range(3)]  # 3 s of audio
        )
        recorder = MeetingRecorder(
            MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR()
        )
        checkpoints: list[int] = []
        transcript = recorder.run(
            provider,
            on_checkpoint=lambda t: checkpoints.append(len(t.entries)),
            checkpoint_interval=1.0,
        )
        # Captured duration reaches 1 s, 2 s, 3 s → a checkpoint at each, non-empty.
        assert len(checkpoints) == 3
        assert all(n > 0 for n in checkpoints)
        assert [e.speaker for e in transcript.entries] == ["Local-1"]

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
