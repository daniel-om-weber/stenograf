"""Muting a channel mid-meeting: silence at the frame intake, spans on the
session clock, and the span's line in the transcript and the notes prompt."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from conftest import FakeASR, ListProvider, WholeBufferVAD, write_wav

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, Channel
from stenograf.capture.file import FileCaptureProvider
from stenograf.config import MeetingProfile
from stenograf.notes.prompt import build_messages
from stenograf.session import MeetingRecorder, MuteSwitch
from stenograf.transcript import MutedSpan, Transcript, TranscriptEntry


def _frame(start: float, seconds: float = 0.1, channel: Channel = Channel.MIC) -> AudioFrame:
    count = round(seconds * SAMPLE_RATE)
    return AudioFrame(channel, start, np.full(count, 1000, dtype=np.int16))


def _clock(value: list[float]):
    return lambda: value[0]


class TestMuteSwitch:
    def test_only_the_samples_inside_the_span_go_silent(self):
        now = [0.05]
        switch = MuteSwitch()
        switch.bind(_clock(now))
        switch.set(Channel.MIC, True)
        now[0] = 0.25
        switch.set(Channel.MIC, False)

        first, second, third = (switch.apply(_frame(t)) for t in (0.0, 0.1, 0.2))
        half = round(0.05 * SAMPLE_RATE)
        assert (first.samples[:half] == 1000).all() and (first.samples[half:] == 0).all()
        assert (second.samples == 0).all()
        assert (third.samples[:half] == 0).all() and (third.samples[half:] == 1000).all()

    def test_the_other_channel_and_the_original_frame_are_untouched(self):
        switch = MuteSwitch()
        switch.bind(lambda: 0.0)
        switch.set(Channel.MIC, True)
        system = _frame(0.0, channel=Channel.SYSTEM)
        assert switch.apply(system) is system
        mic = _frame(0.0)
        muted = switch.apply(mic)
        assert (muted.samples == 0).all()
        assert (mic.samples == 1000).all(), "the tee shares the frame; silence a copy"

    def test_spans_close_at_the_stop_and_empty_ones_vanish(self):
        now = [1.0]
        switch = MuteSwitch()
        switch.bind(_clock(now))
        switch.set(Channel.MIC, True)
        switch.set(Channel.MIC, False)  # pressed twice within one instant
        switch.set(Channel.SYSTEM, True)
        switch.set(Channel.SYSTEM, True)  # repeating a state changes nothing
        assert switch.muted(Channel.SYSTEM) and not switch.muted(Channel.MIC)
        assert switch.spans(until=5.0) == [MutedSpan("system", 1.0, 5.0)]

    def test_without_a_clock_presses_land_where_consumption_is(self):
        switch = MuteSwitch()  # never bound (unpaced replay has no clock)
        switch.apply(_frame(0.0, seconds=1.0))
        switch.set(Channel.MIC, True)
        assert switch.spans(until=2.0) == [MutedSpan("mic", 1.0, 2.0)]


def _recorder() -> MeetingRecorder:
    return MeetingRecorder(
        MeetingProfile(local_speakers=1, remote_speakers=0), asr=FakeASR(), vad=WholeBufferVAD()
    )


class TestMutedMeeting:
    def test_a_call_muted_while_frames_still_queue_is_silenced_anyway(self):
        # The backlog at a meeting's start: frames wait in the provider queue
        # while the models load. Muting 1 s in and unmuting 2 s in BEFORE any
        # frame is consumed must still silence 1–2 s. A switch that applied its
        # current state to the next consumed frame would keep all of it.
        frames = [_frame(i * 0.1) for i in range(30)]
        switch = MuteSwitch()
        now = [1.0]
        switch.bind(_clock(now))
        switch.set(Channel.MIC, True)
        now[0] = 2.0
        switch.set(Channel.MIC, False)

        teed: list[AudioFrame] = []
        result = _recorder().run(ListProvider(frames), on_frame=teed.append, mute=switch)

        audio = np.concatenate([f.samples for f in teed])
        assert (audio[: SAMPLE_RATE] == 1000).all()
        assert (audio[SAMPLE_RATE : 2 * SAMPLE_RATE] == 0).all()
        assert (audio[2 * SAMPLE_RATE :] == 1000).all()
        assert result.transcript.muted == [MutedSpan("mic", 1.0, 2.0)]

    def test_a_meeting_that_ends_muted_closes_the_span_at_the_stop(self):
        frames = [_frame(i * 0.1) for i in range(20)]
        switch = MuteSwitch()
        switch.bind(lambda: 0.5)
        switch.set(Channel.MIC, True)
        result = _recorder().run(ListProvider(frames), mute=switch)
        assert result.transcript.muted == [MutedSpan("mic", 0.5, pytest.approx(2.0))]

    def test_the_live_pass_hears_the_silence_too(self, tmp_path):
        # Live mode stores through the capture thread: same seam, same result.
        wav = tmp_path / "mic.wav"
        write_wav(wav, np.full(SAMPLE_RATE, 1000, dtype=np.int16))
        switch = MuteSwitch()
        switch.bind(lambda: 0.0)
        switch.set(Channel.MIC, True)
        teed: list[AudioFrame] = []
        result = _recorder().run(
            FileCaptureProvider({Channel.MIC: wav}),
            live=True,
            on_frame=teed.append,
            mute=switch,
        )
        assert all((f.samples == 0).all() for f in teed)
        assert result.transcript.muted == [MutedSpan("mic", 0.0, pytest.approx(1.0))]


class TestPacedReplayClock:
    def test_the_clock_starts_when_frames_are_first_read(self, tmp_path):
        wav = tmp_path / "mic.wav"
        write_wav(wav, seconds=0.3)
        assert FileCaptureProvider({Channel.MIC: wav}).now() is None  # unpaced: none
        provider = FileCaptureProvider({Channel.MIC: wav}, paced=True)
        provider.start({Channel.MIC})
        assert provider.now() == 0.0  # models still loading: no replayed time yet
        frames = provider.frames()
        next(frames)
        time.sleep(0.05)
        assert provider.now() >= 0.05


def _transcript(muted: list[MutedSpan]) -> Transcript:
    return Transcript(
        language=None,
        profile=MeetingProfile(),
        entries=[
            TranscriptEntry("Local-1", "Guten Morgen.", 0.0, 2.0),
            TranscriptEntry("Remote-1", "Hallo.", 70.0, 72.0),
        ],
        muted=muted,
    )


class TestTheTranscriptLine:
    def test_markdown_marks_the_span_where_it_happened(self):
        md = _transcript([MutedSpan("mic", 30.0, 62.0)]).to_markdown()
        assert md.split("\n\n") == [
            "**Local-1** [0:00]: Guten Morgen.",
            "*(microphone muted 0:30–1:02)*",
            "**Remote-1** [1:10]: Hallo.\n",
        ]

    def test_a_span_after_the_last_line_comes_last(self):
        md = _transcript([MutedSpan("system", 80.0, 90.0)]).to_markdown()
        assert md.rstrip().endswith("*(system audio muted 1:20–1:30)*")

    def test_json_round_trips_and_older_files_still_load(self):
        transcript = _transcript([MutedSpan("mic", 30.0, 62.0)])
        assert Transcript.from_json(transcript.to_json()) == transcript
        old = json.loads(transcript.to_json())
        del old["muted"]
        assert Transcript.from_json(json.dumps(old)).muted == []

    def test_the_notes_prompt_names_the_gap(self):
        transcript = _transcript([MutedSpan("mic", 30.0, 62.0)])
        whole = build_messages(transcript)[1]["content"]
        assert "*(microphone muted 0:30–1:02)*" in whole
        # A chunk names only the gaps inside its own stretch.
        first = build_messages(transcript, entries=transcript.entries[:1])[1]["content"]
        assert "muted" not in first


class TestEchoCancellerKeepsItsReference:
    def test_muting_system_audio_leaves_the_mic_cancellation_unchanged(self):
        # The span is applied after the canceller, so it still cancels against
        # the real far end; muting before it would let the remote voices leak
        # into the mic channel with nothing to cancel them against.
        from test_aec import _echo_of, _energy_db, _i16, _interleave, _speech

        from stenograf.aec import EchoCancellingProvider

        far = _speech(4.0, seed=3)
        frames = _interleave(_i16(_echo_of(far)), _i16(far))

        def mic_out(mute: MuteSwitch | None) -> np.ndarray:
            teed: list[AudioFrame] = []
            recorder = MeetingRecorder(
                MeetingProfile(local_speakers=1, remote_speakers=1),
                asr=FakeASR(),
                vad=WholeBufferVAD(),
            )
            recorder.run(
                EchoCancellingProvider(ListProvider(frames)), on_frame=teed.append, mute=mute
            )
            return np.concatenate([f.samples for f in teed if f.channel is Channel.MIC])

        switch = MuteSwitch()
        switch.bind(lambda: 0.0)
        switch.set(Channel.SYSTEM, True)
        muted, plain = mic_out(switch), mic_out(None)
        np.testing.assert_array_equal(muted, plain)
        assert _energy_db(plain) < _energy_db(_i16(_echo_of(far))) - 10  # it did cancel
