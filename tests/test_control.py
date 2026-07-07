"""Stage B3: the reverse-control channel (`MeetingSession` + `FinalizeRequest`).

Label-free — synthetic in-RAM store, fake ASR/diarizer/resolver, no backends.
"""

import numpy as np
import pytest

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, Channel
from stenograf.config import Language, MeetingProfile, Provenance
from stenograf.control import FinalizeRequest, MeetingSession
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.session import MeetingRecorder, SessionStore

GERMAN = "und das ist wirklich eine gute idee für uns"


class ProbeASR(ASRBackend):
    """Returns fixed segments; records every load and the language of each call."""

    name = "probe"

    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.loads = 0
        self.languages: list[Language | None] = []

    def load(self) -> None:
        self.loads += 1

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.languages.append(language)
        return list(self._segments)

    def unload(self) -> None:
        pass


class FakeDiarizer(Diarizer):
    """Returns fixed turns; records the requested count and any embedding call."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self.turns = turns
        self.seen_num_speakers: object = "unset"

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns

    def diarize_with_embeddings(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return DiarizationResult(
            turns=self.turns, embeddings={t.speaker: np.zeros(4, np.float32) for t in self.turns}
        )


class FakeResolver:
    """A SpeakerResolver that matches nothing — proves whether re-ID ran at all."""

    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, embeddings: dict[str, np.ndarray]) -> dict[str, str]:
        self.calls += 1
        return {}


def _two_speaker_segments() -> list[Segment]:
    # Two well-separated words so a two-turn diarization yields two entries.
    return [
        Segment(
            text="alpha beta",
            start=0.2,
            end=1.4,
            words=(Word("alpha", 0.2, 0.4), Word("beta", 1.2, 1.4)),
        )
    ]


def _store(*channels: Channel) -> SessionStore:
    store = SessionStore(set(channels))
    pcm = np.ones(2 * SAMPLE_RATE, dtype=np.int16)
    for ch in channels:
        store.append(AudioFrame(channel=ch, timestamp=0.0, samples=pcm))
    return store


def _two_channel_session(
    profile: MeetingProfile, **kw
) -> tuple[MeetingSession, ProbeASR, FakeDiarizer]:
    """A finalized session: mic (single) + diarized system, over a synthetic store."""
    asr = ProbeASR(_two_speaker_segments())
    diarizer = FakeDiarizer([SpeakerTurn("S0", 0.0, 1.0), SpeakerTurn("S1", 1.0, 2.0)])
    recorder = MeetingRecorder(profile, asr=asr, diarizer=diarizer, **kw)
    store = _store(Channel.MIC, Channel.SYSTEM)
    transcript = recorder.finalize(store)
    session = MeetingSession(recorder, store, transcript=transcript)
    return session, asr, diarizer


class TestRefinalize:
    def test_empty_request_reruns_without_reloading_backends(self):
        session, asr, _ = _two_channel_session(MeetingProfile(local_speakers=1, remote_speakers=2))
        before = session.transcript
        assert asr.loads == 0  # finalize never (re)loads; the factory loads once

        after = session.refinalize(FinalizeRequest())

        # Same warm backend object reused; still no load triggered by refinalize.
        assert session.recorder.asr is asr
        assert asr.loads == 0
        # An unchanged request reproduces the same transcript.
        assert [(e.speaker, e.text) for e in after.entries] == [
            (e.speaker, e.text) for e in before.entries
        ]
        assert session.transcript is after

    def test_changing_remote_count_changes_plan_and_provenance(self):
        # Remote count unspecified → the finalize estimates it (DETECTED).
        session, _, diarizer = _two_channel_session(MeetingProfile(local_speakers=1))
        assert session.transcript.parameters.speakers["system"].provenance == Provenance.DETECTED
        assert diarizer.seen_num_speakers is None  # estimated

        session.refinalize(FinalizeRequest(remote_speakers=3))

        # The plan now requests an explicit count, tagged EXPLICIT on the transcript.
        assert diarizer.seen_num_speakers == 3
        system = session.transcript.parameters.speakers["system"]
        assert system.value == 3
        assert system.provenance == Provenance.EXPLICIT
        # The override is sticky on the recorder's profile.
        assert session.recorder.profile.remote_speakers == 3

    def test_language_override_beats_a_locked_detection(self):
        # Single in-room channel whose German text auto-detects to German.
        asr = ProbeASR([Segment(GERMAN, 0.1, 1.0, words=(Word(GERMAN, 0.1, 1.0),))])
        recorder = MeetingRecorder(MeetingProfile(local_speakers=1, remote_speakers=0), asr=asr)
        store = _store(Channel.MIC)
        session = MeetingSession(recorder, store, transcript=recorder.finalize(store))
        assert session.transcript.language is Language.GERMAN
        assert session.transcript.parameters.language.provenance == Provenance.DETECTED

        session.refinalize(FinalizeRequest(language=Language.ENGLISH))

        assert session.transcript.language is Language.ENGLISH
        lang = session.transcript.parameters.language
        assert lang.value is Language.ENGLISH
        assert lang.provenance == Provenance.EXPLICIT
        # The correction reached the ASR, overriding the earlier German lock.
        assert asr.languages[-1] is Language.ENGLISH

    def test_keep_language_preserves_a_detected_lock(self):
        asr = ProbeASR([Segment(GERMAN, 0.1, 1.0, words=(Word(GERMAN, 0.1, 1.0),))])
        recorder = MeetingRecorder(MeetingProfile(local_speakers=1, remote_speakers=0), asr=asr)
        store = _store(Channel.MIC)
        session = MeetingSession(recorder, store, transcript=recorder.finalize(store))

        session.refinalize(FinalizeRequest(remote_speakers=None))  # nothing about language

        assert session.transcript.language is Language.GERMAN
        assert session.transcript.parameters.language.provenance == Provenance.DETECTED
        assert asr.languages[-1] is Language.GERMAN  # re-used the lock, not re-detected

    def test_reid_toggle_is_sticky(self):
        resolver = FakeResolver()
        session, _, _ = _two_channel_session(
            MeetingProfile(local_speakers=1, remote_speakers=2), reid=resolver
        )
        assert resolver.calls == 1  # the initial finalize ran re-ID

        session.refinalize(FinalizeRequest(reid=False))
        assert session.recorder.reid is None
        assert resolver.calls == 1  # disabled — did not run

        session.refinalize(FinalizeRequest())  # keep: stays off
        assert resolver.calls == 1

        session.refinalize(FinalizeRequest(reid=True))
        assert session.recorder.reid is resolver
        assert resolver.calls == 2  # switched back on

    def test_reid_enable_without_a_resolver_is_a_noop(self):
        session, _, _ = _two_channel_session(MeetingProfile(local_speakers=1, remote_speakers=2))
        session.refinalize(FinalizeRequest(reid=True))  # no resolver available
        assert session.recorder.reid is None  # stays off, no crash

    def test_invalid_override_raises(self):
        session, _, _ = _two_channel_session(
            MeetingProfile(local_speakers=1, remote_speakers=0)  # in-room
        )
        with pytest.raises(ValueError, match="at least one speaker"):
            session.refinalize(FinalizeRequest(local_speakers=0))  # would zero out the meeting


class TestRenameSpeaker:
    def test_renames_only_the_target_label(self):
        session, _, _ = _two_channel_session(MeetingProfile(local_speakers=1, remote_speakers=2))
        before = {e.speaker for e in session.transcript.entries}
        assert before == {"Local-1", "Remote-1", "Remote-2"}
        originals = {e.speaker: e for e in session.transcript.entries}

        after = session.rename_speaker("Remote-2", "Bob")

        assert {e.speaker for e in after.entries} == {"Local-1", "Remote-1", "Bob"}
        # Every non-target entry is untouched down to text/timestamps/words.
        for e in after.entries:
            if e.speaker != "Bob":
                assert e == originals[e.speaker]
        bob = next(e for e in after.entries if e.speaker == "Bob")
        old = originals["Remote-2"]
        assert (bob.text, bob.start, bob.end, bob.words) == (
            old.text,
            old.start,
            old.end,
            old.words,
        )
        assert session.transcript is after

    def test_rename_of_absent_label_is_a_noop(self):
        session, _, _ = _two_channel_session(MeetingProfile(local_speakers=1, remote_speakers=2))
        before = list(session.transcript.entries)
        after = session.rename_speaker("Nobody", "X")
        assert list(after.entries) == before

    def test_rename_before_finalize_raises(self):
        session = MeetingSession(_bare_recorder(), _store(Channel.MIC))  # transcript=None
        with pytest.raises(ValueError, match="finalize the meeting first"):
            session.rename_speaker("Local-1", "Alice")


def _bare_recorder() -> MeetingRecorder:
    return MeetingRecorder(MeetingProfile(local_speakers=1, remote_speakers=0), asr=ProbeASR([]))


class TestStop:
    def test_stop_delegates_to_the_hook(self):
        calls = []
        stop = lambda: calls.append(1)  # noqa: E731 — a one-line stub is clearest here
        MeetingSession(_bare_recorder(), _store(Channel.MIC), stop=stop).stop()
        assert calls == [1]

    def test_stop_without_a_hook_is_a_noop(self):
        MeetingSession(_bare_recorder(), _store(Channel.MIC)).stop()  # must not raise
