"""The two-voice check: the voice split, the rerun's share and sides, and the
split itself, with the voice model and the rerun stood in for.

Whether the checks are right about real voices is the eval harness's question
(``eval/README.md``); here they are pinned to what the measured rule says.
The audio stands in for both models: a sample's value names its talker, the
fake embedding and the fake rerun read it back."""

from __future__ import annotations

import numpy as np
import pytest

from stenograf.diarization import split
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.split import (
    HOP,
    RERUN_GAP,
    joined,
    second_voice,
    split_two_voice_speakers,
    to_span,
    voice_crop,
    voice_crops,
    voice_split,
)

SR = 16000


def _turns(label: str, n: int, every: float = 10.0, length: float = 4.0, at: float = 0.0):
    return [SpeakerTurn(label, at + i * every, at + i * every + length) for i in range(n)]


def test_spans_sit_on_the_frame_grid_and_crops_take_the_middle():
    assert to_span(1.234, 2.0) == (19680, 32000)
    assert voice_crop((0, 8 * SR)) == (1 * SR, 7 * SR)
    assert voice_crop((0, 5 * SR)) == (0, 5 * SR)


def test_voice_crops_pick_long_turns_evenly():
    turns = _turns("S0", 100, every=10.0, length=8.0) + [SpeakerTurn("S0", 2000.0, 2001.5)]
    crops = voice_crops(turns)
    assert len(crops) == 60
    assert all(hi - lo == 6 * SR for lo, hi in crops)
    assert crops[0] == (1 * SR, 7 * SR)
    assert crops[-1] == (991 * SR, 997 * SR)  # the last long turn, evenly reached
    assert voice_crops(_turns("S0", 3, length=1.9)) == []


def _vectors(a: int, b: int, angle: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base, other = np.zeros(16), np.zeros(16)
    base[0] = 1.0
    other[0], other[1] = np.cos(angle), np.sin(angle)
    x = np.array([base] * a + [other] * b) + rng.normal(0, 0.02, (a + b, 16))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_voice_split_finds_two_groups_and_their_sizes():
    cosine, minor = voice_split(_vectors(30, 10, np.pi / 2))
    assert cosine < 0.1
    assert minor == pytest.approx(0.25)


def test_voice_split_of_one_voice_has_close_centroids():
    assert voice_split(_vectors(40, 0, 0.0))[0] > 0.9


def test_voice_split_of_identical_vectors_does_not_cut():
    assert voice_split(np.tile(np.eye(4)[0], (12, 1))) == (1.0, 0.0)


def test_joined_puts_silence_after_every_span():
    audio = np.arange(10 * SR, dtype=np.float32)
    out = joined(audio, [(0, 160), (1600, 1920)])
    assert len(out) == 160 + 320 + 2 * RERUN_GAP
    assert np.array_equal(out[:160], audio[:160])
    assert not out[160 : 160 + RERUN_GAP].any()
    assert np.array_equal(out[160 + RERUN_GAP : 480 + RERUN_GAP], audio[1600:1920])


def _rerun(audio: np.ndarray, voices: int = 2) -> np.ndarray:
    """The fake rerun: a frame is on the slot its samples' value names."""
    frames = len(audio) // HOP + 1
    probs = np.zeros((frames, 4), np.float32)
    values = np.zeros(frames * HOP, np.float32)
    values[: len(audio)] = audio
    talker = values.reshape(frames, HOP)[:, 0].round().astype(int)
    for k in range(1, voices + 1):
        probs[talker == k, k - 1] = 0.9
    if voices == 1:
        probs[talker > 0, 0] = 0.9
    return probs


def test_second_voice_measures_the_second_slot_and_sides_each_span():
    audio = np.zeros(20 * SR, np.float32)
    spans = [(0, SR), (2 * SR, 4 * SR), (5 * SR, 6 * SR), (7 * SR, 9 * SR)]
    for (lo, hi), who in zip(spans, [1, 2, 1, 2], strict=True):
        audio[lo:hi] = who
    share, sides = second_voice(_rerun(joined(audio, spans)), spans)
    # talker 1 holds 2 s, talker 2 holds 4 s: talker 2's slot is the most active
    assert share == pytest.approx(2 / 6)
    assert sides == [1, 0, 1, 0]


def test_second_voice_of_one_voice_is_zero_and_silent_spans_have_no_side():
    audio = np.zeros(10 * SR, np.float32)
    spans = [(0, SR), (2 * SR, 4 * SR)]
    audio[2 * SR : 4 * SR] = 1
    assert second_voice(_rerun(joined(audio, spans)), spans) == (0.0, [None, 0])


def _onehot_voices(audio: np.ndarray):
    calls: list[int] = []

    def voices(crops):
        calls.append(len(crops))
        out = []
        for k, (lo, hi) in enumerate(crops):
            v = np.zeros(8, np.float32)
            v[int(round(float(np.median(audio[lo:hi]))))] = 1.0
            v[7] = 0.01 * k  # not all identical
            out.append(v / np.linalg.norm(v))
        return out

    return voices, calls


def _meeting(turns, talker_of):
    """Audio whose turns carry their talker's value, and one entry per turn."""
    audio = np.zeros(int(max(t.end for t in turns) * SR) + SR, np.float32)
    for t in turns:
        audio[int(t.start * SR) : int(t.end * SR)] = talker_of(t)
    segments = [(t.speaker, t.start, t.end) for t in turns]
    return audio, segments


def test_a_speaker_both_checks_call_two_people_is_split():
    s0 = _turns("S0", 30)  # 120 s
    s1 = _turns("S1", 20, at=5.0)  # 80 s, one voice
    turns = sorted(s0 + s1, key=lambda t: t.start)
    talker = lambda t: 2 if (t.speaker == "S0" and 100 <= t.start < 250) else 1  # noqa: E731
    audio, segments = _meeting(turns, talker)
    segments.append(("S0", 400.0, 400.3))  # too short to rerun: stays
    voices, _ = _onehot_voices(audio)
    result = split_two_voice_speakers(turns, segments, audio, voices=voices, rerun=_rerun)
    assert result.halves == {"S2": "S0"}
    moved = [seg for seg, who in zip(segments, result.speakers, strict=True) if who == "S2"]
    assert {(a, b) for _, a, b in moved} == {(t.start, t.end) for t in s0 if 100 <= t.start < 250}
    assert result.speakers[-1] == "S0"
    # turns follow the entries that cover them
    assert {(t.start, t.end) for t in result.turns if t.speaker == "S2"} == {
        (a, b) for _, a, b in moved
    }
    assert len(result.turns) == len(turns)
    first, second = result.checks
    assert first.split_as == "S2" and first.second_share == pytest.approx(0.5)
    assert second.split_as is None and second.second_share is None  # never rerun


def test_one_voice_is_never_rerun():
    s0 = _turns("S0", 30)
    audio, segments = _meeting(s0, lambda t: 1)
    voices, _ = _onehot_voices(audio)
    reruns: list = []
    result = split_two_voice_speakers(
        s0, segments, audio, voices=voices, rerun=lambda a: reruns.append(a) or _rerun(a)
    )
    assert result.halves == {} and result.turns == s0 and reruns == []
    assert result.checks[0].centroid_cosine is not None
    assert result.checks[0].centroid_cosine > split.MAX_CENTROID_COSINE


def test_the_voice_split_alone_never_splits():
    s0 = _turns("S0", 30)
    talker = lambda t: 2 if t.start >= 150 else 1  # noqa: E731
    audio, segments = _meeting(s0, talker)
    voices, _ = _onehot_voices(audio)
    result = split_two_voice_speakers(
        s0, segments, audio, voices=voices, rerun=lambda a: _rerun(a, voices=1)
    )
    assert result.halves == {}
    assert result.checks[0].second_share == 0.0


def test_a_speaker_under_a_minute_is_not_checked():
    s0 = _turns("S0", 14)  # 56 s
    audio, segments = _meeting(s0, lambda t: 1)
    voices, calls = _onehot_voices(audio)
    result = split_two_voice_speakers(s0, segments, audio, voices=voices, rerun=_rerun)
    assert calls == [] and result.checks == []


def test_too_few_long_turns_never_split():
    s0 = _turns("S0", 9, length=7.0)  # 63 s in nine turns: under ten crops
    audio, segments = _meeting(s0, lambda t: 2 if t.start >= 40 else 1)
    voices, _ = _onehot_voices(audio)
    result = split_two_voice_speakers(s0, segments, audio, voices=voices, rerun=_rerun)
    assert result.halves == {}
    assert result.checks[0].turns_embedded == 9


def test_a_turn_no_entry_covers_stays_with_its_speaker():
    s0 = _turns("S0", 30)
    talker = lambda t: 2 if t.start >= 150 else 1  # noqa: E731
    audio, segments = _meeting(s0, talker)
    wordless = s0[-1]
    segments = [seg for seg in segments if seg[1] != wordless.start]
    voices, _ = _onehot_voices(audio)
    result = split_two_voice_speakers(s0, segments, audio, voices=voices, rerun=_rerun)
    assert result.halves == {"S1": "S0"}
    assert wordless in result.turns


def test_a_stated_count_is_never_exceeded():
    s0 = _turns("S0", 30)
    s1 = _turns("S1", 30, at=5.0)
    turns = sorted(s0 + s1, key=lambda t: t.start)
    talker = lambda t: (2 if t.start >= 150 else 1) + (2 if t.speaker == "S1" else 0)  # noqa: E731
    audio, segments = _meeting(turns, talker)

    def rerun(a):
        return _rerun(np.where(a > 2, a - 2, a))  # each speaker's two talkers on slots 0, 1

    def run(count):
        voices, _ = _onehot_voices(audio)
        return split_two_voice_speakers(
            turns, segments, audio, voices=voices, rerun=rerun, num_speakers=count
        ).halves

    assert run(2) == {}
    assert run(3) == {"S2": "S0"}  # one more speaker, the first one found
    assert run(None) == {"S2": "S0", "S3": "S1"}


def test_a_turn_spanning_the_handover_is_cut_at_the_entries():
    # The diarizer merged two people into 10 s turns. The first ten are one
    # person; in the last ten the other takes over after 3 s, and the
    # transcript's entries end at the handover.
    s0 = [SpeakerTurn("S0", 10.0 * i, 10.0 * i + 10.0) for i in range(20)]
    audio = np.zeros(201 * SR, np.float32)
    segments = []
    for i, t in enumerate(s0):
        audio[int(t.start * SR) : int(t.end * SR)] = 1
        if i < 10:
            segments.append(("S0", t.start, t.end))
            continue
        audio[int((t.start + 3) * SR) : int(t.end * SR)] = 2
        segments += [("S0", t.start, t.start + 3.0), ("S0", t.start + 3.0, t.end)]
    result = split_two_voice_speakers(
        s0, segments, audio, voices=_onehot_voices(audio)[0], rerun=_rerun
    )
    assert result.halves == {"S1": "S0"}
    moved = [t for t in result.turns if t.speaker == "S1"]
    want = [(10.0 * i + 3.0, 10.0 * i + 10.0) for i in range(10, 20)]
    assert [(t.start, t.end) for t in moved] == want
    stayed = [t for t in result.turns if t.speaker == "S0"]
    assert len(stayed) == 20  # ten whole turns, ten 3 s heads
    assert sum(t.end - t.start for t in result.turns) == pytest.approx(200.0)


def test_an_entry_touching_a_turn_takes_none_of_it():
    s0 = _turns("S0", 30)  # 4 s turns every 10 s
    talker = lambda t: 1 if t.start < 150 else 2  # noqa: E731
    audio, segments = _meeting(s0, talker)
    segments = segments[1:]  # the first turn has no words
    segments.insert(0, ("S0", 4.0, 6.0))  # an entry touching that turn's end
    audio[4 * SR : 6 * SR] = 1
    result = split_two_voice_speakers(
        s0, segments, audio, voices=_onehot_voices(audio)[0], rerun=_rerun
    )
    assert result.halves == {"S1": "S0"}
    assert result.speakers[0] == "S1"  # the touching entry moved...
    assert s0[0] in result.turns  # ...and took none of the turn it touches
