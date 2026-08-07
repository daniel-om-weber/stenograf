"""Unit tests for the owned diarization loop's pure pieces.

The loop's reference semantics live in ``eval/diarization-loop-spec.md`` and
its behavioral gate is ``eval/loop_parity.py`` (byte parity against sherpa's
implementation on the corpus harness); these tests pin the pieces that are
cheap to break silently — clustering edge cases, the turn assembly rules —
without loading any ONNX model.
"""

from __future__ import annotations

import numpy as np

from stenograf.diarization.loop import (
    _POWERSET,
    MIN_DURATION_OFF,
    MIN_DURATION_ON,
    RF_SHIFT,
    RF_SIZE,
    _cluster,
    _runs,
    _to_turns,
)


def _unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestCluster:
    def test_empty_and_singleton(self):
        assert _cluster(np.zeros((0, 4)), 0.5, None).tolist() == []
        assert _cluster(np.ones((1, 4)), 0.5, None).tolist() == [0]

    def test_known_count_caps_at_n(self):
        vectors = np.stack([_unit(1.0, 0.0), _unit(0.0, 1.0)])
        labels = _cluster(vectors, 0.5, 5, "ward")
        assert sorted(set(labels.tolist())) == [0, 1]

    def test_ward_separates_known_count(self):
        vectors = np.stack(
            [_unit(1.0, 0.02 * i) for i in range(6)]
            + [_unit(0.02 * i, 1.0) for i in range(6)]
        )
        labels = _cluster(vectors, 0.5, 2, "ward")
        assert len(set(labels.tolist())) == 2
        assert len({labels[i] for i in range(6)}) == 1
        assert len({labels[i] for i in range(6, 12)}) == 1

    def test_deterministic(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(40, 8)).astype(np.float32)
        for method in ("complete", "ward", "nmesc"):
            first = _cluster(vectors, 0.5, 3, method)
            second = _cluster(vectors, 0.5, 3, method)
            assert first.tolist() == second.tolist()

    def test_threshold_mode_ignores_method(self):
        # Production reaches threshold mode on installs without stenodiar; it
        # keeps the reference complete-linkage cut whatever method says.
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(20, 8)).astype(np.float32)
        reference = _cluster(vectors, 0.5, None, "complete")
        for method in ("ward", "nmesc", "average"):
            assert _cluster(vectors, 0.5, None, method).tolist() == reference.tolist()

    def test_labels_are_first_appearance_ordered(self):
        vectors = np.stack([_unit(0.0, 1.0), _unit(1.0, 0.0), _unit(0.0, 1.0)])
        labels = _cluster(vectors, 0.5, 2, "ward")
        assert labels[0] == 0  # first vector defines cluster 0


class TestAssemblyRules:
    def test_runs(self):
        mask = np.array([False, True, True, False, True], dtype=bool)
        assert _runs(mask) == [(1, 3), (4, 5)]

    def test_to_turns_min_duration_and_merge(self):
        frames_on = int(MIN_DURATION_ON * 16_000 / RF_SHIFT) + 2  # safely > 0.3 s
        gap = int(MIN_DURATION_OFF * 16_000 / RF_SHIFT) - 2  # gap ≤ 0.5 s: merges
        active = np.zeros((3 * frames_on + gap, 1), dtype=np.int64)
        active[:frames_on, 0] = 1
        active[frames_on + gap : 2 * frames_on + gap, 0] = 1
        turns = _to_turns(active, active.sum(axis=1))
        assert [t.speaker for t in turns] == ["S0"]  # one merged turn
        offset = 0.5 * RF_SIZE / 16_000
        assert turns[0].start == offset  # frame 0 lands mid receptive field

    def test_to_turns_drops_short(self):
        frames = int(MIN_DURATION_ON * 16_000 / RF_SHIFT) - 1  # strictly under
        active = np.zeros((frames + 10, 1), dtype=np.int64)
        active[:frames, 0] = 1
        assert _to_turns(active, active.sum(axis=1)) == []


class _FakeSession:
    """Segmentation-session stub: fixed logits for one padded chunk."""

    def __init__(self, logits: np.ndarray) -> None:
        self._logits = logits

    def run(self, outputs, feeds):
        return (self._logits[None],)


def test_single_chunk_ignores_count_and_returns_powerset_speakers():
    # ≤10 s of audio is one padded chunk: the three local powerset speakers
    # ARE the output — no clustering, no embeddings, and num_speakers is
    # ignored (spec §1.8; sherpa's HandleOneChunkSpecialCase behaves
    # identically, verified against the installed package 2026-08-07).
    # Production reaches this through `profiles enroll` short clips.
    from stenograf.diarization.loop import FRAMES, OwnDiarizer

    logits = np.zeros((FRAMES, 7), dtype=np.float32)
    logits[:80, 1] = 5.0  # local speaker 0 active ~1.35 s; class 0 elsewhere
    d = OwnDiarizer()
    d._session = _FakeSession(logits)
    audio = np.zeros(6 * 16_000, dtype=np.float32)
    outputs = [d.diarize(audio, k) for k in (None, 1, 2)]
    assert outputs[0] == outputs[1] == outputs[2]
    assert [t.speaker for t in outputs[0]] == ["S0"]


def test_powerset_rows_are_pyannote_order():
    assert _POWERSET.shape == (7, 3)
    cardinality = _POWERSET.sum(axis=1)
    assert cardinality.tolist() == sorted(cardinality.tolist())  # 0, then 1s, then 2s
