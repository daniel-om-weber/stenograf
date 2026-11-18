"""Unit coverage for the voice embedder's pure-python aggregation.

The real-backend test (``test_diarization_loop_real.py``) is gated on cached
ONNX models + private audio, so on a fresh checkout / Linux CI ``embedding.py``
executes zero test lines — yet its aggregation feeds every platform's re-ID.
These drive ``cluster_embeddings`` and ``l2_normalize`` through the real
``embed`` with a fake ``SpeakerEmbeddingExtractor``, so the aggregation logic
(per-cluster unit-norm mean, duration weighting, empty-cluster omission,
short-turn fallback, overlap exclusion, zero-vector guard) is verified
without any model.

The precedent for why this matters is the MLX thread-stream bug: a real backend
broke what every mocked test passed green.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from stenograf.audio import SAMPLE_RATE, l2_normalize
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.embedding import (
    VoiceCollector,
    VoiceEmbedder,
    VoiceRule,
    cluster_embeddings,
)


class _FakeStream:
    def accept_waveform(self, rate: int, audio: np.ndarray) -> None:
        self.audio = audio

    def input_finished(self) -> None:
        pass


class _SeqExtractor:
    """Returns preset vectors in call order; ``embed`` L2-normalizes the result."""

    def __init__(self, vectors: list) -> None:
        self._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        self._i = 0

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def is_ready(self, stream: _FakeStream) -> bool:
        return True

    def compute(self, stream: _FakeStream) -> np.ndarray:
        vector = self._vectors[self._i]
        self._i += 1
        return vector


def _embed(vectors: list):
    """The real ``embed`` (slice guard + extractor protocol + l2) over fakes:
    only the extractor seam is replaced, no ONNX loads."""
    embedder = VoiceEmbedder()
    embedder._extractor = _SeqExtractor(vectors)
    return embedder.embed


def _turns(*specs: tuple[str, float, float]) -> list[SpeakerTurn]:
    return [SpeakerTurn(speaker, start, end) for speaker, start, end in specs]


_AUDIO = np.ones(3 * SAMPLE_RATE, dtype=np.int16)


def test_embeddings_are_per_cluster_and_unit_norm():
    turns = _turns(("S0", 0.0, 1.0), ("S1", 1.0, 2.0))
    embeddings = cluster_embeddings(turns, _AUDIO, _embed([[3.0, 0.0, 0.0], [0.0, 0.0, 4.0]]))

    assert set(embeddings) == {"S0", "S1"}
    for vector in embeddings.values():
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    assert float(embeddings["S0"] @ embeddings["S1"]) == pytest.approx(0.0, abs=1e-6)


def test_mean_is_duration_weighted():
    # One cluster, two long turns of 1 s and 2 s pointing along x and y. The longer
    # turn dominates: mean = normalize(x*1 + y*2) = [1, 2, 0]/sqrt(5).
    turns = _turns(("S0", 0.0, 1.0), ("S0", 1.0, 3.0))
    embeddings = cluster_embeddings(turns, _AUDIO, _embed([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    expected = np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0)
    assert embeddings["S0"] == pytest.approx(expected, abs=1e-6)


def test_cluster_with_no_embeddable_audio_is_omitted():
    # S1's only turn is zero-length → empty slice → no embedding → cluster dropped.
    turns = _turns(("S0", 0.0, 1.0), ("S1", 2.0, 2.0))
    embeddings = cluster_embeddings(turns, _AUDIO, _embed([[1.0, 0.0, 0.0]]))
    assert set(embeddings) == {"S0"}


def test_short_only_cluster_falls_back_to_its_short_turns():
    # A cluster whose only turn is below MIN_EMBED_SECONDS still gets embedded
    # (the `long or spans` fallback), rather than being dropped.
    turns = _turns(("S0", 0.0, 0.2))
    embeddings = cluster_embeddings(turns, _AUDIO, _embed([[0.0, 5.0, 0.0]]))
    assert set(embeddings) == {"S0"}
    assert np.linalg.norm(embeddings["S0"]) == pytest.approx(1.0, abs=1e-6)


def test_overlap_with_another_cluster_is_excluded():
    # S0 speaks 0–2 and 3–5; S1 speaks 1–2 (inside S0's first turn). S0's
    # first embeddable span shrinks to 0–1, so its weight is 1 against 2 for
    # the clean 3–5 span; with overlap included both would weigh 2 and the
    # mean would sit at 45°.
    turns = _turns(("S0", 0.0, 2.0), ("S0", 3.0, 5.0), ("S1", 1.0, 2.0))
    embeddings = cluster_embeddings(
        turns,
        np.ones(6 * SAMPLE_RATE, dtype=np.int16),
        _embed([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    expected = np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0)
    assert embeddings["S0"] == pytest.approx(expected, abs=1e-6)


def test_fully_overlapped_cluster_falls_back_to_raw_turns():
    # S1 speaks only while S0 does: no clean audio at all. It still gets an
    # embedding (from its raw, contaminated turn) — an absent one would block
    # naming and enrollment.
    turns = _turns(("S0", 0.0, 3.0), ("S1", 1.0, 2.0))
    embeddings = cluster_embeddings(
        turns,
        np.ones(3 * SAMPLE_RATE, dtype=np.int16),
        _embed([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert set(embeddings) == {"S0", "S1"}


def test_overlap_spans_ignore_self_overlap_and_touching_turns():
    from stenograf.diarization.embedding import _overlap_spans

    def turns(speaker, *spans):
        return [SpeakerTurn(speaker, s, e) for s, e in spans]

    # A cluster overlapping itself is one voice, not overlapped speech.
    assert _overlap_spans({"S0": turns("S0", (0.0, 2.0), (1.0, 3.0))}) == []
    # Touching turns of different clusters share no time.
    assert _overlap_spans({"S0": turns("S0", (0.0, 1.0)), "S1": turns("S1", (1.0, 2.0))}) == []
    # A genuine cross-cluster overlap, exactly the shared interval.
    assert _overlap_spans({"S0": turns("S0", (0.0, 2.0)), "S1": turns("S1", (1.0, 3.0))}) == [
        (1.0, 2.0)
    ]


def test_subtract_spans_splits_and_clips():
    from stenograf.diarization.embedding import _subtract_spans

    assert _subtract_spans([(0.0, 5.0)], [(1.0, 2.0), (3.0, 4.0)]) == [
        (0.0, 1.0),
        (2.0, 3.0),
        (4.0, 5.0),
    ]
    assert _subtract_spans([(1.0, 3.0)], [(0.0, 2.0)]) == [(2.0, 3.0)]
    assert _subtract_spans([(1.0, 3.0)], [(0.0, 4.0)]) == []


def test_l2_normalize_guards_the_zero_vector():
    zero = l2_normalize(np.zeros(3, dtype=np.float32))
    assert np.all(zero == 0.0) and not np.any(np.isnan(zero))  # no div-by-zero / NaN
    assert l2_normalize(np.array([3.0, 4.0, 0.0])) == pytest.approx([0.6, 0.8, 0.0])


def test_pool_workers_falls_back_when_probes_fail(monkeypatch):
    # sysctl dead (macOS branch) and /proc/cpuinfo unreadable (Linux branch)
    # must land on the capped logical-count fallback, never raise.
    from stenograf.diarization import embedding

    def broken_run(*args, **kwargs):
        raise OSError("no sysctl here")

    monkeypatch.setattr(embedding.subprocess, "run", broken_run)
    workers = embedding._pool_workers()
    assert 1 <= workers <= embedding._MAX_POOL
    ceiling = min(embedding._MAX_THREADS, os.cpu_count() or 1)
    if sys.platform not in ("darwin",) and not sys.platform.startswith("linux"):
        assert workers <= ceiling


# -- the rule over diarizer activity ---------------------------------------

_HOP = 160


def _span_embed(calls: list[tuple[int, int]] | None = None):
    """A model-free ``embed`` over audio whose sample i holds the value i:
    the vector encodes where the span starts and how long it is."""

    def embed(audio: np.ndarray) -> np.ndarray | None:
        if len(audio) == 0:
            return None
        if calls is not None:
            calls.append((int(audio[0]), int(audio[0]) + len(audio)))
        return np.array([float(audio[0]) + 1.0, float(len(audio)), 1.0])

    return embed


def _ramp(frames: int) -> np.ndarray:
    return np.arange(frames * _HOP, dtype=np.float64)


def _collect(active: np.ndarray, rule: VoiceRule, cuts: list[int], calls=None):
    audio = _ramp(len(active))
    collector = VoiceCollector(rule, hop=_HOP)
    read = lambda lo, hi: audio[lo:hi]  # noqa: E731
    embed = _span_embed(calls)
    for lo, hi in zip([0, *cuts], [*cuts, len(active)], strict=True):
        collector.feed(active[lo:hi], read, embed)
    raw = {
        slot: [(0, len(active) * _HOP)] for slot in range(active.shape[1]) if active[:, slot].any()
    }
    return collector.finish(raw, read, embed)


def _activity(frames: int, *runs: tuple[int, int, int]) -> np.ndarray:
    active = np.zeros((frames, 3), bool)
    for slot, lo, hi in runs:
        active[lo:hi, slot] = True
    return active


_RUNS = _activity(
    3000,
    (0, 0, 700),  # crosses the cuts below
    (1, 600, 640),  # overlaps slot 0: neither banks 600–640
    (0, 900, 2200),
    (1, 2300, 2330),  # a fragment only
    (2, 2500, 3000),  # still running when the activity ends
)


@pytest.mark.parametrize("rule", [VoiceRule(), VoiceRule(budget_s=8.0, max_span_s=3.0)])
@pytest.mark.parametrize(
    "cuts", [[1], [650, 651], [100, 999, 2100, 2999], list(range(50, 3000, 50))]
)
def test_collector_result_does_not_depend_on_how_activity_is_fed(rule, cuts):
    whole = _collect(_RUNS, rule, [])
    streamed = _collect(_RUNS, rule, cuts)
    assert set(whole) == set(streamed) == {0, 1, 2}
    for slot in whole:
        assert np.array_equal(whole[slot], streamed[slot])


def test_collector_embeds_only_speech_a_slot_has_alone():
    calls: list[tuple[int, int]] = []
    _collect(_RUNS, VoiceRule(), [], calls)
    spans = sorted((lo // _HOP, hi // _HOP) for lo, hi in calls)
    # Slot 0: 0–600, 640–700, 900–2200. Slot 1 never had half a second alone,
    # so its fragment stands in. Slot 2's open run is closed by the end.
    assert spans == [(0, 600), (640, 700), (900, 2200), (2300, 2330), (2500, 3000)]


def test_collector_budget_takes_the_first_seconds_and_splits_long_spans():
    calls: list[tuple[int, int]] = []
    _collect(_RUNS, VoiceRule(budget_s=8.0, max_span_s=3.0), [], calls)
    slot0 = sorted((lo // _HOP, hi // _HOP) for lo, hi in calls if lo < 2300 * _HOP)
    # 6 s in two equal parts, 0.6 s, then 1.4 s of the third run fills 8 s.
    assert slot0 == [(0, 300), (300, 600), (640, 700), (900, 1040)]


def test_collector_falls_back_to_raw_turns_for_a_slot_never_alone():
    active = _activity(400, (0, 0, 400), (1, 100, 300))
    calls: list[tuple[int, int]] = []
    voices = _collect(active, VoiceRule(), [150], calls)
    assert set(voices) == {0, 1}
    assert (0, 400 * _HOP) in calls  # slot 1's raw turn, as handed to finish()


def test_collector_keeps_its_state_when_an_embedding_call_is_abandoned():
    audio = _ramp(len(_RUNS))
    read = lambda lo, hi: audio[lo:hi]  # noqa: E731
    collector = VoiceCollector(VoiceRule(), hop=_HOP)
    collector.feed(_RUNS[:1000], read, _span_embed())

    def abandoned(_audio):
        raise RuntimeError("terminated")

    with pytest.raises(RuntimeError):
        collector.feed(_RUNS[1000:], read, abandoned)
    assert collector.frames == 1000
    collector.feed(_RUNS[1000:], read, _span_embed())
    raw = {slot: [(0, 1)] for slot in range(3)}
    resumed = collector.finish(raw, read, _span_embed())
    whole = _collect(_RUNS, VoiceRule(), [])
    assert all(np.array_equal(resumed[slot], whole[slot]) for slot in whole)


def test_cluster_embeddings_applies_the_budget_in_time_order():
    calls: list[tuple[int, int]] = []
    audio = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
    turns = _turns(("S0", 4.0, 9.0), ("S0", 0.0, 2.0))
    cluster_embeddings(turns, audio, _span_embed(calls), rule=VoiceRule(budget_s=3.0))
    assert calls == [(0, 2 * SAMPLE_RATE), (4 * SAMPLE_RATE, 5 * SAMPLE_RATE)]
