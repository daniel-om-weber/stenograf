"""The voice embedder: ReDimNet2-B6 on ONNX Runtime, and the rule that turns
a speaker's speech into one voiceprint.

Re-ID voiceprints must come from one embedding model whichever path produced
the turns — saved profiles are cosine-matched against
``assets.SPEAKER_EMBEDDING`` vectors, so a second embedding model would
silently break every enrolled profile. Hence one :class:`VoiceEmbedder`, and
one rule (:class:`VoiceRule`) behind both :class:`VoiceCollector` and
:func:`cluster_embeddings`."""

from __future__ import annotations

import contextlib
import copy
import math
import os
import threading
from collections.abc import Callable, Collection, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stenograf import assets
from stenograf.audio import SAMPLE_RATE, l2_normalize, to_float32
from stenograf.diarization.base import SpeakerTurn

MIN_EMBED_SECONDS = 0.5
"""Speech spans shorter than this are too brief for a reliable voice embedding;
they are skipped when a cluster has any longer span, and used only as a last
resort (see :func:`_preferred`)."""
_MIN_EMBED_SAMPLES = int(MIN_EMBED_SECONDS * SAMPLE_RATE)

_MAX_POOL = 4
"""Pool workers at most. The model is memory-bound: four concurrent 5 s spans
cost 0.82 CPU-s per 10 s of speech against 0.59 for one, eight cost 1.20, and
each concurrent call adds its share of arena (1.4 GB at four, 2.5 GB at
eight; M4 Max, 2026-11-18). With two minutes per speaker to embed, four is
seconds at the stop."""


def _pool_workers() -> int:
    """Worker threads for the embedding pool: :data:`_MAX_POOL`, or fewer
    where the process may run on fewer processors (the scheduler affinity
    mask on Linux, so a cgroup-quota'd container is not oversubscribed)."""
    available = os.cpu_count() or 1
    sched_getaffinity = getattr(os, "sched_getaffinity", None)  # Linux-only
    if sched_getaffinity is not None:
        with contextlib.suppress(OSError):
            available = len(sched_getaffinity(0))
    return max(1, min(_MAX_POOL, available))


@dataclass(frozen=True)
class VoiceRule:
    """Which of a speaker's speech becomes the voiceprint.

    One rule for every path — a live meeting's stop, a transcribed file, an
    enrollment — because a saved profile is only comparable with a cluster
    built the same way."""

    budget_s: float | None = None
    """Speech per speaker, taken in meeting order; ``None`` = all of it."""
    max_span_s: float | None = None
    """A longer span is embedded as equal parts no longer than this, which
    bounds one forward pass's memory and time; ``None`` = never split."""


_MIN_INPUT_SAMPLES = 800
"""The graph's shortest input: 800 samples embed, 400 raise (its front end
and first stride need the frames; onnxruntime 1.27, 2026-11-18)."""


class VoiceEmbedder:
    """The model's session (lazy), its worker pool, and per-cluster voices.

    ``model`` overrides the fetched ``assets.SPEAKER_EMBEDDING`` with another
    export of the same interface: a ``(1, samples)`` float32 16 kHz waveform
    in, a ``(1, dim)`` embedding out (``eval/embedders/export_redimnet.py``)."""

    rule = VoiceRule(budget_s=120.0, max_span_s=5.0)
    """The first two minutes of a speaker name exactly as the whole meeting
    does, and so does one minute — two keeps the wider worst-case margin,
    +0.215 against +0.162 (2026-11-18, ``eval/README.md``). Five-second parts
    because the model's memory grows with the span — about 80 MB per second
    held by the session's arena, 570 MB at 5 s against 980 MB at 10 s and
    8 GB at 120 s — and so does its time per second of speech (0.60 CPU-s per
    10 s at 5 s, 0.68 at 10 s, 1.66 at 120 s; M4 Max, same date)."""

    def __init__(
        self,
        model: Path | None = None,
        *,
        progress: assets.ProgressHook | None = None,
    ) -> None:
        self._model = model
        self._progress = progress
        self._session: Any = None
        self._pool: ThreadPoolExecutor | None = None
        self._init_lock = threading.Lock()  # guards every lazy field

    def pool(self) -> ThreadPoolExecutor:
        """The worker pool every embedding call of this embedder runs on.

        One pool, sized once (:func:`_pool_workers`), so concurrent channels
        share one budget and never oversubscribe the machine. Each call is
        independent and the session runs single-threaded, so the result does
        not depend on the worker count."""
        if self._pool is None:
            with self._init_lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=_pool_workers(), thread_name_prefix="embed"
                    )
        return self._pool

    def cluster_embeddings(
        self, turns: list[SpeakerTurn], samples: np.ndarray
    ) -> dict[str, np.ndarray]:
        """:func:`cluster_embeddings` through this embedder and its pool."""
        return cluster_embeddings(turns, samples, self.embed, pool=self.pool(), rule=self.rule)

    def collector(self, hop: int) -> VoiceCollector:
        """A :class:`VoiceCollector` under this embedder's rule, for activity
        rows ``hop`` samples apart."""
        return VoiceCollector(self.rule, hop=hop)

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        """One voice over a single-speaker channel's speech ``turns`` (see
        :meth:`~stenograf.diarization.base.Diarizer.channel_embedding`)."""
        if not turns:
            return None
        return self.cluster_embeddings(turns, samples).get(turns[0].speaker)

    def embed(self, audio: np.ndarray, run_options: Any = None) -> np.ndarray | None:
        """L2-normalized voice embedding of a mono 16 kHz float32 slice, or None
        when the slice is too short for the model. ``run_options`` is an
        ``onnxruntime.RunOptions`` whose ``terminate`` abandons the call."""
        if len(audio) < _MIN_INPUT_SAMPLES:
            return None
        waveform = np.ascontiguousarray(audio, dtype=np.float32)[None]
        (embedding,) = self._load().run(None, {"waveform": waveform}, run_options)
        vector = l2_normalize(np.asarray(embedding[0], dtype=np.float32))
        return vector if np.isfinite(vector).all() else None

    def load(self, *, fetch: bool = True) -> bool:
        """Load the session now, on the calling thread; ``fetch=False`` loads
        only from the model cache and answers whether it could."""
        cached = self._model or assets.cached_path(assets.SPEAKER_EMBEDDING)
        if cached is None and not fetch:
            return False
        self._load()
        return True

    def _load(self) -> Any:
        # ONE shared session at ONE intra-op thread, used from every pool
        # worker concurrently (ORT's Run is thread-safe): the calling thread
        # does the work, so a caller on the efficiency cores embeds there,
        # parallelism belongs to the pool, and every path computes the same
        # vectors.
        if self._session is None:
            with self._init_lock:
                if self._session is None:
                    import onnxruntime

                    from stenograf import cpu

                    path = self._model or assets.fetch(assets.SPEAKER_EMBEDDING, self._progress)
                    self._session = onnxruntime.InferenceSession(
                        str(path),
                        cpu.session_options(cpu.ThreadPlan(1, None)),
                        providers=["CPUExecutionProvider"],
                    )
        return self._session


Span = tuple[int, int]
"""``[lo, hi)`` in samples of one channel."""


class _Sums:
    """Per-speaker running sums of duration-weighted span embeddings.

    A voiceprint is a mean, so it needs no span kept: the weighted sum and
    how much speech was planned are the whole state. Planning and committing
    are separate so a caller can embed between them and lose nothing when an
    embedding call is abandoned."""

    def __init__(self, rule: VoiceRule) -> None:
        self.rule = rule
        self._sum: dict[Hashable, np.ndarray] = {}
        self._planned: dict[Hashable, int] = {}  # samples handed to plan(), embedded or not

    def room(self, speaker: Hashable) -> int | None:
        """Samples ``speaker`` may still bank; ``None`` = no budget."""
        if self.rule.budget_s is None:
            return None
        budget = int(self.rule.budget_s * SAMPLE_RATE)
        return max(0, budget - self._planned.get(speaker, 0))

    def planned(self, speaker: Hashable) -> bool:
        return self._planned.get(speaker, 0) > 0

    def plan(self, speaker: Hashable, spans: list[Span]) -> list[tuple[Hashable, Span]]:
        """The pieces of ``spans`` (time-ordered) to embed for ``speaker``:
        cut where the budget fills, split at the rule's span limit. Counts
        them against the budget whether or not they embed — the budget is
        "the first N seconds of speech", the same seconds on every path."""
        jobs: list[tuple[Hashable, Span]] = []
        for lo, hi in spans:
            room = self.room(speaker)
            if room is not None:
                hi = min(hi, lo + room)
            if hi <= lo:
                continue
            self._planned[speaker] = self._planned.get(speaker, 0) + hi - lo
            jobs += [(speaker, part) for part in self._split(lo, hi)]
        return jobs

    def _split(self, lo: int, hi: int) -> list[Span]:
        if self.rule.max_span_s is None:
            return [(lo, hi)]
        parts = math.ceil((hi - lo) / (self.rule.max_span_s * SAMPLE_RATE))
        edges = np.linspace(lo, hi, parts + 1).round().astype(int).tolist()
        return list(zip(edges[:-1], edges[1:], strict=True))

    def bank(self, jobs: list[tuple[Hashable, Span]], vectors: list[np.ndarray | None]) -> None:
        for (speaker, (lo, hi)), vector in zip(jobs, vectors, strict=True):
            if vector is not None:
                weighted = np.asarray(vector, np.float64) * ((hi - lo) / SAMPLE_RATE)
                self._sum[speaker] = self._sum.get(speaker, 0.0) + weighted

    def vectors(self) -> dict[Hashable, np.ndarray]:
        return {k: l2_normalize(v).astype(np.float32) for k, v in self._sum.items()}


Embed = Callable[[np.ndarray], np.ndarray | None]


def _run(
    jobs: list[tuple[Hashable, Span]],
    read: Callable[[int, int], np.ndarray],
    embed: Embed,
    pool: ThreadPoolExecutor | None,
) -> list[np.ndarray | None]:
    # Order-preserving, so each speaker's sum adds in the sequential order
    # and the result does not depend on the pool.
    mapper = pool.map if pool is not None else map
    return list(mapper(embed, [read(lo, hi) for _, (lo, hi) in jobs]))


def _preferred(clean: list[Span], raw: list[Span]) -> list[Span]:
    """The spans a speaker's voice comes from: clean ones of at least
    :data:`MIN_EMBED_SECONDS` if there are any, else every clean one, else
    the raw turns — a contaminated embedding still serves naming and
    enrollment, an absent one blocks them."""
    spans = clean or raw
    return [s for s in spans if s[1] - s[0] >= _MIN_EMBED_SAMPLES] or spans


class VoiceCollector:
    """One channel's voiceprints, fed the diarizer's speaker activity in
    meeting order — all at once, or step by step while the meeting records.

    Speech counts as a speaker's own only where no other slot is active:
    overlapped audio measurably worsens the embedding (12.84 → 14.11 % EER
    with overlap included, and frame-level identity in overlap is near-chance
    — ``eval/diarization-sota-2026.md``). Activity is judged before any
    speaker-count cap, which is only known at the stop; a slot the cap later
    removes is simply never asked for (:meth:`finish`).

    The spans it embeds depend only on the activity, never on how it was cut
    into :meth:`feed` calls: a run still open at the end of a call waits for
    the next one. A call that raises leaves the collector as it was, so an
    abandoned embedding costs only that call."""

    def __init__(self, rule: VoiceRule, *, hop: int) -> None:
        self._sums = _Sums(rule)
        self._hop = hop
        self.frames = 0
        """Activity rows consumed so far."""
        self._open: dict[int, int] = {}  # slot → first frame of its running clean stretch
        self._short: dict[int, list[Span]] = {}  # clean fragments of slots with nothing banked

    def feed(
        self,
        active: np.ndarray,
        read: Callable[[int, int], np.ndarray],
        embed: Embed,
        *,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
        """Consume the next rows of (frames, slots) boolean ``active``."""
        sums, opened, short = copy.deepcopy((self._sums, self._open, self._short))
        alone = active & (active.sum(1, keepdims=True) == 1)
        end = self.frames + len(active)
        jobs: list[tuple[Hashable, Span]] = []
        for slot in range(active.shape[1]):
            if sums.room(slot) == 0:
                opened.pop(slot, None)
                continue
            edges = np.diff(np.concatenate([[0], alone[:, slot].astype(np.int8), [0]]))
            starts = (np.nonzero(edges == 1)[0] + self.frames).tolist()
            ends = (np.nonzero(edges == -1)[0] + self.frames).tolist()
            if slot in opened:  # the stretch the last call left running
                if starts and starts[0] == self.frames:
                    starts[0] = opened.pop(slot)
                else:
                    starts.insert(0, opened.pop(slot))
                    ends.insert(0, self.frames)
            for first, last in zip(starts, ends, strict=True):
                lo, hi = first * self._hop, last * self._hop
                room = sums.room(slot)
                if last == end and (room is None or hi - lo < max(room, _MIN_EMBED_SAMPLES)):
                    opened[slot] = first  # may still grow; a full budget needs no more of it
                elif hi - lo >= _MIN_EMBED_SAMPLES:
                    jobs += sums.plan(slot, [(lo, hi)])
                    short.pop(slot, None)
                elif not sums.planned(slot):
                    short.setdefault(slot, []).append((lo, hi))
        sums.bank(jobs, _run(jobs, read, embed, pool))
        self._sums, self._open, self._short = sums, opened, short
        self.frames = end

    def finish(
        self,
        raw: dict[int, list[Span]],
        read: Callable[[int, int], np.ndarray],
        embed: Embed,
        *,
        pool: ThreadPoolExecutor | None = None,
    ) -> dict[int, np.ndarray]:
        """Voiceprints of the slots in ``raw`` (each slot's final turns, the
        last resort of a slot never heard alone), once the activity ended."""
        sums = copy.deepcopy(self._sums)
        jobs: list[tuple[Hashable, Span]] = []
        for slot, turns in raw.items():
            clean = list(self._short.get(slot, []))
            if slot in self._open:
                clean.append((self._open[slot] * self._hop, self.frames * self._hop))
            if sums.planned(slot):
                clean = [s for s in clean if s[1] - s[0] >= _MIN_EMBED_SAMPLES]
                jobs += sums.plan(slot, clean)
            else:
                jobs += sums.plan(slot, _preferred(clean, sorted(turns)))
        sums.bank(jobs, _run(jobs, read, embed, pool))
        vectors = sums.vectors()
        return {slot: vectors[slot] for slot in raw if slot in vectors}


def cluster_embeddings(
    turns: list[SpeakerTurn],
    samples: np.ndarray,
    embed: Embed,
    *,
    pool: ThreadPoolExecutor | None = None,
    rule: VoiceRule | None = None,
    only: Collection[str] | None = None,
) -> dict[str, np.ndarray]:
    """A duration-weighted mean voice embedding per cluster of ``turns``.

    :class:`VoiceCollector`'s rule for callers that hold turns rather than
    the diarizer's activity (a solo channel's speech spans, eval's reference
    turns): spans where a *different* cluster is active too are excluded,
    then :func:`_preferred` picks, ``rule`` cuts, and the L2-normalized span
    embeddings are averaged by duration and re-normalized. A cluster with no
    embeddable audio is omitted, and so is every cluster outside ``only``
    (whose turns still count as overlap)."""
    audio = to_float32(samples)
    by_cluster: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_cluster.setdefault(turn.speaker, []).append(turn)
    overlap = _overlap_spans(by_cluster)

    sums = _Sums(rule or VoiceRule())
    jobs: list[tuple[Hashable, Span]] = []
    for speaker, cluster_turns in by_cluster.items():
        if only is not None and speaker not in only:
            continue
        raw = sorted((t.start, t.end) for t in cluster_turns)
        clean, rough = (
            [(int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)) for s, e in spans]
            for spans in (_subtract_spans(raw, overlap), raw)
        )
        jobs += sums.plan(speaker, _preferred(clean, rough))
    sums.bank(jobs, _run(jobs, lambda lo, hi: audio[lo:hi], embed, pool))
    return {str(k): v for k, v in sums.vectors().items()}


def _overlap_spans(by_cluster: dict[str, list[SpeakerTurn]]) -> list[tuple[float, float]]:
    """Spans where at least two clusters speak at once, sorted and disjoint.

    Each cluster's turns are unioned first, so a cluster overlapping itself
    (back-to-back or re-emitted turns) never counts as overlapped speech."""
    events: list[tuple[float, int]] = []
    for cluster_turns in by_cluster.values():
        merged: list[tuple[float, float]] = []
        for t in sorted(cluster_turns, key=lambda t: t.start):
            if merged and t.start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t.end))
            else:
                merged.append((t.start, t.end))
        for start, end in merged:
            events += [(start, 1), (end, -1)]

    events.sort()  # ends before starts at equal times: touching spans never overlap
    spans: list[tuple[float, float]] = []
    depth = 0
    overlap_start = 0.0
    for time_, delta in events:
        was = depth
        depth += delta
        if was < 2 <= depth:
            overlap_start = time_
        elif was >= 2 > depth:
            spans.append((overlap_start, time_))
    return spans


def _subtract_spans(
    spans: list[tuple[float, float]], cuts: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``spans`` minus ``cuts`` (both lists of (start, end); cuts sorted, disjoint)."""
    result = []
    for start, end in spans:
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                result.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result
