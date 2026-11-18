"""The voice embedder: ERes2Net via sherpa-onnx's ``SpeakerEmbeddingExtractor``.

Re-ID voiceprints must come from one embedding model whichever diarizer
produced the turns — saved profiles are cosine-matched against
``assets.SPEAKER_EMBEDDING`` vectors (see that asset for why not CAM++), so a
second embedding model would silently break every enrolled profile. Hence one
:class:`VoiceEmbedder` and one :func:`cluster_embeddings`, which embeds a
channel's spans in parallel on the embedder's own worker pool."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stenograf import assets
from stenograf.audio import SAMPLE_RATE, l2_normalize, to_float32
from stenograf.diarization.base import SpeakerTurn

MIN_EMBED_SECONDS = 0.5
"""Speech spans shorter than this are too brief for a reliable voice embedding;
they are skipped when a cluster has any longer span, and used only as a last
resort (see :func:`_preferred`)."""
_MIN_EMBED_SAMPLES = int(MIN_EMBED_SECONDS * SAMPLE_RATE)

_MAX_THREADS = 8
"""The pool's fallback worker cap where physical cores are unknowable. The
embedder's ORT session runs at ONE intra-op thread and the pool parallelizes
across calls instead — see :func:`_pool_workers`."""

_MAX_POOL = 16
"""Hard ceiling on pool workers: efficiency decays past the physical core
count (2026-08-09: 12 workers ≈ 8 on 16 logical CPUs) and a many-core
server gains nothing this workload can use."""


def _pool_workers() -> int:
    """Worker threads for the embedding pool: one per core worth using.

    Outer call-level threading at one intra-op thread each beats intra-op
    threading on this workload (2026-08-09, M4 Max, 37.6-min channel:
    intra-op scaling caps at 1.5× over single-thread while a pool of
    intra-op-1 workers reaches 2.5–2.6× over the 8-intra-op-thread config,
    bit-exact against the sequential intra-op-1 reference; the pool wins by
    less on x86, where intra-op threading scales better — `eval/README.md`).
    Logical CPUs oversubscribe — hyperthreads and E-cores measured to add
    nothing (12 workers ≈ 8 on 16 logical CPUs) — so: P-cores via sysctl on
    macOS, physical cores via /proc/cpuinfo on Linux (bounded by the
    scheduler affinity mask, so a cgroup-quota'd container never spawns a
    host-sized pool), logical count capped at :data:`_MAX_THREADS`
    elsewhere; every branch capped at :data:`_MAX_POOL`.

    The Linux branch counts every physical core, fast or not, where macOS
    asks for performance cores specifically — no counterpart is universally
    available here (`cpu_capacity`, `cpuinfo_max_freq` and the PMU's
    `cpu_core` list each cover only some machines). Measured 2026-08-11 on a
    4+8 hybrid: counting all 12 is indistinguishable from 8 once arms are
    interleaved. That was a same-ISA split (Zen5 + Zen5c, clock and cache
    only); hybrid Intel, where E-cores differ more, is untested."""
    affinity = None
    sched_getaffinity = getattr(os, "sched_getaffinity", None)  # Linux-only
    if sched_getaffinity is not None:
        try:
            affinity = len(sched_getaffinity(0))
        except OSError:
            affinity = None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if out.isdigit() and int(out) > 0:
            return max(1, min(int(out), _MAX_POOL))
    if sys.platform.startswith("linux"):
        cores: set[tuple[str, str]] = set()
        try:
            package = ""
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    package = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    cores.add((package, line.split(":", 1)[1].strip()))
        except OSError:
            cores.clear()
        if cores:
            return max(1, min(len(cores), affinity or len(cores), _MAX_POOL))
    fallback = min(_MAX_THREADS, os.cpu_count() or 1)
    if affinity is not None:
        fallback = min(fallback, affinity)
    return max(1, fallback)


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


class VoiceEmbedder:
    """The extractor (lazy), its worker pool, and per-cluster voices.

    ``model`` overrides the fetched ``assets.SPEAKER_EMBEDDING`` (eval A/Bs)."""

    def __init__(
        self,
        model: Path | None = None,
        *,
        progress: assets.ProgressHook | None = None,
    ) -> None:
        self._model = model
        self._progress = progress
        self._extractor = None
        self._pool: ThreadPoolExecutor | None = None
        self._init_lock = threading.Lock()  # guards every lazy field

    def pool(self) -> ThreadPoolExecutor:
        """The worker pool every embedding call of this embedder runs on.

        One pool, sized once (:func:`_pool_workers`), so concurrent channels
        share one budget and never oversubscribe the machine. Results are
        bit-identical at any worker count because each call is independent and
        the ORT session runs single-threaded (measured 2026-08-09,
        ``eval/README.md``)."""
        if self._pool is None:
            with self._init_lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=_pool_workers(), thread_name_prefix="embed"
                    )
        return self._pool

    rule = VoiceRule()

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

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """L2-normalized voice embedding of a mono 16 kHz float32 slice, or None
        when the slice is empty or the extractor cannot form an embedding."""
        if len(audio) == 0:
            return None
        extractor = self._load()
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return l2_normalize(np.asarray(extractor.compute(stream), dtype=np.float32))

    def _load(self):
        # ONE shared extractor at ONE intra-op thread, used from every pool
        # worker concurrently: ORT's Run is thread-safe, per-call fbank state
        # lives in create_stream(), and sharing was measured bit-exact and as
        # fast as per-worker copies (which cost 26 MB of weights each) at 8
        # workers, 2026-08-09. Intra-op stays 1 even for one-off embed()
        # callers (enroll): parallelism belongs to the pool, and a fixed
        # thread count is what keeps results identical across all paths.
        if self._extractor is None:
            with self._init_lock:
                if self._extractor is None:
                    import sherpa_onnx

                    embedding = self._model or assets.fetch(
                        assets.SPEAKER_EMBEDDING, self._progress
                    )
                    self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                            model=str(embedding), num_threads=1
                        )
                    )
        return self._extractor


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
) -> dict[str, np.ndarray]:
    """A duration-weighted mean voice embedding per cluster of ``turns``.

    :class:`VoiceCollector`'s rule for callers that hold turns rather than
    the diarizer's activity (a solo channel's speech spans, eval's reference
    turns): spans where a *different* cluster is active too are excluded,
    then :func:`_preferred` picks, ``rule`` cuts, and the L2-normalized span
    embeddings are averaged by duration and re-normalized. A cluster with no
    embeddable audio is omitted."""
    audio = to_float32(samples)
    by_cluster: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        by_cluster.setdefault(turn.speaker, []).append(turn)
    overlap = _overlap_spans(by_cluster)

    sums = _Sums(rule or VoiceRule())
    jobs: list[tuple[Hashable, Span]] = []
    for speaker, cluster_turns in by_cluster.items():
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
