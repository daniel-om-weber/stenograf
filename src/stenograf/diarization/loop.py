"""Our own diarization loop: segmentation-3.0 via onnxruntime + our clustering.

Replaces sherpa's ``OfflineSpeakerDiarization.process()`` monolith with the
same three stages under our control — powerset segmentation, per-(chunk,
speaker) embeddings, agglomerative clustering — so clustering and stride
become changeable. The reference semantics are sherpa's own implementation,
extracted line-by-line into ``eval/diarization-loop-spec.md`` (2026-08-07);
every constant and edge case below cites that spec, and the parity gate is
``eval/loop_parity.py``. The embedding extractor and all re-ID plumbing are
inherited from :class:`SherpaOnnxDiarizer` — one embedding model, one code
path, whichever loop produced the turns.

Two reference behaviors kept deliberately, because parity is the gate:

- **Two frame→sample scales.** Embedding spans use ``WINDOW / FRAMES``
  (≈271.6 samples/frame) while output times use ``RF_SHIFT`` (270) plus half
  the receptive field — up to ~61 ms apart at a chunk's end (spec §1.5).
- **Overlap is excluded from embeddings but re-enters the output** via the
  per-frame top-k vote (spec §1.4/§1.7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stenograf import assets
from stenograf.audio import SAMPLE_RATE, to_float32
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.sherpa import SherpaOnnxDiarizer, _num_threads

WINDOW = 160_000
"""Segmentation window, samples (10 s; ONNX metadata ``window_size``)."""

SHIFT = 16_000
"""Window stride, samples (sherpa hardcodes 0.1 × window in 1.13.4)."""

FRAMES = 589
"""Model output frames per window."""

RF_SHIFT = 270
"""Output frame step, samples (0.016875 s)."""

RF_SIZE = 991
"""Receptive field, samples — half of it offsets every output time."""

MIN_EMBED_FRAMES = 10
"""A (chunk, speaker) pair needs this many clean active frames (≈0.17 s,
summed, post overlap-exclusion) to contribute a clustering embedding."""

MIN_DURATION_ON = 0.3
"""Segments must be strictly longer than this to survive (sherpa default)."""

MIN_DURATION_OFF = 0.5
"""Same-speaker gaps at or under this merge (sherpa default)."""

_POWERSET = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ],
    dtype=bool,
)
"""The 7-class powerset rows in pyannote's order (cardinality, then
lexicographic) — argmax index → active local speakers (spec §1.2)."""


class OwnDiarizer(SherpaOnnxDiarizer):
    """Drop-in for :class:`SherpaOnnxDiarizer` with the loop under our control.

    ``diarize`` is reimplemented; embeddings, re-ID, and every consumer of the
    ``Diarizer`` interface are inherited unchanged.
    """

    def __init__(
        self,
        segmentation_model: Path | None = None,
        embedding_model: Path | None = None,
        *,
        clustering_threshold: float = 0.5,
        cluster_method: str = "ward",
        shift: int = SHIFT,
        progress: assets.ProgressHook | None = None,
    ) -> None:
        super().__init__(
            segmentation_model,
            embedding_model,
            clustering_threshold=clustering_threshold,
            progress=progress,
        )
        self._cluster_method = cluster_method
        self._shift = shift  # window stride, samples — THE knob owning the loop unlocks
        self._session = None  # lazy onnxruntime session for segmentation

    # ---- segmentation ------------------------------------------------------

    def _segmentation(self):
        if self._session is None:
            import onnxruntime

            model = self._segmentation_model or assets.fetch(
                assets.PYANNOTE_SEGMENTATION, self._progress
            )
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = _num_threads()
            self._session = onnxruntime.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def _chunk_labels(self, audio: np.ndarray) -> list[np.ndarray]:
        """Per-chunk boolean [FRAMES, 3] active-speaker matrices (powerset
        argmax). Chunking mirrors the reference exactly: one zero-padded chunk
        for short audio, stride SHIFT, and a zero-padded tail chunk whenever
        samples remain (spec §1.1)."""
        session = self._segmentation()
        n = len(audio)

        def run(chunk: np.ndarray) -> np.ndarray:
            (y,) = session.run(None, {"x": chunk.reshape(1, 1, -1)})
            logits = np.asarray(y)[0]  # ort returns ndarray; asarray narrows the type
            return _POWERSET[np.argmax(logits, axis=-1)]

        if n <= WINDOW:
            padded = np.zeros(WINDOW, dtype=np.float32)
            padded[:n] = audio
            return [run(padded)]
        shift = self._shift
        num_chunks = (n - WINDOW) // shift + 1
        labels = [run(audio[i * shift : i * shift + WINDOW]) for i in range(num_chunks)]
        if (n - WINDOW) % shift > 0:
            padded = np.zeros(WINDOW, dtype=np.float32)
            tail = audio[num_chunks * shift :]
            padded[: len(tail)] = tail
            labels.append(run(padded))
        return labels

    # ---- the loop ----------------------------------------------------------

    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        audio = to_float32(samples)
        n = len(audio)
        labels = self._chunk_labels(audio)

        if len(labels) == 1:
            # Short audio: the three local powerset speakers ARE the output —
            # no clustering, no embeddings (spec §1.8).
            active = labels[0][: n // RF_SHIFT + 1]
            return _to_turns(active.astype(np.int64), np.sum(active, axis=1))

        pairs, vectors = self._pair_embeddings(audio, labels)
        if not pairs:
            return []
        clusters = _cluster(
            np.stack(vectors), self._threshold, num_speakers, self._cluster_method
        )
        return self._assemble(n, labels, pairs, clusters)

    def _assemble(
        self,
        n: int,
        labels: list[np.ndarray],
        pairs: list[tuple[int, int]],
        clusters: np.ndarray,
    ) -> list[SpeakerTurn]:
        """Clustered (chunk, speaker) pairs → output turns (votes, top-k,
        times). Split from :meth:`diarize` so eval can freeze a channel's
        segmentation + embeddings once and swap partitioners cheaply."""
        num_frames = (WINDOW + (len(labels) - 1) * self._shift) // RF_SHIFT + 1
        starts = [int(i * self._shift / RF_SHIFT + 0.5) for i in range(len(labels))]

        # How many speakers each global frame has: per-chunk counts averaged
        # over covering chunks, round-half-up (spec §1.3). Overlap-inclusive.
        count = np.zeros(num_frames, dtype=np.float64)
        weight = np.zeros(num_frames, dtype=np.float64)
        for start, chunk in zip(starts, labels, strict=True):
            count[start : start + FRAMES] += np.sum(chunk, axis=1)
            weight[start : start + FRAMES] += 1.0
        speakers_per_frame = (count / (weight + 1e-12) + 0.5).astype(np.int64)

        # Vote accumulation on the global grid: each clustered (chunk, local
        # speaker)'s active frames vote for its cluster (spec §1.7).
        num_clusters = int(clusters.max()) + 1
        votes = np.zeros((num_frames, num_clusters), dtype=np.int64)
        by_chunk: dict[int, list[tuple[int, int]]] = {}
        for (chunk_index, speaker), cluster in zip(pairs, clusters, strict=True):
            by_chunk.setdefault(chunk_index, []).append((speaker, int(cluster)))
        for chunk_index, assigned in by_chunk.items():
            chunk = labels[chunk_index]
            start = starts[chunk_index]
            for speaker, cluster in assigned:
                votes[start : start + FRAMES, cluster] += chunk[:, speaker]

        if (n - WINDOW) % self._shift > 0:  # padded tail chunk: truncate its padding
            votes = votes[: n // RF_SHIFT + 1]
            speakers_per_frame = speakers_per_frame[: n // RF_SHIFT + 1]

        # Per frame, the top-k voted clusters are active, k = speaker count.
        final = np.zeros_like(votes)
        for f in range(len(votes)):
            k = min(int(speakers_per_frame[f]), num_clusters)
            if k <= 0:
                continue
            top = np.argpartition(-votes[f], k - 1)[:k]
            final[f, np.sort(top)] = 1
        return _to_turns(final, np.sum(final, axis=1))

    def _pair_embeddings(
        self, audio: np.ndarray, labels: list[np.ndarray]
    ) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
        """One embedding per (chunk, local speaker): overlap frames zeroed,
        pairs under MIN_EMBED_FRAMES clean frames skipped, contiguous runs
        sliced on the WINDOW/FRAMES scale and embedded as one concatenated
        stream (spec §1.4)."""
        n = len(audio)
        pairs: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        for chunk_index, chunk in enumerate(labels):
            clean = chunk.copy()
            clean[np.sum(chunk, axis=1) >= 2] = False
            offset = chunk_index * self._shift
            for speaker in range(clean.shape[1]):
                column = clean[:, speaker]
                if int(column.sum()) < MIN_EMBED_FRAMES:
                    continue
                slices = []
                for run_start, run_end in _runs(column):
                    start = int(run_start / FRAMES * WINDOW) + offset
                    if run_end >= FRAMES:
                        end = int((FRAMES - 1) / FRAMES * WINDOW) + offset
                    else:
                        end = int(run_end / FRAMES * WINDOW) + offset
                    if start < n:
                        slices.append(audio[start : min(end, n)])
                if not slices:
                    continue
                vector = self.embed(np.concatenate(slices))
                # embed() can hand back the zero vector on degenerate audio
                # (l2_normalize passes it through), and an extractor NaN/inf
                # must not reach the affinity math; the reference drops NaN
                # embeddings too.
                if vector is not None and np.all(np.isfinite(vector)) and np.any(vector):
                    pairs.append((chunk_index, speaker))
                    vectors.append(vector)
        return pairs, vectors


def _cluster(
    vectors: np.ndarray,
    threshold: float,
    num_speakers: int | None,
    method: str = "complete",
) -> np.ndarray:
    """Cluster (chunk, speaker) embeddings into global speakers.

    ``method`` picks the known-count partitioner: ``complete``/``average`` are
    AHC linkages on cosine distance (complete is the reference's choice,
    average is BUT VBx's AHC-init choice), ``centroid``/``ward`` are geometric
    linkages on Euclidean over unit vectors (pyannote 3.1 ships centroid), and
    ``nmesc`` is spectral clustering on the NME-binarized affinity
    (:func:`_nmesc`). Ward is the shipped known-count default (2026-08-07:
    mean loop DER 16.5 → 13.2 % with the gated fold, worst per-channel
    regression +0.5 pt across 40 channels — ``eval/README.md``).
    Threshold mode (no count) always uses the reference's
    complete-linkage cut — first merge height ≥ ``threshold``, a cosine
    DISTANCE (spec §1.6). Threshold mode is REACHABLE in production — an
    install without stenodiar (manylinux_2_28, source builds without Rust)
    estimates counts right here — so a partitioner decision covers both
    branches; they must not quietly fork (2026-08-07 review)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-10)  # zero rows stay zero (Eigen semantics)
    if num_speakers is not None and num_speakers > 0 and method == "nmesc":
        return _nmesc(normalized, min(num_speakers, n))
    linkage_method = method if num_speakers is not None and num_speakers > 0 else "complete"
    if linkage_method in ("centroid", "ward"):
        # Geometric linkages need raw observations. On unit vectors Euclidean
        # DISTANCE is monotone in cosine distance (pyannote 3.1's own
        # configuration); the merge criteria are still their own — ward's adds
        # a cluster-size penalty with no cosine analogue.
        tree = linkage(normalized.astype(np.float64), method=linkage_method, metric="euclidean")
    else:
        # Inputs are gated finite; macOS Accelerate raises spurious FP flags
        # on every clean float32 GEMM (measured 2026-08-07: all three flags on
        # provably-clean unit vectors), so the flags mean nothing here.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            distance = np.clip(1.0 - normalized @ normalized.T, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method=linkage_method)
    if num_speakers is not None and num_speakers > 0:
        flat = fcluster(tree, t=min(num_speakers, n), criterion="maxclust")
    else:
        heights = tree[:, 2]
        stop = next((i for i, h in enumerate(heights) if h >= threshold), n - 1)
        flat = fcluster(tree, t=n - stop, criterion="maxclust")
    # 0-based, first-appearance order (labels are arbitrary either way).
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(c), len(remap)) for c in flat], dtype=np.int64)


_NMESC_P_RATIOS = (0.01, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25)
"""Neighbor-fraction sweep for the NME criterion (Park et al. 2020's search
space, coarsened to eight points — the criterion is smooth in p)."""


def _nmesc(normalized: np.ndarray, k: int) -> np.ndarray:
    """Spectral clustering at known ``k`` with NME-tuned p-binarization.

    Park et al., *Auto-tuning spectral clustering for speaker diarization
    using normalized maximum eigengap* (IEEE SPL 27, 2020; NeMo's NMESC is
    the Apache-2.0 reference): keep each row's top-p cosine affinities, zero
    the rest, symmetrize, and pick p by the eigengap at k normalized by the
    neighbor fraction; then embed rows in the top-k eigenvectors of the
    normalized affinity and k-means them (seeded — deterministic)."""
    from scipy.cluster.vq import kmeans2
    from scipy.linalg import eigh

    n = len(normalized)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        affinity = normalized @ normalized.T  # gated-finite inputs; Accelerate flag noise
    best: tuple[float, np.ndarray] | None = None
    for ratio in _NMESC_P_RATIOS:
        p = min(n - 1, max(2, round(ratio * n)))
        keep = np.argpartition(-affinity, p, axis=1)[:, : p + 1]  # self survives
        pruned = np.zeros_like(affinity)
        np.put_along_axis(pruned, keep, np.take_along_axis(affinity, keep, axis=1), axis=1)
        w = 0.5 * (pruned + pruned.T)
        degree = np.maximum(w.sum(axis=1), 1e-10)
        scale = 1.0 / np.sqrt(degree)
        p_matrix = scale[:, None] * w * scale[None, :]
        values, vectors = eigh(
            p_matrix, subset_by_index=(n - k - 1, n - 1)
        )  # ascending; top k+1 of the affinity spectrum
        gap = float(values[1] - values[0])  # λ_k − λ_{k+1} in descending terms
        score = gap / (p / n)
        if best is None or score > best[0]:
            best = (score, vectors[:, 1:])  # top-k eigenvectors
    assert best is not None
    embedding = best[1]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-10)
    # seed is real (scipy ≥1.7 keyword-only); the bundled stub predates it.
    _, labels = kmeans2(embedding, k, minit="++", seed=0)  # pyright: ignore[reportCallIssue]
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(c), len(remap)) for c in labels], dtype=np.int64)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open [start, end) frame indexes."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _to_turns(active: np.ndarray, per_frame: np.ndarray) -> list[SpeakerTurn]:
    """Frame matrix → merged, minimum-length turns on the output time grid
    (spec §1.7): times = frame × RF_SHIFT + RF_SIZE/2, same-speaker gaps ≤
    MIN_DURATION_OFF merged, then strictly-longer-than MIN_DURATION_ON kept."""
    scale = RF_SHIFT / SAMPLE_RATE
    offset = 0.5 * RF_SIZE / SAMPLE_RATE
    turns: list[SpeakerTurn] = []
    for cluster in range(active.shape[1]):
        spans = [
            (start * scale + offset, end * scale + offset)
            for start, end in _runs(active[:, cluster].astype(bool))
        ]
        merged: list[tuple[float, float]] = []
        for start, end in spans:
            if merged and start - merged[-1][1] <= MIN_DURATION_OFF:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        turns += [
            SpeakerTurn(speaker=f"S{cluster}", start=start, end=end)
            for start, end in merged
            if end - start > MIN_DURATION_ON
        ]
    turns.sort(key=lambda t: (t.start, t.speaker))
    return turns
