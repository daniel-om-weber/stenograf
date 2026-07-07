"""Diarization Error Rate + word-attribution scoring for stenograf's diarizer.

Phase 3, Task 0d (PLAN.md §5): the gating measurement. Nothing speaker-centric —
re-ID threshold tuning, diarization upgrades — is measurable without a
speaker-labeled reference and a scorer, so this is the long pole.

Two metrics, both computed here as pure functions (numpy + scipy only, no audio,
no models, no stenograf import) so they are unit-testable in isolation:

- **DER** — the standard frame-based Diarization Error Rate (NIST md-eval style):
  discretize time, find the reference↔hypothesis speaker mapping that maximizes
  agreement (Hungarian assignment), then sum missed speech + false alarm +
  speaker confusion over a ±``collar`` no-score zone around reference
  boundaries. DER = error_time / total_reference_speech_time. Overlap is scored
  natively (a frame may have several active speakers) unless ``ignore_overlap``.
- **Word attribution** — of the finalized transcript's words, the fraction placed
  on the correct speaker, under the label mapping that maximizes agreement. This
  is what the user actually reads, and it is what re-ID must not regress.

CLI (scores ``eval/refs/<id>.rttm`` against ``eval/out/diar/<id>.rttm`` and, when
present, ``eval/out/diar/<id>.words.json``)::

    uv run --group eval eval/der.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from rttm import Turn, parse_rttm, speakers
from scipy.optimize import linear_sum_assignment

FRAME_S = 0.01
COLLAR = 0.25  # NIST md-eval default no-score collar around reference boundaries


@dataclass(frozen=True)
class DiarizationScore:
    der: float
    """Total error as a fraction of scored reference speech (missed+fa+confusion)."""
    missed: float
    false_alarm: float
    confusion: float
    total_ref_s: float
    mapping: dict[str, str]
    """Reference speaker → the hypothesis speaker it was optimally matched to."""

    def as_row(self, name: str) -> str:
        return (
            f"| {name} | {self.der:.1%} | {self.missed:.1%} | "
            f"{self.false_alarm:.1%} | {self.confusion:.1%} | {self.total_ref_s:.0f}s |"
        )


def _activity(turns: list[Turn], index: dict[str, int], n_frames: int) -> np.ndarray:
    """Boolean [n_speakers, n_frames] speech-activity matrix."""
    mat = np.zeros((len(index), n_frames), dtype=bool)
    for t in turns:
        lo = max(0, round(t.start / FRAME_S))
        hi = min(n_frames, round(t.end / FRAME_S))
        if hi > lo:
            mat[index[t.speaker], lo:hi] = True
    return mat


def _scored_mask(ref: list[Turn], n_frames: int, collar: float) -> np.ndarray:
    """Frames kept for scoring: everything except a ±collar zone around every
    reference boundary (start and end)."""
    scored = np.ones(n_frames, dtype=bool)
    half = round(collar / FRAME_S)
    if half == 0:
        return scored
    for t in ref:
        for boundary in (round(t.start / FRAME_S), round(t.end / FRAME_S)):
            scored[max(0, boundary - half) : boundary + half] = False
    return scored


def score_der(
    ref: list[Turn],
    hyp: list[Turn],
    *,
    collar: float = COLLAR,
    ignore_overlap: bool = False,
) -> DiarizationScore:
    """Frame-based DER of ``hyp`` against ``ref`` with optimal speaker mapping."""
    ref_spk = speakers(ref)
    hyp_spk = speakers(hyp)
    end = max([t.end for t in ref + hyp], default=0.0)
    n_frames = round(end / FRAME_S) + 1

    ref_idx = {s: i for i, s in enumerate(ref_spk)}
    hyp_idx = {s: i for i, s in enumerate(hyp_spk)}
    R = _activity(ref, ref_idx, n_frames)
    H = _activity(hyp, hyp_idx, n_frames)

    scored = _scored_mask(ref, n_frames, collar)
    if ignore_overlap:
        scored &= R.sum(axis=0) <= 1
    R = R[:, scored]
    H = H[:, scored]

    # Optimal reference→hypothesis mapping: maximize total co-active frames.
    overlap = R.astype(np.int64) @ H.T.astype(np.int64)  # [n_ref, n_hyp]
    mapping: dict[str, str] = {}
    correct = np.zeros(R.shape[1], dtype=np.int64)
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        for i, j in zip(rows, cols, strict=False):
            mapping[ref_spk[i]] = hyp_spk[j]
            if overlap[i, j]:
                correct += R[i] & H[j]

    n_ref = R.sum(axis=0)
    n_hyp = H.sum(axis=0)
    total_ref = int(n_ref.sum())
    missed = int(np.maximum(n_ref - n_hyp, 0).sum())
    false_alarm = int(np.maximum(n_hyp - n_ref, 0).sum())
    confusion = int((np.minimum(n_ref, n_hyp) - correct).sum())

    denom = total_ref or 1  # avoid /0; with no ref speech every rate is 0
    return DiarizationScore(
        der=(missed + false_alarm + confusion) / denom,
        missed=missed / denom,
        false_alarm=false_alarm / denom,
        confusion=confusion / denom,
        total_ref_s=total_ref * FRAME_S,
        mapping=mapping,
    )


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AttributionScore:
    accuracy: float
    """Fraction of reference-covered words placed on the correct speaker."""
    correct: int
    scored: int
    """Words that fall on some reference speaker (the denominator)."""
    total: int
    mapping: dict[str, str]
    """Hypothesis word-label → reference speaker, chosen to maximize agreement."""

    def as_row(self, name: str) -> str:
        return f"| {name} | {self.accuracy:.1%} | {self.correct}/{self.scored} | {self.total} |"


def _ref_speaker(word: Word, ref: list[Turn]) -> str | None:
    """The reference speaker overlapping ``word`` the most, or None if none does."""
    best, best_overlap = None, 0.0
    for t in ref:
        overlap = min(word.end, t.end) - max(word.start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    return best


def score_attribution(words: list[Word], ref: list[Turn]) -> AttributionScore:
    """Fraction of words attributed to the right speaker, under the best mapping.

    Each word's true speaker is the reference turn it overlaps most; words over no
    reference speech are unscored. The hypothesis's per-word speaker labels are
    mapped to reference speakers by the assignment that maximizes correct words
    (Hungarian), so the metric does not depend on arbitrary label names."""
    hyp_labels = sorted({w.speaker for w in words})
    ref_labels = speakers(ref)
    hyp_idx = {s: i for i, s in enumerate(hyp_labels)}
    ref_idx = {s: i for i, s in enumerate(ref_labels)}

    counts = np.zeros((len(hyp_labels), len(ref_labels)), dtype=np.int64)
    scored = 0
    for w in words:
        truth = _ref_speaker(w, ref)
        if truth is None:
            continue
        scored += 1
        counts[hyp_idx[w.speaker], ref_idx[truth]] += 1

    mapping: dict[str, str] = {}
    correct = 0
    if counts.size and scored:
        rows, cols = linear_sum_assignment(-counts)
        for i, j in zip(rows, cols, strict=False):
            mapping[hyp_labels[i]] = ref_labels[j]
            correct += int(counts[i, j])

    return AttributionScore(
        accuracy=correct / scored if scored else 0.0,
        correct=correct,
        scored=scored,
        total=len(words),
        mapping=mapping,
    )


# -- CLI -------------------------------------------------------------------


def _load_words(path) -> list[Word]:
    record = json.loads(path.read_text())
    return [Word(w["text"], w["start"], w["end"], w["speaker"]) for w in record["words"]]


def main() -> int:
    from common import OUT_DIR, REFS_DIR

    diar_dir = OUT_DIR / "diar"
    refs = sorted(p for p in REFS_DIR.glob("*.rttm") if not p.name.endswith(".draft.rttm"))
    if not refs:
        print(
            "no reference RTTMs in eval/refs/*.rttm yet — hand-label speaker turns "
            "first (see eval/README.md 'Diarization scoring'); bootstrap a draft with "
            "`eval/diarize.py --bootstrap`",
            file=sys.stderr,
        )
        return 1

    der_rows, attr_rows = [], []
    for ref_path in refs:
        seg = ref_path.stem
        ref = parse_rttm(ref_path)
        hyp_path = diar_dir / f"{seg}.rttm"
        if hyp_path.exists():
            der_rows.append(score_der(ref, parse_rttm(hyp_path)).as_row(seg))
        words_path = diar_dir / f"{seg}.words.json"
        if words_path.exists():
            attr_rows.append(score_attribution(_load_words(words_path), ref).as_row(seg))

    if not der_rows and not attr_rows:
        print("references exist but no hypotheses — run eval/diarize.py first", file=sys.stderr)
        return 1

    lines: list[str] = []
    if der_rows:
        lines += [
            "### Diarization Error Rate",
            "",
            "| Segment | DER | Missed | False alarm | Confusion | Ref speech |",
            "|---|---|---|---|---|---|",
            *der_rows,
            "",
        ]
    if attr_rows:
        lines += [
            "### Word attribution",
            "",
            "| Segment | Accuracy | Correct | Words |",
            "|---|---|---|---|",
            *attr_rows,
            "",
        ]
    report = "\n".join(lines)
    print(report)
    diar_dir.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diar-report.md").write_text(report + "\n")
    print(f"wrote {OUT_DIR / 'diar-report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
