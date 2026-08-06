"""Open-set speaker-naming metric: DIR @ FAR and the FAR/FRR curve.

`der.py` scores who-spoke-when; this scores the naming stage — given a gallery
of enrolled speaker profiles, does a diarization cluster get the *right name*,
and does a stranger get *no name*? The conventions are the open-set
identification ones (TST-Bench / VoxBlink2):

- A **trial** is one diarization cluster scored against every enrolled profile.
  A *known* trial's true speaker is in the gallery; an *unknown* trial's is not
  (strangers come from enrolling only a subset of participants).
- Decision rule at threshold t: accept iff the top-scoring profile clears t
  (``score >= t``, matching :meth:`stenograf.voiceprints.ProfileStore.match`),
  and identify as that profile.
- **DIR(t)** — detection & identification rate: fraction of known trials
  accepted *and* correctly named. **FAR(t)**: fraction of unknown trials
  accepted (a stranger got somebody's name). **FRR(t)**: fraction of known
  trials rejected outright (a colleague stayed unnamed; a known trial accepted
  under the *wrong* name counts against DIR but not FRR).
- The headline number is **DIR @ FAR=x%**: the best DIR reachable while holding
  false accepts at or under x% — i.e. the smallest threshold whose FAR is
  within budget. Strangers-must-not-be-named is the product bias, so FAR is
  the constrained axis.

Production matching adds a one-to-one constraint per meeting
(:class:`stenograf.voiceprints.SpeakerReID`); the metric here is per-trial,
the published convention, so numbers stay comparable across our own runs.

The scoring is pure (no audio, no models) and unit-tested; trials come from a
JSON file built by the AMI harness (``ami.py run`` — enroll from session A,
trial clusters from sessions B–D)::

    uv run --group eval eval/reid_score.py            # score eval/out/reid/trials.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

FAR_TARGETS = (0.005, 0.01, 0.05)
"""Headline false-accept budgets: TST-Bench reports at 0.5 % and 1 %; 5 % shows
the curve's permissive end."""


@dataclass(frozen=True)
class Trial:
    """One cluster's scores against the whole gallery.

    ``true_speaker`` is the reference identity behind the cluster; the trial is
    *known* iff that identity is enrolled (a key of ``scores``). ``name`` tags
    the trial for error listings (e.g. ``ES2002b.loop/S1``)."""

    name: str
    true_speaker: str | None
    scores: dict[str, float]

    @property
    def known(self) -> bool:
        return self.true_speaker is not None and self.true_speaker in self.scores

    @property
    def top(self) -> tuple[str, float]:
        """The gallery decision: best-scoring profile and its score."""
        best = max(self.scores.items(), key=lambda kv: kv[1])
        return best


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    dir_: float
    """Known trials accepted under the correct name (detection & identification)."""
    far: float
    """Unknown trials accepted under any name."""
    frr: float
    """Known trials rejected outright."""

    def as_row(self, label: str) -> str:
        return (
            f"| {label} | {self.threshold:.3f} | {self.dir_:.1%} | "
            f"{self.far:.1%} | {self.frr:.1%} |"
        )


def operating_point(trials: list[Trial], threshold: float) -> OperatingPoint:
    """DIR/FAR/FRR of ``trials`` at one accept threshold."""
    known = [t for t in trials if t.known]
    unknown = [t for t in trials if not t.known]
    identified = sum(
        1 for t in known if (top := t.top)[1] >= threshold and top[0] == t.true_speaker
    )
    accepted_strangers = sum(1 for t in unknown if t.top[1] >= threshold)
    rejected = sum(1 for t in known if t.top[1] < threshold)
    return OperatingPoint(
        threshold=threshold,
        dir_=identified / len(known) if known else 0.0,
        far=accepted_strangers / len(unknown) if unknown else 0.0,
        frr=rejected / len(known) if known else 0.0,
    )


def sweep(trials: list[Trial]) -> list[OperatingPoint]:
    """The full curve: one operating point per distinct decision score.

    Thresholds are the observed top-1 scores (the decision statistic — between
    two of them every rate is constant), ascending, plus a just-above-max
    sentinel so the reject-everything end (FAR 0) is always on the curve."""
    tops = sorted({t.top[1] for t in trials})
    if not tops:
        return []
    return [operating_point(trials, t) for t in [*tops, tops[-1] + 1e-6]]


def dir_at_far(trials: list[Trial], far_target: float) -> OperatingPoint | None:
    """The best (lowest-threshold) operating point with FAR ≤ ``far_target``.

    DIR and FAR both fall as the threshold rises, so the smallest threshold
    inside the FAR budget maximizes DIR. ``None`` when there are no trials."""
    for point in sweep(trials):
        if point.far <= far_target:
            return point
    return None


def eer_point(trials: list[Trial]) -> OperatingPoint | None:
    """The operating point where FAR and FRR cross (nearest measured point)."""
    curve = sweep(trials)
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p.far - p.frr))


# -- trials file -----------------------------------------------------------


def load_trials(path: Path) -> list[Trial]:
    data = json.loads(Path(path).read_text())
    return [Trial(t["name"], t.get("true_speaker"), t["scores"]) for t in data["trials"]]


def save_trials(path: Path, trials: list[Trial]) -> None:
    payload = {
        "trials": [
            {"name": t.name, "true_speaker": t.true_speaker, "scores": t.scores} for t in trials
        ]
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


# -- report ----------------------------------------------------------------


def report(trials: list[Trial]) -> str:
    known = sum(1 for t in trials if t.known)
    lines = [
        "### Speaker naming (open-set identification)",
        "",
        f"{len(trials)} trials: {known} known, {len(trials) - known} unknown.",
        "",
        "| Operating point | Threshold | DIR | FAR | FRR |",
        "|---|---|---|---|---|",
    ]
    for target in FAR_TARGETS:
        point = dir_at_far(trials, target)
        label = f"DIR @ FAR≤{target:.1%}"
        lines.append(point.as_row(label) if point else f"| {label} | — | — | — | — |")
    if eer := eer_point(trials):
        lines.append(eer.as_row("FAR≈FRR"))

    curve = sweep(trials)
    step = max(1, len(curve) // 20)  # the full curve, downsampled to ~20 rows
    lines += [
        "",
        "| Threshold | DIR | FAR | FRR |",
        "|---|---|---|---|",
        *(
            f"| {p.threshold:.3f} | {p.dir_:.1%} | {p.far:.1%} | {p.frr:.1%} |"
            for p in curve[::step]
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    from common import OUT_DIR

    trials_path = OUT_DIR / "reid" / "trials.json"
    if not trials_path.exists():
        print(f"no {trials_path} — run `eval/ami.py run` first", file=sys.stderr)
        return 1
    trials = load_trials(trials_path)
    if not trials:
        print("trials file is empty", file=sys.stderr)
        return 1
    text = report(trials)
    print(text)
    out = OUT_DIR / "reid-report.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
