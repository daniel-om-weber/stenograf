"""Unit tests for the open-set naming scorer (eval/reid_score.py).

DIR @ FAR gates every re-ID change (threshold, profile store, embedding model),
so its math is pinned here against hand-computable trials. Like the DER scorer,
the eval harness is standalone tooling: its directory goes on the path and the
flat module is imported directly — the scoring is pure, no audio or models.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from reid_score import (  # noqa: E402
    Trial,
    dir_at_far,
    eer_point,
    load_trials,
    operating_point,
    save_trials,
    sweep,
)


def known(name: str, true: str, **scores: float) -> Trial:
    return Trial(name, true, scores)


def stranger(name: str, **scores: float) -> Trial:
    return Trial(name, None, scores)


class TestOperatingPoint:
    def test_perfect_separation(self):
        trials = [
            known("k1", "A", A=0.9, B=0.1),
            known("k2", "B", A=0.2, B=0.8),
            stranger("u1", A=0.3, B=0.4),
        ]
        point = operating_point(trials, 0.8)
        assert point.dir_ == 1.0
        assert point.far == 0.0
        assert point.frr == 0.0

    def test_low_threshold_accepts_the_stranger(self):
        trials = [known("k1", "A", A=0.9, B=0.1), stranger("u1", A=0.5, B=0.1)]
        point = operating_point(trials, 0.4)
        assert point.dir_ == 1.0
        assert point.far == 1.0

    def test_wrong_name_hits_dir_but_not_frr(self):
        # The known trial is accepted — under the wrong profile. That is an
        # identification miss (DIR 0) but not a rejection (FRR 0).
        trials = [known("k1", "A", A=0.2, B=0.9)]
        point = operating_point(trials, 0.5)
        assert point.dir_ == 0.0
        assert point.frr == 0.0
        assert operating_point(trials, 0.95).frr == 1.0

    def test_true_speaker_outside_gallery_is_an_unknown_trial(self):
        # Subset enrollment: C exists in the references but not the gallery.
        trial = known("k1", "C", A=0.6, B=0.2)
        assert not trial.known
        point = operating_point([trial], 0.5)
        assert point.far == 1.0
        assert point.dir_ == 0.0  # no known trials; defined as 0


class TestSweep:
    def test_curve_is_monotone_in_threshold(self):
        trials = [
            known("k1", "A", A=0.9, B=0.1),
            known("k2", "B", A=0.2, B=0.7),
            known("k3", "A", A=0.4, B=0.5),
            stranger("u1", A=0.6, B=0.1),
            stranger("u2", A=0.1, B=0.2),
        ]
        curve = sweep(trials)
        thresholds = [p.threshold for p in curve]
        assert thresholds == sorted(thresholds)
        for lo, hi in zip(curve, curve[1:], strict=False):
            assert hi.dir_ <= lo.dir_
            assert hi.far <= lo.far
            assert hi.frr >= lo.frr

    def test_sentinel_reaches_reject_everything(self):
        trials = [known("k1", "A", A=0.9), stranger("u1", A=0.8)]
        end = sweep(trials)[-1]
        assert end.far == 0.0
        assert end.frr == 1.0

    def test_empty(self):
        assert sweep([]) == []
        assert dir_at_far([], 0.01) is None
        assert eer_point([]) is None


class TestDirAtFar:
    def test_far_budget_forces_the_threshold_up(self):
        # The stranger scores 0.5; holding FAR at 0 must lift the threshold
        # past it, which sacrifices the weak known trial (0.45).
        trials = [
            known("k1", "A", A=0.9, B=0.1),
            known("k2", "B", A=0.1, B=0.45),
            stranger("u1", A=0.5, B=0.2),
        ]
        point = dir_at_far(trials, 0.01)
        assert point is not None
        assert point.far == 0.0
        assert point.dir_ == pytest.approx(0.5)
        assert point.threshold > 0.5

    def test_loose_budget_keeps_the_low_threshold(self):
        trials = [
            known("k1", "A", A=0.9, B=0.1),
            known("k2", "B", A=0.1, B=0.45),
            stranger("u1", A=0.5, B=0.2),
        ]
        point = dir_at_far(trials, 1.0)
        assert point is not None
        assert point.dir_ == 1.0


class TestEer:
    def test_crossing_is_found(self):
        trials = [
            known("k1", "A", A=0.9, B=0.1),
            known("k2", "B", A=0.1, B=0.3),
            stranger("u1", A=0.5, B=0.2),
            stranger("u2", A=0.2, B=0.1),
        ]
        point = eer_point(trials)
        assert point is not None
        assert abs(point.far - point.frr) <= 0.5


class TestTrialsFile:
    def test_roundtrip(self, tmp_path):
        trials = [known("k1", "A", A=0.9, B=0.1), stranger("u1", A=0.3, B=0.4)]
        path = tmp_path / "trials.json"
        save_trials(path, trials)
        assert load_trials(path) == trials
