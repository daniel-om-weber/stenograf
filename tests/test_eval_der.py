"""Unit tests for the diarization scorer (eval/der.py, eval/rttm.py) — Task 0d.

The scorer is the gating measurement for everything speaker-centric, so its math
is pinned here against hand-computable cases. The eval harness is standalone
tooling (not a package), so we put its directory on the path and import the flat
modules directly; the scorer is pure (numpy + scipy), needing no audio or models.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import rttm  # noqa: E402
from der import Word, score_attribution, score_der  # noqa: E402
from rttm import Turn  # noqa: E402


def turns(*spans: tuple[str, float, float]) -> list[Turn]:
    return [Turn(spk, s, e) for spk, s, e in spans]


class TestDer:
    def test_identical_is_zero(self):
        ref = turns(("A", 0, 5), ("B", 5, 10), ("A", 10, 12))
        assert score_der(ref, ref, collar=0.0).der == 0.0

    def test_label_names_do_not_matter(self):
        # Perfect segmentation, different label strings → optimal mapping → DER 0.
        ref = turns(("A", 0, 5), ("B", 5, 10))
        hyp = turns(("spk1", 0, 5), ("spk0", 5, 10))
        score = score_der(ref, hyp, collar=0.0)
        assert score.der == 0.0
        assert score.mapping == {"A": "spk1", "B": "spk0"}

    def test_all_missed(self):
        score = score_der(turns(("A", 0, 10)), [], collar=0.0)
        assert score.der == pytest.approx(1.0)
        assert score.missed == pytest.approx(1.0)
        assert score.false_alarm == 0.0 and score.confusion == 0.0

    def test_false_alarm(self):
        # Hyp speaks 5 s past the reference → 50% false alarm on 10 s of ref.
        ref = turns(("A", 0, 10))
        hyp = turns(("X", 0, 15))
        score = score_der(ref, hyp, collar=0.0)
        assert score.false_alarm == pytest.approx(0.5)
        assert score.missed == 0.0 and score.confusion == 0.0
        assert score.der == pytest.approx(0.5)

    def test_confusion_from_a_split_speaker(self):
        # One ref speaker, two hyp speakers over the halves: half is confused.
        ref = turns(("A", 0, 10))
        hyp = turns(("Y", 0, 5), ("Z", 5, 10))
        score = score_der(ref, hyp, collar=0.0)
        assert score.confusion == pytest.approx(0.5)
        assert score.missed == 0.0 and score.false_alarm == 0.0

    def test_collar_excludes_boundary_errors(self):
        # Hyp is 0.1 s short at each boundary. A 0.25 s collar swallows the error;
        # without it, 0.2 s of 10 s reads as 2% missed.
        ref = turns(("A", 0, 10))
        hyp = turns(("X", 0.1, 9.9))
        assert score_der(ref, hyp, collar=0.25).der == pytest.approx(0.0, abs=1e-9)
        assert score_der(ref, hyp, collar=0.0).der == pytest.approx(0.02, abs=2e-3)


class TestAttribution:
    def _words(self, *spans: tuple[str, float, float, str]) -> list[Word]:
        return [Word(t, s, e, spk) for t, s, e, spk in spans]

    def test_all_correct(self):
        words = self._words(("a", 0, 1, "H1"), ("b", 1, 2, "H1"), ("c", 2, 3, "H2"))
        ref = turns(("A", 0, 2), ("B", 2, 3))
        score = score_attribution(words, ref)
        assert score.accuracy == 1.0
        assert (score.correct, score.scored, score.total) == (3, 3, 3)

    def test_one_misattributed(self):
        # b truly belongs to A but is labelled with H2 (which maps to B).
        words = self._words(("a", 0, 1, "H1"), ("b", 1, 2, "H2"), ("c", 2, 3, "H2"))
        ref = turns(("A", 0, 2), ("B", 2, 3))
        score = score_attribution(words, ref)
        assert score.accuracy == pytest.approx(2 / 3)
        assert score.correct == 2 and score.scored == 3

    def test_word_over_no_reference_is_unscored(self):
        words = self._words(("x", 5, 6, "H1"))
        ref = turns(("A", 0, 2))
        score = score_attribution(words, ref)
        assert score.scored == 0 and score.accuracy == 0.0 and score.total == 1


class TestRttmRoundtrip:
    def test_write_then_parse(self, tmp_path):
        original = turns(("Local-1", 0.5, 2.25), ("Remote-1", 2.25, 4.0))
        path = tmp_path / "seg.rttm"
        rttm.write_rttm(path, original, "seg")
        assert rttm.parse_rttm(path) == original

    def test_ignores_comments_and_zero_duration(self, tmp_path):
        path = tmp_path / "seg.rttm"
        path.write_text(
            ";; a comment\n"
            "SPEAKER seg 1 1.000 2.000 <NA> <NA> A <NA> <NA>\n"
            "SPEAKER seg 1 3.000 0.000 <NA> <NA> B <NA> <NA>\n"  # zero duration dropped
        )
        parsed = rttm.parse_rttm(path)
        assert parsed == [Turn("A", 1.0, 3.0)]

    def test_malformed_line_raises(self, tmp_path):
        path = tmp_path / "seg.rttm"
        path.write_text("SPEAKER seg 1 1.0\n")
        with pytest.raises(ValueError):
            rttm.parse_rttm(path)
