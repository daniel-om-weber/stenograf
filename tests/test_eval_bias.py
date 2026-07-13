"""Unit tests for the contextual-biasing scorer (eval/bias_score.py).

The scorer decides whether biasing ships and with which defaults, so it is pinned
twice: against hand-computable cases, and — the part that matters — against a
*published* oracle. ``facebookresearch/fbai-speech/is21_deep_bias`` ships 44
hypothesis files together with the result files its own scorer produced for the
INTERSPEECH 2021 paper. Our scorer reads their hypotheses and must land on their
numbers to the digit; if it cannot, no number this harness prints downstream means
anything.

The oracle needs their data, which ``eval/bias_data.py --fetch is21`` caches; those
tests skip when it is absent, so the default suite stays offline. Everything above
the oracle class runs always. The eval harness is standalone tooling, so we put its
directory on the path and import the flat module, as ``tests/test_eval_der.py`` does.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import bias_score  # noqa: E402
from bias_score import Code, RefUtt, align, score  # noqa: E402


def normalize(word: str) -> str:
    """A stand-in for eval/score.py's normalizer: lowercase, punctuation out.

    Copied rather than imported so these tests stay pure (score.py imports jiwer);
    the properties under test are "it lowercases" and "it can split or drop a
    word", which is all the scorer's provenance logic cares about.
    """
    return re.sub(r"[^\w\s]", " ", word.lower()).strip()


def refs(*utts: tuple[str, str, list[str], list[str]]) -> dict[str, RefUtt]:
    return {
        uttid: RefUtt(uttid, text, tuple(rare), tuple(biasing))
        for uttid, text, rare, biasing in utts
    }


class TestAlign:
    def test_identical_is_all_matches(self):
        steps = align(["a", "b"], ["a", "b"])
        assert [code for code, _, _ in steps] == [Code.match, Code.match]

    def test_substitution(self):
        steps = align(["a", "b"], ["a", "x"])
        assert [code for code, _, _ in steps] == [Code.match, Code.substitution]

    def test_insertion_and_deletion(self):
        assert [c for c, _, _ in align(["a"], ["a", "b"])] == [Code.match, Code.insertion]
        assert [c for c, _, _ in align(["a", "b"], ["a"])] == [Code.match, Code.deletion]

    def test_substitution_beats_the_indel_pair(self):
        # sub costs 4, delete+insert costs 6 — the alignment must not shred a
        # substitution into a pair, or every B-WER substitution would be
        # double-counted as an insertion plus a deletion.
        steps = align(["a"], ["x"])
        assert [c for c, _, _ in steps] == [Code.substitution]

    def test_empty_sides(self):
        assert align([], []) == []
        assert [c for c, _, _ in align([], ["a"])] == [Code.insertion]
        assert [c for c, _, _ in align(["a"], [])] == [Code.deletion]

    def test_indices_point_at_the_aligned_words(self):
        steps = align(["a", "b", "c"], ["a", "x", "c"])
        code, ref_idx, hyp_idx = steps[1]
        assert (code, ref_idx, hyp_idx) == (Code.substitution, 1, 1)


class TestBUwerSplit:
    def test_error_on_a_listed_word_lands_in_b_wer(self):
        report = score(
            refs(("u1", "call ada now", ["ada"], ["ada"])),
            {"u1": "call adder now"},
        )
        assert (report.b_wer.ref_words, report.b_wer.subs) == (1, 1)
        assert report.b_wer.wer == 1.0
        # The two unlisted words were right: U-WER must be untouched.
        assert (report.u_wer.ref_words, report.u_wer.errors) == (2, 0)
        assert report.u_wer.wer == 0.0
        assert report.wer.wer == pytest.approx(1 / 3)

    def test_error_on_an_unlisted_word_lands_in_u_wer(self):
        report = score(refs(("u1", "call ada now", ["ada"], ["ada"])), {"u1": "call ada later"})
        assert (report.u_wer.ref_words, report.u_wer.subs) == (2, 1)
        assert report.b_wer.errors == 0 and report.b_wer.ref_words == 1

    def test_empty_b_set_is_nan_not_zero(self):
        # No rare words in the reference → B-WER is undefined. Reporting 0 % would
        # read as "biasing was perfect here", which is the opposite of the truth.
        report = score(refs(("u1", "one two", [], [])), {"u1": "one two"})
        assert report.b_wer.ref_words == 0
        assert report.b_wer.wer != report.b_wer.wer  # nan

    def test_deletion_of_a_listed_word(self):
        report = score(refs(("u1", "call ada now", ["ada"], ["ada"])), {"u1": "call now"})
        assert (report.b_wer.dels, report.b_wer.ref_words) == (1, 1)

    def test_missing_hyp_raises_unless_lenient(self):
        r = refs(("u1", "a", [], []), ("u2", "b", [], []))
        with pytest.raises(ValueError):
            score(r, {"u1": "a"})
        report = score(r, {"u1": "a"}, lenient=True)
        assert report.utts == 1


class TestFalseInsertions:
    def test_hallucinated_distractor_is_counted(self):
        # "kafka" is a distractor: fed to the model, absent from the reference.
        report = score(
            refs(("u1", "we met on monday", [], ["kafka"])),
            {"u1": "we met kafka on monday"},
        )
        assert report.false_insertions == 1
        assert report.fp_examples == ["∅ → kafka"]

    def test_overwriting_a_correct_word_with_a_listed_term_is_counted(self):
        # The worse half of the failure, and it never shows up as an insertion.
        report = score(
            refs(("u1", "we met on monday", [], ["kafka"])),
            {"u1": "we kafka on monday"},
        )
        assert report.false_insertions == 1
        assert report.fp_examples == ["met → kafka"]

    def test_a_term_the_reference_really_contains_is_not_a_false_insertion(self):
        report = score(refs(("u1", "kafka streams", ["kafka"], ["kafka"])), {"u1": "kafka streams"})
        assert report.false_insertions == 0
        assert (report.tp, report.gt) == (1, 1)
        assert report.precision == 1.0

    def test_precision_and_recall(self):
        report = score(
            refs(
                ("u1", "call ada", ["ada"], ["ada", "kafka"]),
                ("u2", "we met", [], ["ada", "kafka"]),
            ),
            {"u1": "call ada", "u2": "we kafka"},
        )
        assert (report.gt, report.tp, report.false_insertions) == (1, 1, 1)
        assert report.recall == 1.0
        assert report.precision == 0.5
        assert report.f1 == pytest.approx(2 / 3)


class TestGermanCompounding:
    def test_term_recovered_inside_an_inflected_word_is_a_loose_hit_only(self):
        # "europa" survives as a substring of "europas" — not a strict hit (B-WER
        # rightly charges a substitution), but a different failure from not hearing
        # the term at all, and the gap is the compounding tax we report.
        ref = refs(("u1", "europa waechst", ["europa"], ["europa"]))
        report = score(ref, {"u1": "europas waechst"})
        assert (report.tp, report.tp_loose, report.gt) == (0, 1, 1)
        assert report.recall == 0.0
        assert report.recall_loose == 1.0
        assert report.b_wer.subs == 1

    def test_a_word_missed_outright_is_neither(self):
        ref = refs(("u1", "europa waechst", ["europa"], ["europa"]))
        report = score(ref, {"u1": "amerika waechst"})
        assert (report.tp, report.tp_loose) == (0, 0)


class TestNormalizationProvenance:
    def test_membership_survives_a_word_the_normalizer_shatters(self):
        # normalize("banti's") == "banti s" — two tokens. Membership is decided on
        # the word, so both tokens inherit it; testing the token would drop the
        # term out of the B set entirely and silently shrink the metric.
        report = score(
            refs(("u1", "the banti's disease", ["banti's"], ["banti's"])),
            {"u1": "the bantis disease"},
            normalize=normalize,
        )
        assert report.b_wer.ref_words == 2  # "banti" and "s"
        assert report.u_wer.ref_words == 2  # "the" and "disease"
        assert report.b_wer.errors > 0 and report.u_wer.errors == 0

    def test_case_folds_away_before_scoring(self):
        report = score(
            refs(("u1", "call ada", ["ada"], ["ada"])),
            {"u1": "Call Ada."},
            normalize=normalize,
        )
        assert report.wer.errors == 0
        assert report.tp == 1


class TestSurfaceChanges:
    def test_case_only_rewrite_is_damage_wer_cannot_see(self):
        changes = bias_score.surface_changes("Call Ada now", "Call ADA now", normalize)
        assert [(c.before, c.after) for c in changes] == [("Ada", "ADA")]

    def test_identical_text_has_no_changes(self):
        assert bias_score.surface_changes("Call Ada", "Call Ada", normalize) == []

    def test_a_real_word_change_is_not_a_surface_change(self):
        # It is a genuine edit; WER already sees it, so it must not be double-counted.
        assert bias_score.surface_changes("Call Ada", "Call Adder", normalize) == []

    def test_changed_spans_pairs_the_disagreements(self):
        spans = bias_score.changed_spans("call adder now", "call ada now", normalize)
        assert spans == [("adder", "ada")]

    def test_changed_spans_marks_an_insertion_with_an_empty_side(self):
        assert bias_score.changed_spans("call now", "call ada now", normalize) == [("", "ada")]


class TestIo:
    def test_refs_roundtrip(self, tmp_path):
        path = tmp_path / "ref.tsv"
        original = [RefUtt("u1", "call ada", ("ada",), ("ada", "kafka"))]
        bias_score.write_refs(path, original)
        assert list(bias_score.read_refs(path).values()) == original

    def test_three_column_refs_treat_rare_words_as_the_fed_list(self, tmp_path):
        # is21's format has four columns; three (no distractors yet) is what a
        # freshly derived German fixture looks like before distractors are drawn.
        path = tmp_path / "ref.tsv"
        path.write_text('u1\tcall ada\t["ada"]\n')
        assert bias_score.read_refs(path)["u1"] == RefUtt("u1", "call ada", ("ada",), ("ada",))

    def test_read_refs_rejects_a_short_row(self, tmp_path):
        path = tmp_path / "ref.tsv"
        path.write_text("u1\tcall ada\n")
        with pytest.raises(ValueError):
            bias_score.read_refs(path)

    def test_hyps_roundtrip_keeps_empty_text(self, tmp_path):
        path = tmp_path / "hyp.tsv"
        bias_score.write_hyps(path, {"u1": "call ada", "u2": ""})
        assert bias_score.read_hyps(path) == {"u1": "call ada", "u2": ""}


# --- The oracle: our arithmetic against is21's published numbers ---------------

IS21 = Path(__file__).resolve().parents[1] / "eval" / "out" / "bias" / "is21"


def _oracle_cases() -> list[tuple[Path, Path]]:
    """(hypothesis, result) pairs is21 published, if they have been fetched."""
    results = IS21 / "results"
    if not results.is_dir():
        return []
    return sorted(
        (IS21 / "hyp" / result.name.removesuffix(".result"), result)
        for result in results.glob("*.result")
    )


def _parse_result(text: str) -> dict[str, dict[str, float]]:
    """Their result file: three ``NAME: error_rate=…, ref_words=…, …`` lines."""
    parsed: dict[str, dict[str, float]] = {}
    for line in text.strip().splitlines():
        name, _, rest = line.partition(":")
        parsed[name.strip()] = {
            key: float(value)
            for key, value in (field.strip().split("=") for field in rest.split(","))
        }
    return parsed


@pytest.mark.skipif(
    not _oracle_cases(),
    reason="needs is21_deep_bias data — uv run --group eval eval/bias_data.py --fetch is21",
)
class TestPublishedOracle:
    """Reproduce the INTERSPEECH-2021 numbers from the authors' own hypotheses.

    Their reference files differ across list sizes only in the *fed* list; the
    rare-word column that B-WER is computed over is identical in all of them
    (verified), so every hypothesis for a test set scores against that set's N=100
    reference — which is why the oracle costs 7 MB and not 235 MB.
    """

    @pytest.mark.parametrize("hyp_path,result_path", _oracle_cases(), ids=lambda p: p.name)
    def test_reproduces_published_result(self, hyp_path: Path, result_path: Path):
        split = hyp_path.name.split(".")[0]  # test-clean / test-other
        refs = bias_score.read_refs(IS21 / "ref" / f"{split}.biasing_100.tsv")
        hyps = bias_score.read_hyps(hyp_path)
        report = score(refs, hyps)  # their text is pre-normalized: no normalizer

        expected = _parse_result(result_path.read_text())
        for name, err in (
            ("WER", report.wer),
            ("U-WER", report.u_wer),
            ("B-WER", report.b_wer),
        ):
            want = expected[name]
            assert err.ref_words == want["ref_words"], name
            assert err.subs == want["subs"], name
            assert err.ins == want["ins"], name
            assert err.dels == want["dels"], name
            # Their error_rate is a percentage; ours is a fraction.
            assert err.wer * 100 == pytest.approx(want["error_rate"], abs=1e-9), name

    def test_baseline_b_wer_is_far_worse_than_u_wer(self):
        """The premise of the whole exercise: rare words are ~6x harder.

        If this ever stops holding, the biasing lists have stopped selecting for
        hard words and every downstream number is measuring nothing.
        """
        refs = bias_score.read_refs(IS21 / "ref" / "test-clean.biasing_100.tsv")
        hyps = bias_score.read_hyps(IS21 / "hyp" / "test-clean.b1.rnnt_baseline.tsv")
        report = score(refs, hyps)
        assert report.b_wer.wer > 5 * report.u_wer.wer
