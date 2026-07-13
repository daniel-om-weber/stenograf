"""Scoring for contextual biasing — B-WER, U-WER, entity F, false insertions.

Pure: no model, no audio, no ``stenograf`` import. That is what lets
``tests/test_eval_bias.py`` pin it against a published oracle (see below) in the
default test run, exactly as ``eval/der.py`` is pinned by ``tests/test_eval_der.py``.

Biasing is a *targeted* intervention, so a single WER hides it twice over: the
words it is meant to fix are ~10 % of tokens (any gain drowns), and the damage it
does — rewriting words that were already right — lands on the other 90 % (any loss
drowns too). The literature's answer is to split WER by whether the reference word
was in the biasing list:

- **B-WER** — WER over reference words *in* the list. Biasing must push this down.
- **U-WER** — WER over every other word. Biasing must not push this up. Over-boosting
  is visible here and nowhere else.

Both come from one alignment, and that alignment is a faithful port of
``facebookresearch/fbai-speech/is21_deep_bias/score.py`` (MIT) — same edit costs
(match 0, insert/delete 3, substitute 4), same tie-breaking, same accounting. Not
jiwer, deliberately: reproducing their arithmetic to the digit turns the 44
hypothesis/result file pairs they publish into a free correctness oracle for this
scorer, which is a far stronger check than any test we could write ourselves.

Two metrics they do not give us, because their harness cannot see them:

- **False insertions.** Their scorer credits an insertion to B only if the inserted
  word is in *that utterance's own rare-word set* — i.e. a word the reference
  actually contains — so a boosted distractor the model hallucinates is scored as a
  U-insertion and B-insertions are 0 in every published result. The failure mode
  biasing is notorious for is thereby invisible. We count hypothesis words drawn
  from the list the model was actually *fed* (rare words **plus** distractors), at
  positions where the reference has something else — see :attr:`Report.false_insertions`.
- **Strict-form damage.** Scoring normalizes case and punctuation away, which is
  right for WER and wrong for us: "Ada" → "ADA" is a real regression that
  normalization erases. :func:`surface_changes` counts it, comparing two systems'
  *raw* text (see the docstring there for why it is paired rather than referenced).

German (:mod:`eval.bias_data` Tier 2) adds a wrinkle English does not have: a term
survives inflection and compounding as a *substring* ("Europa" inside "Europas",
"Dashboard" inside "Grafana-Dashboard"), so strict word equality under-counts what
the decoder actually recovered. Every entity number is therefore reported twice,
strict and prefix-tolerant (:attr:`Report.recall` / :attr:`Report.recall_loose`).
Neither is the truth on its own; the gap between them is the compounding tax.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

Normalizer = Callable[[str], str]


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


# is21's costs. Substitution (4) is dearer than a delete+insert pair is cheap
# (3 each) but cheaper than the two together (6), which is what keeps the
# alignment from shredding a substitution into an indel pair.
_COST_INDEL = 3
_COST_SUB = 4


def align(refs: Sequence[str], hyps: Sequence[str]) -> list[tuple[Code, int, int]]:
    """Levenshtein alignment, is21-exact — ``(code, ref_idx, hyp_idx)`` per step.

    Indices are into ``refs``/``hyps``; the one that does not apply to a code
    (hyp for a deletion, ref for an insertion) still carries the neighbouring
    position, as in the original, and callers must not read it.

    The tie-breaking is load-bearing for reproducing published numbers: the
    diagonal is taken first and an insertion or deletion only displaces it on a
    *strict* improvement.
    """
    n, m = len(refs), len(hyps)
    if n == 0 and m == 0:
        return []

    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i == 0:
                scores[i][j] = scores[i][j - 1] + _COST_INDEL
                back[i][j] = (i, j - 1)
                continue
            if j == 0:
                scores[i][j] = scores[i - 1][j] + _COST_INDEL
                back[i][j] = (i - 1, j)
                continue
            best = scores[i - 1][j - 1] + (0 if refs[i - 1] == hyps[j - 1] else _COST_SUB)
            prev = (i - 1, j - 1)
            if scores[i][j - 1] + _COST_INDEL < best:
                best = scores[i][j - 1] + _COST_INDEL
                prev = (i, j - 1)
            if scores[i - 1][j] + _COST_INDEL < best:
                best = scores[i - 1][j] + _COST_INDEL
                prev = (i - 1, j)
            scores[i][j] = best
            back[i][j] = prev

    steps: list[tuple[Code, int, int]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        ref_idx, hyp_idx = i - 1, j - 1
        if pi == i - 1 and pj == j:
            code = Code.deletion
        elif pi == i and pj == j - 1:
            code = Code.insertion
        elif refs[ref_idx] == hyps[hyp_idx]:
            code = Code.match
        else:
            code = Code.substitution
        steps.append((code, ref_idx, hyp_idx))
        i, j = pi, pj
    steps.reverse()
    return steps


@dataclass
class WordError:
    """One WER accumulator: errors over reference words."""

    ref_words: int = 0
    subs: int = 0
    ins: int = 0
    dels: int = 0

    @property
    def errors(self) -> int:
        return self.subs + self.ins + self.dels

    @property
    def wer(self) -> float:
        """Errors per reference word. ``nan`` when nothing was scored — an empty
        B set is a fact about the data, not a 0 % error rate."""
        if self.ref_words == 0:
            return float("nan")
        return self.errors / self.ref_words


def tokenize(words: Iterable[str], normalize: Normalizer | None) -> tuple[list[str], list[int]]:
    """Scoring tokens, plus the index of the source word each one came from.

    A normalizer can split one word into several tokens ("15.7." → "15 7") or drop
    it entirely, so token indices do not map 1:1 to words — the same trap
    ``eval/parity.py::_tokens_with_sources`` exists to sidestep. Biasing needs the
    provenance for a sharper reason than timestamps: list membership is a property
    of the *word* ("banti's"), and once normalization has shattered it into "banti"
    and "s" the membership test can no longer be made on the token. So we test the
    word and let its tokens inherit the answer.
    """
    tokens: list[str] = []
    sources: list[int] = []
    for index, word in enumerate(words):
        pieces = word.split() if normalize is None else normalize(word).split()
        for piece in pieces:
            tokens.append(piece)
            sources.append(index)
    return tokens, sources


@dataclass(frozen=True)
class RefUtt:
    """A reference utterance and its two word lists.

    ``rare_words`` is what the reference actually contains and is therefore the B
    set — the words B-WER is computed over. ``biasing_words`` is the whole list
    handed to the *model*, rare words plus distractors, and is what the decoder is
    biased with. Keeping them apart is the point: a metric computed over the fed
    list would count the model right for not saying words that were never there.
    """

    uttid: str
    text: str
    rare_words: tuple[str, ...] = ()
    biasing_words: tuple[str, ...] = ()


@dataclass
class Report:
    wer: WordError = field(default_factory=WordError)
    u_wer: WordError = field(default_factory=WordError)
    b_wer: WordError = field(default_factory=WordError)
    utts: int = 0

    gt: int = 0
    """Reference words in the B set — the denominator of recall."""
    tp: int = 0
    tp_loose: int = 0
    false_insertions: int = 0
    """Hypothesis words from the *fed* list (distractors included) at positions the
    reference fills with something else. Ground truth, not a guess: a distractor is
    known not to be in the reference."""
    fp_examples: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.tp / self.gt if self.gt else float("nan")

    @property
    def recall_loose(self) -> float:
        """Recall crediting a term recovered inside an inflected or compounded word
        ("Europa" in "Europas"). Always ≥ :attr:`recall`; the gap is what strict
        word-level scoring costs us in German."""
        return self.tp_loose / self.gt if self.gt else float("nan")

    @property
    def precision(self) -> float:
        hits = self.tp + self.false_insertions
        return self.tp / hits if hits else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not p or not r or p != p or r != r:  # nan or zero
            return float("nan")
        return 2 * p * r / (p + r)


def score(
    refs: dict[str, RefUtt],
    hyps: dict[str, str],
    *,
    normalize: Normalizer | None = None,
    lenient: bool = False,
) -> Report:
    """Score hypotheses against references.

    ``normalize`` is applied per *word* (see :func:`tokenize`); pass ``None`` to
    score the text exactly as given — which is what the oracle test does, because
    is21's published files are already normalized and touching them would change
    their numbers.
    """
    missing = [uttid for uttid in refs if uttid not in hyps]
    if missing and not lenient:
        raise ValueError(f"{len(missing)} reference utts missing from hyps (e.g. {missing[0]})")

    report = Report()
    for uttid, ref in refs.items():
        hyp_text = hyps.get(uttid)
        if hyp_text is None:
            continue
        report.utts += 1
        _score_utt(report, ref, hyp_text, normalize)
    return report


def _key(word: str, normalize: Normalizer | None) -> str:
    return word if normalize is None else normalize(word)


def _score_utt(report: Report, ref: RefUtt, hyp_text: str, normalize: Normalizer | None) -> None:
    ref_words = ref.text.split()
    hyp_words = hyp_text.split()
    ref_tokens, ref_sources = tokenize(ref_words, normalize)
    hyp_tokens, hyp_sources = tokenize(hyp_words, normalize)

    # Membership is decided on the word, before normalization can shatter it, and
    # the word's tokens inherit the verdict (see tokenize()).
    b_set = {_key(w, normalize) for w in ref.rare_words}
    fed_set = {_key(w, normalize) for w in ref.biasing_words} or b_set
    ref_is_b = [_key(w, normalize) in b_set for w in ref_words]
    hyp_in_b = [_key(w, normalize) in b_set for w in hyp_words]
    hyp_in_fed = [_key(w, normalize) in fed_set for w in hyp_words]

    for code, ref_idx, hyp_idx in align(ref_tokens, hyp_tokens):
        biased = code != Code.insertion and ref_is_b[ref_sources[ref_idx]]
        target = report.b_wer if biased else report.u_wer

        if code == Code.match:
            report.wer.ref_words += 1
            target.ref_words += 1
            if biased:
                report.gt += 1
                report.tp += 1
                report.tp_loose += 1
        elif code == Code.substitution:
            report.wer.ref_words += 1
            report.wer.subs += 1
            target.ref_words += 1
            target.subs += 1
            if biased:
                report.gt += 1
                # The decoder may have recovered the term *inside* a longer word —
                # inflected or compounded — which strict equality scores as a plain
                # substitution. It is not a hit, but it is not the same failure as
                # not hearing the term at all, so it is counted separately.
                term = ref_tokens[ref_idx]
                if term and term in hyp_tokens[hyp_idx]:
                    report.tp_loose += 1
        elif code == Code.deletion:
            report.wer.ref_words += 1
            report.wer.dels += 1
            target.ref_words += 1
            target.dels += 1
            if biased:
                report.gt += 1
        elif code == Code.insertion:
            report.wer.ins += 1
            # is21 charges the insertion to B only when the inserted word is one of
            # *this* reference's own rare words — kept for numeric fidelity, which
            # is also why their B-insertions are always 0. Our own false-insertion
            # count below is the metric that actually sees hallucinated terms.
            if hyp_in_b[hyp_sources[hyp_idx]]:
                report.b_wer.ins += 1
            else:
                report.u_wer.ins += 1

        # A hypothesis word from the fed list, where the reference disagrees, is a
        # term the boosting tree talked the decoder into. Substitutions count too,
        # not just insertions: overwriting a correct word with a glossary term is
        # the *worse* half of the failure, and it never appears as an insertion.
        if code in (Code.insertion, Code.substitution) and hyp_in_fed[hyp_sources[hyp_idx]]:
            report.false_insertions += 1
            if len(report.fp_examples) < 40:
                ref_word = "∅" if code == Code.insertion else ref_words[ref_sources[ref_idx]]
                report.fp_examples.append(f"{ref_word} → {hyp_words[hyp_sources[hyp_idx]]}")


@dataclass(frozen=True)
class SurfaceChange:
    before: str
    after: str


def surface_changes(baseline: str, biased: str, normalize: Normalizer) -> list[SurfaceChange]:
    """Words whose *surface form* biasing changed while leaving the token identical.

    "Ada" → "ADA" is damage, and it is exactly the damage every WER-shaped metric
    is built to ignore: the normalizer folds it away before scoring, so B-WER, U-WER
    and entity F all report nothing happened.

    Paired against the unbiased run rather than the reference, because the reference
    cannot answer this. LibriSpeech and MLS transcripts are lowercase and
    unpunctuated while Parakeet emits true case — measured against them, *every*
    correctly transcribed word is a surface change and the number is noise. Two runs
    of the same model differ only where biasing intervened, which is the question.
    """
    a, b = baseline.split(), biased.split()
    changes: list[SurfaceChange] = []
    for code, i, j in align([normalize(w) for w in a], [normalize(w) for w in b]):
        if code == Code.match and a[i] != b[j]:
            changes.append(SurfaceChange(a[i], b[j]))
    return changes


def changed_spans(baseline: str, biased: str, normalize: Normalizer) -> list[tuple[str, str]]:
    """Every disagreement between two systems, as ``(baseline, biased)`` word pairs.

    The unit of Tier 6 adjudication: two systems that agree carry no information
    about which is better, so only these are worth a human's ear.
    """
    a, b = baseline.split(), biased.split()
    spans: list[tuple[str, str]] = []
    for code, i, j in align([normalize(w) for w in a], [normalize(w) for w in b]):
        if code == Code.match:
            continue
        before = "" if code == Code.insertion else a[i]
        after = "" if code == Code.deletion else b[j]
        spans.append((before, after))
    return spans


def read_refs(path: Path) -> dict[str, RefUtt]:
    """Read an is21-format reference TSV.

    Four tab-separated columns: utterance id, reference text, the reference's own
    rare words (JSON list), and the biasing list fed to the model (JSON list,
    a superset of the third). Three columns — no distractors — is accepted too.
    """
    refs: dict[str, RefUtt] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            raise ValueError(f"{path}: expected >=3 tab-separated columns, got {len(cols)}")
        rare = tuple(json.loads(cols[2]))
        biasing = tuple(json.loads(cols[3])) if len(cols) > 3 else rare
        refs[cols[0]] = RefUtt(uttid=cols[0], text=cols[1], rare_words=rare, biasing_words=biasing)
    return refs


def write_refs(path: Path, refs: Iterable[RefUtt]) -> None:
    lines = [
        "\t".join(
            [r.uttid, r.text, json.dumps(sorted(r.rare_words)), json.dumps(sorted(r.biasing_words))]
        )
        for r in refs
    ]
    path.write_text("\n".join(lines) + "\n")


def read_hyps(path: Path) -> dict[str, str]:
    """Read a hypothesis TSV: utterance id, then text (which may be empty)."""
    hyps: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        hyps[cols[0]] = cols[1] if len(cols) > 1 else ""
    return hyps


def write_hyps(path: Path, hyps: dict[str, str]) -> None:
    path.write_text("\n".join(f"{uttid}\t{text}" for uttid, text in hyps.items()) + "\n")


def format_report(report: Report) -> str:
    """The three is21 lines, verbatim in shape, plus what they cannot see."""
    lines = []
    for name, err in (("WER", report.wer), ("U-WER", report.u_wer), ("B-WER", report.b_wer)):
        lines.append(
            f"{name}: error_rate={err.wer:.4%}, ref_words={err.ref_words}, "
            f"subs={err.subs}, ins={err.ins}, dels={err.dels}"
        )
    lines.append(
        f"entity: recall={report.recall:.1%} (loose {report.recall_loose:.1%}), "
        f"precision={report.precision:.1%}, F1={report.f1:.1%}, "
        f"false_insertions={report.false_insertions}, gt={report.gt}"
    )
    return "\n".join(lines)


__all__ = [
    "Code",
    "Report",
    "RefUtt",
    "SurfaceChange",
    "WordError",
    "align",
    "changed_spans",
    "format_report",
    "read_hyps",
    "read_refs",
    "score",
    "surface_changes",
    "tokenize",
    "write_hyps",
    "write_refs",
]
