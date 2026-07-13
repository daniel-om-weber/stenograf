"""Contextual-biasing evaluation — the driver that turns anecdote into numbers.

We ship decode-time biasing (``stenograf.asr.biasing``) whose *tree* is verified
against NeMo's golden vectors, but whose *effect* rested on one TTS clip and three
meeting WAVs. That proves the mechanism fires. It cannot tune ``[asr] boost``, and
it cannot defend the two places we deliberately diverge from NeMo (``unk_score=1.0``
— the paper's greedy recommendation, where their code ships 0.0 — and a German
compound-tail tokenization NeMo does not have at all). This driver settles those,
against real metrics, with zero hand labeling.

Four tiers, each answering a question the others cannot:

``--tier english``
    The **correctness gate**, and the only reason English is here at all: it is the
    one language with *published* numbers to check ourselves against. If B-WER
    moves the way the literature says it does, the port is right end-to-end — a
    stronger claim than any unit test, and far cheaper than installing torch+NeMo
    to diff against. We are not shipping English transcription quality.

``--tier german``
    The benchmark that decides what ships. It did not exist (no German biasing
    benchmark does); ``eval/bias_data.py`` derives it by the same recipe as the
    English artifact, from MLS's own free transcripts.

``--tier distractor``
    False insertions with ground truth and no references at all: bias with terms
    known to be *absent*, so any change is by definition a false insertion. Runs on
    any audio, including Daniel's own meetings. This is the ad-hoc check that
    caught the ``unk_score`` damage in the first place — formalized.

``--tier tts``
    A **diagnostic, never a metric** (see ``eval/bias_tts.py``): can biasing reach
    this class of error at all? Synthetic speech mispronounces the very jargon it is
    supposed to test, so a "gain" measured here is a number about the TTS engine.

Every tier decodes through the *real shipped backend* (``create_backend``, the seam
``eval/parity.py`` uses), so what is measured is what ships.

**Two layers, on any tier: ``--post``.** stenograf steers the decoder toward the
glossary *and* fuzzy-corrects the finished text against it (``stenograf.glossary``),
and until 2026-07-13 only the first had ever been measured. ``--post`` scores each
layer alone and both together, on the same terms — and costs no decoding at all,
since post-correction is a pure text transform over hypotheses the other arms have
already cached. It immediately convicted the *second* layer: at its then-default
0.82 threshold it bought rare words by rewriting words that were never wrong (U-WER
+6.5 % German, +86 % English; 84 false insertions against biasing's 3), which is
worse than the ``boost = 2.0`` config we had rejected for that exact failure. The
default is now 0.95, where the layer adds B-WER on top of biasing with U-WER flat.
Keep pointing this arm at any knob that only ever gets judged by B-WER.

Usage:
    uv run --group eval eval/bias.py --tier english --n 100
    uv run --group eval eval/bias.py --tier german --sweep
    uv run --group eval eval/bias.py --tier german --post          # both layers
    uv run --group eval eval/bias.py --tier german --post 0.88 0.92 0.95
    uv run --group eval eval/bias.py --tier distractor --wav eval/audio/*.wav
    uv run --group eval eval/bias.py --tier tts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bias_score  # noqa: E402
from bias_data import BIAS_DIR, DE_DIR, IS21_DIR  # noqa: E402
from common import OUT_DIR, read_pcm16  # noqa: E402
from score import normalize  # noqa: E402

from stenograf.asr import create_backend  # noqa: E402
from stenograf.asr.biasing import DEFAULT_ALPHA, DEFAULT_UNK_SCORE, boost_terms  # noqa: E402
from stenograf.asr.biasing import build as build_tree  # noqa: E402
from stenograf.asr.tokens import load_encoder  # noqa: E402
from stenograf.glossary import DEFAULT_THRESHOLD  # noqa: E402

REPORT = OUT_DIR / "bias-report-{tier}.md"
"""Per tier, because one shared filename means the last run silently overwrites the
table before it — an English gate erasing the German sweep that took 20 minutes."""

HYP_DIR = BIAS_DIR / "hyp"
TTS_MANIFEST = BIAS_DIR / "tts" / "manifest.json"

SUBSAMPLE = 500
"""Utterances per config unless ``--limit 0``. The full grid over the full test sets
is 5–10 h of decoding on this Mac; the subsample is pinned by seed, so every config
sees the *same* utterances and the comparison between them is exact. Only the
winning config is re-run over the full set — and every table says which it was."""

SEED = 20260713

# Tier-1 acceptance band, from the is21 authors' own published results (RNN-T +
# trie deep biasing, test-clean): baseline B-WER 14.08 vs U-WER 2.37 (≈6x), and at
# N=100 deep biasing takes B-WER down ~30 % relative with U-WER flat (2.37 → 2.28).
# Judge the *relative* B-WER drop and the U-WER delta, never the absolute numbers:
# Parakeet-v3 is a far stronger baseline than their 2021 model, and greedy decoding
# captures roughly half of what beam search gets (TurboBias: recall +30 pts greedy
# vs +47 beam). Do not tune `boost` into the ground chasing beam headlines.
MIN_B_WER_DROP = 0.15
"""Relative. Below this at N=100, biasing is not really wired into the decoder."""
MAX_U_WER_RISE = 0.05
"""Relative. Above this, biasing is paying for its wins by damaging everything else
— the failure mode NVIDIA warns about, and the reason `boost` is a setting."""


@dataclass(frozen=True)
class Config:
    """One point in the ablation grid.

    ``alpha`` and ``post`` are the two *layers* stenograf ships — the decoder is
    biased while it transcribes (``stenograf.asr.biasing``) and the finished text is
    fuzzy-corrected against the same terms (``stenograf.glossary``). They are
    independent knobs here because the pipeline runs both and we have never priced
    them apart: whether the second still earns its keep now that the first lands
    upstream of it is an open question, and ``post=0`` / ``alpha=0`` answer it.
    """

    alpha: float = DEFAULT_ALPHA
    unk: float = DEFAULT_UNK_SCORE
    compound: bool = True
    n: int = 100
    post: float = 0.0
    """Post-correction similarity threshold; 0 disables the layer."""

    @property
    def is_baseline(self) -> bool:
        """Neither layer on — the stock decode loop with untouched text, the number
        every other config is measured against."""
        return self.alpha == 0 and not self.post

    @property
    def decoded_as(self) -> Config:
        """The config whose *decode* produces this one's hypotheses.

        Post-correction is a pure text transform, so a post arm re-uses the cached
        hypotheses of its non-post twin instead of decoding again — which is what
        makes the whole two-layer grid free.
        """
        return replace(self, post=0.0)

    @property
    def tag(self) -> str:
        if self.is_baseline:
            return "baseline"
        if not self.alpha:
            return f"post{self.post:g}-n{self.n}"
        tail = f"a{self.alpha:g}-u{self.unk:g}-c{int(self.compound)}-n{self.n}"
        return f"{tail}-p{self.post:g}" if self.post else tail

    def __str__(self) -> str:
        if self.is_baseline:
            return "neither layer"
        if not self.alpha:
            return f"post-correction only ({self.post:g}), N={self.n}"
        tail = "" if self.compound else ", no compound-tail"
        post = f" + post ({self.post:g})" if self.post else ""
        return f"boost={self.alpha:g}, unk={self.unk:g}, N={self.n}{tail}{post}"


class Biaser:
    """Re-arms the shipped decode loop with a fresh tree, per utterance.

    The benchmark hands every utterance its *own* biasing list, but the shipped
    backend compiles the tree once, at ``load()`` — reasonably, since a meeting has
    one glossary. Reloading a model per utterance is not an option, so the eval
    reaches past the public API and re-splices the same decode loop the backend
    installs, with a new tree. That is deliberate, and it is contained here.

    It also reaches for two knobs the shipped API does not expose — ``unk_score``
    and compound-tail tokenization — for the plain reason that this harness exists
    to *set* their defaults. A knob whose value is a guess is not a constant, and
    the only way to stop guessing is to sweep it.
    """

    _stock_loops: ClassVar[dict[int, object]] = {}
    """The *unbiased* decode loop, per loaded backend, captured the first time we
    touch it and never again. Re-reading it per Biaser would be a trap: a config
    whose decode died mid-run leaves its tree spliced in, and the next config would
    then capture *that* as its baseline and silently score itself against the
    previous config's glossary — the kind of bug that invalidates a whole sweep
    while every number still looks plausible."""

    def __init__(self, backend, config: Config) -> None:
        self._backend = backend
        self._config = config
        self._name = backend.name
        self.build_seconds = 0.0
        if self._name == "parakeet":
            model = backend._model
            self._vocab_size = len(model.vocabulary)
            self._stock = self._stock_loops.setdefault(id(backend), model.decode_greedy)
        elif self._name == "parakeet-onnx":
            asr = backend._model.asr
            self._vocab_size = asr._blank_idx
            self._stock = self._stock_loops.setdefault(id(backend), asr._decoding)
        else:
            raise SystemExit(f"biasing eval supports the parakeet backends, not {self._name!r}")

    def arm(self, terms: list[str]) -> None:
        """Install a tree for ``terms`` (or restore the stock loop when empty)."""
        started = time.perf_counter()
        tree = None
        if terms and self._config.alpha:
            tree = build_tree(
                boost_terms(terms),
                _encoder(compound=self._config.compound),
                vocab_size=self._vocab_size,
                unk_score=self._config.unk,
            )
        self.build_seconds += time.perf_counter() - started

        if tree is None:
            self._restore()
            return
        if self._name == "parakeet":
            from stenograf.asr.parakeet import _biased_decode_greedy

            model = self._backend._model
            model.decode_greedy = _biased_decode_greedy(
                model, tree, self._config.alpha, self._backend._mx
            )
        else:
            from stenograf.asr.parakeet_onnx import _biased_decoding

            asr = self._backend._model.asr
            asr._decoding = _biased_decoding(asr, tree, self._config.alpha)

    def _restore(self) -> None:
        if self._name == "parakeet":
            self._backend._model.decode_greedy = self._stock
        else:
            self._backend._model.asr._decoding = self._stock


@cache
def _encoder(*, compound: bool):
    """The model's own encoder, memoized across utterances.

    Memoizing is what makes per-utterance arming affordable: every utterance draws
    its distractors from one shared pool, so the same term is re-encoded thousands of
    times across a run, and SentencePiece is the bulk of the tree-build cost.

    ``@cache`` on the *factory* is the load-bearing half. Without it every ``arm()``
    built a fresh closure around a fresh empty cache — a memoizer that never got to
    remember anything, which is worse than none at all, because it looks like one.

    ``compound=False`` is the ablation: keep only the word-initial tokenization and
    drop the compound-tail form, i.e. the shipped encoder as it would be without our
    German-specific addition.
    """
    encode = load_encoder()

    @cache
    def cached(term: str) -> tuple[tuple[int, ...], ...]:
        forms = encode(term)
        return tuple(tuple(form) for form in (forms[:1] if not compound else forms))

    return lambda term: [list(form) for form in cached(term)]


# --- Corpora ------------------------------------------------------------------

CORPORA = {
    "english": ("openslr/librispeech_asr", "clean/test/0000.parquet"),
    "german": ("facebook/multilingual_librispeech", "german/test-00000-of-00001.parquet"),
}
"""The *exact* test-split parquet, not the config.

``load_dataset(repo, config, split="test")`` downloads every file the config
declares and selects the split afterwards — for LibriSpeech's "clean" that is
train-clean-360 too, 30+ GB to read a 350 MB test set (measured, the hard way).
Naming the file keeps it honest."""


def ref_path(tier: str, n: int) -> Path:
    if tier == "english":
        return IS21_DIR / "ref" / f"test-clean.biasing_{n}.tsv"
    return DE_DIR / "ref" / f"mls-de-test.biasing_{n}.tsv"


def load_audio(tier: str, uttids: set[str]) -> dict[str, np.ndarray]:
    """The subset's audio, keyed by utterance id.

    Column names are discovered rather than assumed — the two corpora disagree
    ("text" vs "transcript") and a silent mismatch would score a whole run against
    the wrong references.
    """
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    repo, parquet = CORPORA[tier]
    print(f"loading {repo} [{parquet}] …", flush=True)
    path = hf_hub_download(repo, parquet, repo_type="dataset")
    table = pq.ParquetFile(path)

    columns = set(table.schema_arrow.names)
    id_column = next((c for c in ("id", "utt_id", "audio_id") if c in columns), None)
    if id_column is None or "audio" not in columns:
        raise SystemExit(f"{repo}: expected an id and 'audio' column, found {sorted(columns)}")

    audio: dict[str, np.ndarray] = {}
    for batch in table.iter_batches(batch_size=64, columns=[id_column, "audio"]):
        for row in batch.to_pylist():
            uttid = row[id_column]
            if uttid not in uttids:
                continue  # only the subsample is decoded, which is most of the win
            samples, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != 16_000:
                raise SystemExit(f"{repo}: expected 16 kHz audio, got {rate}")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            audio[uttid] = samples

    missing = uttids - set(audio)
    if missing:
        raise SystemExit(
            f"{len(missing)} utts have references but no audio (e.g. {next(iter(missing))}) — "
            f"the reference ids and {repo}'s '{id_column}' column disagree"
        )
    return audio


def subsample(refs: dict[str, bias_score.RefUtt], limit: int, seed: int, tier: str) -> list[str]:
    """A pinned subset — the same utterances for every config, on every machine."""
    uttids = sorted(refs)
    if limit <= 0 or limit >= len(uttids):
        return uttids
    return sorted(random.Random(f"{seed}:{tier}").sample(uttids, limit))


# --- Decoding -----------------------------------------------------------------


def decode(
    backend,
    config: Config,
    audio: dict[str, np.ndarray],
    refs: dict[str, bias_score.RefUtt],
    *,
    tier: str,
    seed: int,
    force: bool = False,
) -> dict[str, str]:
    """Transcribe every utterance under ``config``, caching to a hypothesis TSV.

    The cache is what makes a 5-hour grid resumable: a config already decoded is
    read back instead of re-run, so an interrupted sweep costs only the configs it
    had not reached.

    The seed is part of the key, not decoration: it chooses *which* utterances the
    subsample holds, so two seeds at the same ``--limit`` would otherwise collide on
    one filename and the second run would silently score the first run's audio.
    """
    path = HYP_DIR / tier / f"{backend.name}.{config.tag}.{len(audio)}s{seed}.tsv"
    if path.exists() and not force:
        cached = bias_score.read_hyps(path)
        if set(cached) == set(audio):
            print(f"  {config} — cached")
            return cached

    biaser = Biaser(backend, config)
    hyps: dict[str, str] = {}
    started = time.perf_counter()
    seconds = 0.0
    for index, uttid in enumerate(sorted(audio), start=1):
        samples = audio[uttid]
        # The *fed* list: the utterance's own rare words plus its distractors. The
        # model is biased with everything, and scored only on what was really said.
        biaser.arm(list(refs[uttid].biasing_words) if not config.is_baseline else [])
        segments = backend.transcribe(samples, None)
        hyps[uttid] = " ".join(segment.text for segment in segments).strip()
        seconds += len(samples) / 16_000
        if index % 50 == 0 or index == len(audio):
            elapsed = time.perf_counter() - started
            print(
                f"  {config} — {index}/{len(audio)} "
                f"({seconds / elapsed:.1f}x RT, {biaser.build_seconds:.0f}s in tree builds)",
                flush=True,
            )
    biaser._restore()

    path.parent.mkdir(parents=True, exist_ok=True)
    bias_score.write_hyps(path, hyps)
    return hyps


def post_correct(
    hyps: dict[str, str], refs: dict[str, bias_score.RefUtt], threshold: float
) -> dict[str, str]:
    """Run the shipped fuzzy post-correction over decoded hypotheses.

    Each utterance is corrected against *its own* biasing list — the same terms the
    tree was armed with — so the two layers are measured on identical information
    and the only difference between them is where they act.

    This calls the shipped ``apply_glossary`` through its public API rather than its
    internals: an entry with no words is exactly what a wordless backend produces,
    and its ``text`` is corrected on its own. Nothing here is eval-only behaviour.
    """
    from stenograf.glossary import apply_glossary
    from stenograf.transcript import TranscriptEntry

    out: dict[str, str] = {}
    for uttid, text in hyps.items():
        entry = TranscriptEntry(speaker="", text=text, start=0.0, end=0.0)
        corrected = apply_glossary(
            [entry], glossary=list(refs[uttid].biasing_words), threshold=threshold
        )
        out[uttid] = corrected[0].text
    return out


# --- Reporting ----------------------------------------------------------------


def _rel(new: float, old: float) -> float:
    """Relative change, guarding the degenerate baseline."""
    if not old or old != old:
        return float("nan")
    return (new - old) / old


@dataclass
class Row:
    config: Config
    report: bias_score.Report
    surface_damage: int
    changed: int


def evaluate(
    tier: str,
    configs: list[Config],
    *,
    limit: int,
    seed: int,
    backend_name: str,
    force: bool,
) -> list[Row]:
    refs_by_n = {}
    for n in sorted({c.n for c in configs}):
        path = ref_path(tier, n)
        if not path.exists():
            raise SystemExit(
                f"missing {path} — uv run --group eval eval/bias_data.py "
                f"--fetch {'is21' if tier == 'english' else 'german'} --sizes {n}"
            )
        refs_by_n[n] = bias_score.read_refs(path)

    # The subset is drawn from the N=100 reference (all sizes cover the same utts).
    any_refs = refs_by_n[min(refs_by_n)]
    uttids = subsample(any_refs, limit, seed, tier)
    audio = load_audio(tier, set(uttids))
    print(f"{len(uttids)} utterances" + ("" if limit <= 0 else f" (pinned subsample, seed {seed})"))

    backend = create_backend(backend_name)
    backend.load()

    rows: list[Row] = []
    baseline_hyps: dict[str, str] = {}
    for config in configs:
        refs = {uttid: refs_by_n[config.n][uttid] for uttid in uttids}
        hyps = decode(backend, config.decoded_as, audio, refs, tier=tier, seed=seed, force=force)
        if config.post:
            hyps = post_correct(hyps, refs, config.post)
        report = bias_score.score(refs, hyps, normalize=normalize)
        if config.is_baseline:
            baseline_hyps = hyps

        damage = changed = 0
        if baseline_hyps and not config.is_baseline:
            for uttid in uttids:
                before, after = baseline_hyps[uttid], hyps[uttid]
                damage += len(bias_score.surface_changes(before, after, normalize))
                changed += len(bias_score.changed_spans(before, after, normalize))
        rows.append(Row(config, report, damage, changed))

    backend.unload()
    return rows


def format_table(tier: str, rows: list[Row], *, limit: int, backend_name: str) -> str:
    base = next((r for r in rows if r.config.is_baseline), None)
    lines = [
        f"## {tier} — {backend_name}",
        "",
        f"{len(rows)} configs; "
        + ("full test set" if limit <= 0 else f"**pinned {limit}-utterance subsample**")
        + f"; {rows[0].report.utts} utts scored.",
        "",
        "| config | B-WER | ΔB | U-WER | ΔU | recall (loose) | false ins. "
        "| surface damage | changed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        r = row.report
        db = du = float("nan")
        if base is not None and not row.config.is_baseline:
            db = _rel(r.b_wer.wer, base.report.b_wer.wer)
            du = _rel(r.u_wer.wer, base.report.u_wer.wer)
        lines.append(
            f"| {row.config} | {r.b_wer.wer:.2%} | "
            + ("—" if db != db else f"{db:+.1%}")
            + f" | {r.u_wer.wer:.2%} | "
            + ("—" if du != du else f"{du:+.1%}")
            + f" | {r.recall:.1%} ({r.recall_loose:.1%}) | {r.false_insertions} "
            f"| {row.surface_damage} | {row.changed} |"
        )
    if base is not None:
        fp = [row for row in rows if not row.config.is_baseline]
        if fp and fp[0].report.fp_examples:
            lines += ["", "False insertions (sample): " + "; ".join(fp[0].report.fp_examples[:8])]
    return "\n".join(lines) + "\n"


def gate(rows: list[Row]) -> bool:
    """The Tier-1 acceptance band — is the port wired in, and is it damaging?"""
    base = next((r for r in rows if r.config.is_baseline), None)
    biased = [r for r in rows if not r.config.is_baseline]
    if base is None or not biased:
        return True

    ok = True
    if not base.report.b_wer.wer > 5 * base.report.u_wer.wer:
        print(
            f"WARN: baseline B-WER ({base.report.b_wer.wer:.2%}) is not ~6x U-WER "
            f"({base.report.u_wer.wer:.2%}) — the lists may not be selecting hard words"
        )
    best = min(biased, key=lambda r: r.report.b_wer.wer)
    drop = -_rel(best.report.b_wer.wer, base.report.b_wer.wer)
    rise = _rel(best.report.u_wer.wer, base.report.u_wer.wer)
    # `drop` is a reduction, so negate it to print the signed *change* — a B-WER
    # that fell by a third must not read as "+34.9%".
    print(f"best config: {best.config} — B-WER {-drop:+.1%}, U-WER {rise:+.1%}")
    if drop < MIN_B_WER_DROP:
        print(
            f"FAIL: B-WER fell only {drop:.1%} (want ≥ {MIN_B_WER_DROP:.0%}) "
            "— biasing is not biting"
        )
        ok = False
    if rise > MAX_U_WER_RISE:
        print(f"FAIL: U-WER rose {rise:.1%} (want ≤ {MAX_U_WER_RISE:.0%}) — over-boosting")
        ok = False

    # B-WER must get *worse* as the list grows: more distractors compete for the
    # same decoder. If it does not, the distractors are not in the tree and the
    # benchmark is measuring nothing — a bug detector, not a quality bar.
    by_n = sorted(
        [
            r
            for r in biased
            if (r.config.alpha, r.config.unk, r.config.compound)
            == (best.config.alpha, best.config.unk, best.config.compound)
        ],
        key=lambda r: r.config.n,
    )
    if len(by_n) > 1:
        wers = [r.report.b_wer.wer for r in by_n]
        if wers != sorted(wers):
            print(f"WARN: B-WER did not degrade monotonically with N: {[f'{w:.2%}' for w in wers]}")
    return ok


# --- Tiers that need no references --------------------------------------------


def run_distractor(
    backend_name: str,
    wavs: list[Path],
    *,
    terms: list[str],
    alpha: float,
    post: list[float],
) -> int:
    """Bias with words known to be absent; any change is a false insertion.

    The strongest no-harm test we have, and the cheapest: it needs no references, so
    it runs on real meeting audio — where the words biasing would damage actually
    live.

    With ``--post`` it prices the *other* layer on the same terms, and this is the
    tier where the two are least alike. Decode-time biasing can only re-rank a token
    the acoustics already support, so an absent word tends to stay absent; fuzzy
    post-correction answers to no acoustics at all — it sees a string within 0.82 of
    a term and rewrites it. Whatever it costs, it costs here, in the column where
    every edit is by construction damage.
    """
    from stenograf.glossary import apply_glossary
    from stenograf.transcript import TranscriptEntry

    def correct(text: str, threshold: float) -> str:
        entry = TranscriptEntry(speaker="", text=text, start=0.0, end=0.0)
        return apply_glossary([entry], glossary=terms, threshold=threshold)[0].text

    backend = create_backend(backend_name)
    backend.load()
    biaser = Biaser(backend, Config(alpha=alpha))

    print(f"{len(terms)} distractor terms, {len(wavs)} files")
    # One column per arm, so the layers are separable: what does biasing alone
    # insert, what does post-correction alone insert, and does stacking them insert
    # more than the sum (a term the boost half-reached, which the fuzzy match then
    # finishes off).
    arms = ["biasing"] + [f"post {t:g}" for t in post] + [f"both {t:g}" for t in post]
    totals = dict.fromkeys(arms, 0)

    for wav in wavs:
        samples = read_pcm16(wav)
        biaser.arm([])
        before = " ".join(s.text for s in backend.transcribe(samples, None))
        biaser.arm(terms)
        biased = " ".join(s.text for s in backend.transcribe(samples, None))

        texts = {"biasing": biased}
        for threshold in post:
            texts[f"post {threshold:g}"] = correct(before, threshold)
            texts[f"both {threshold:g}"] = correct(biased, threshold)

        for arm, after in texts.items():
            spans = bias_score.changed_spans(before, after, normalize)
            surface = bias_score.surface_changes(before, after, normalize)
            totals[arm] += len(spans) + len(surface)
            verdict = "clean" if not spans and not surface else "CHANGED"
            print(f"{wav.name:<24} {arm:<12} {verdict}: {len(spans)} edits, {len(surface)} surface")
            for a, b in spans[:5]:
                print(f"    {a or '∅'} → {b or '∅'}")
            for change in surface[:5]:
                print(f"    {change.before} → {change.after} (surface)")

    biaser._restore()
    backend.unload()

    print("\nFALSE INSERTIONS")
    for arm in arms:
        print(f"  {arm:<12} {totals[arm]}")
    # The verdict is on the full stack — both layers, at the last threshold given —
    # never on a layer in isolation. A layer that only looks clean alone is not clean:
    # biasing can half-reach a term and leave a string the fuzzy matcher then finishes
    # off, so the stack can insert what neither layer inserts by itself.
    stack = f"both {post[-1]:g}" if post else "biasing"
    total = totals[stack]
    print(f"\nSTACK ({stack}): {total} — {'PASS' if total == 0 else 'FAIL'}")
    return 0 if total == 0 else 1


def run_tts(backend_name: str, *, alpha: float) -> int:
    """Tier 5 — can biasing *reach* each error class? Diagnostic, never a metric.

    A class passes when biasing recovers its term under **at least one** of the two
    pronunciations, and the absent-term probes stay clean.

    Not "under both", which is what this tier was originally designed to demand —
    the design assumed both renderings actually contain the term, and measurement
    says they do not. A German acoustic model fed English phonemes can produce audio
    that no longer holds the word at all ("Kafka-Consumer" comes out as something the
    decoder hears as "Kafketon Zoomer"), and — the sharper case — it can produce a
    *different real word*: the English rendering of "Ada" is heard as "Ede", and the
    audio then genuinely says "Ede". Demanding recovery there would be demanding the
    decoder overwrite what was said with what we hoped for. That is a false
    insertion, the very failure the absent-term probes exist to catch. Requiring it
    would be requiring damage.

    Which is the whole reason this tier may not price biasing: the confound does not
    cancel, it *inverts*. So the tier answers only reachability, and every
    non-recovery prints its unbiased hypothesis, because whether the engine mangled
    the word is a two-second read for a human and unknowable for the harness.
    """
    if not TTS_MANIFEST.exists():
        raise SystemExit(f"missing {TTS_MANIFEST} — uv run --group eval eval/bias_tts.py")
    fixtures = json.loads(TTS_MANIFEST.read_text())

    backend = create_backend(backend_name)
    backend.load()
    biaser = Biaser(backend, Config(alpha=alpha))

    by_case: dict[str, list[dict]] = {}
    for fixture in fixtures:
        by_case.setdefault(fixture["case"], []).append(fixture)

    misses: list[str] = []
    reached: dict[str, bool] = {}
    inserted: dict[str, bool] = {}
    print(f"{'case':<20} {'class':<16} {'pron':<6} {'unbiased':<9} {'biased':<8}")
    for case, group in sorted(by_case.items()):
        klass = group[0]["class"]
        absent = group[0].get("absent", False)

        for fixture in sorted(group, key=lambda f: f["pronunciation"]):
            samples = read_pcm16(Path(fixture["wav"]))
            terms = fixture["terms"]

            biaser.arm([])
            before = " ".join(s.text for s in backend.transcribe(samples, None))
            biaser.arm(terms)
            after = " ".join(s.text for s in backend.transcribe(samples, None))

            was = all(term in before for term in terms)
            now = all(term in after for term in terms)
            print(
                f"{case:<20} {klass:<16} {fixture['pronunciation']:<6} "
                f"{'yes' if was else 'no':<9} {'yes' if now else 'no':<8}"
            )
            if absent:
                # These terms are not in the audio: producing any of them is a false
                # insertion, and one is enough to condemn the whole tier.
                inserted[klass] = inserted.get(klass, False) or now
            else:
                reached[klass] = reached.get(klass, False) or now
                if not now:
                    misses.append(f"  {case}/{fixture['pronunciation']}: unbiased “{before}”")

    biaser._restore()
    backend.unload()

    # The verdict is per error *class*, not per fixture, because the only question
    # this tier may ask is the plan's own: can biasing reach this class of error at
    # all? A fixture whose audio does not contain the term cannot answer it — and we
    # have watched the same sentence, in two voices of the same engine, land on
    # either side of that line ("Grafana-Dashboard" survives one voice and comes out
    # of the other as "Grafanada-Sport"). Excluding a rendering that lost the word is
    # not a lowered bar; it is an invalid measurement removed. Every one of them is
    # printed below, with what the engine actually said, so nothing hides here.
    print()
    for klass in sorted(reached):
        print(f"{klass:<16} reachable: {'YES' if reached[klass] else 'NO'}")
    for klass in sorted(inserted):
        print(f"{klass:<16} clean: {'NO — TERM INSERTED' if inserted[klass] else 'YES'}")

    if misses:
        print("\nNot recovered — check whether the engine said the word at all:")
        print("\n".join(misses))

    all_ok = all(reached.values()) and not any(inserted.values())
    print("\nTTS DIAGNOSTIC:", "PASS" if all_ok else "FAIL")
    print("Reachability only — no quality claim may be made from synthetic audio.")
    return 0 if all_ok else 1


# --- Entry point --------------------------------------------------------------


def build_configs(args) -> list[Config]:
    """Baseline plus the configs under test.

    ``--sweep`` is one-factor-at-a-time around the shipped default, not a cartesian
    product: the full grid is 5–10 h of decoding and the questions are independent
    (what is the best boost; is our unk_score right; does compound-tail earn its
    keep). ``--n`` repeated adds the list-size ablation, ``--boosts`` a finer alpha
    grid than the sweep's.

    ``--post`` is the one place a cartesian product *is* affordable, and where it
    earns its keep: it pairs **every** biasing arm with a post-corrected twin, which
    is what settles whether the fuzzy layer contributes anything a better-tuned
    decoder could not. It costs no decoding — the twin re-uses its arm's hypotheses.
    """
    shipped = Config(alpha=args.boost, unk=args.unk, compound=True, n=args.n[0])
    configs = [Config(alpha=0, n=args.n[0])]

    if args.sweep:
        configs += [replace(shipped, alpha=a) for a in (0.5, 1.0, 2.0, 3.0)]
        # The two divergences from NeMo, each at the shipped boost.
        configs += [replace(shipped, unk=0.0), replace(shipped, compound=False)]
        configs += [replace(shipped, n=n) for n in args.n[1:]]
    elif args.boosts:
        configs += [replace(shipped, alpha=a) for a in args.boosts]
    else:
        configs += [replace(shipped, n=n) for n in args.n]

    # The other layer, last so the table reads decode-time → post-hoc → both.
    biased = [c for c in configs if c.alpha and not c.post]
    for threshold in args.post:
        configs.append(Config(alpha=0, n=args.n[0], post=threshold))
        configs += [replace(c, post=threshold) for c in biased]
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=("english", "german", "distractor", "tts"), required=True)
    parser.add_argument("--backend", default="parakeet", help="parakeet | parakeet-onnx")
    parser.add_argument("--n", type=int, nargs="+", default=[100], help="biasing-list size(s)")
    parser.add_argument("--boost", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--boosts",
        type=float,
        nargs="+",
        help="a finer alpha grid than --sweep's; each arm also gets a --post twin",
    )
    parser.add_argument("--unk", type=float, default=DEFAULT_UNK_SCORE)
    parser.add_argument(
        "--sweep", action="store_true", help="ablate boost, unk_score, compound-tail"
    )
    parser.add_argument(
        "--post",
        type=float,
        nargs="*",
        default=None,
        help=f"also score the post-correction layer, at these thresholds "
        f"(bare --post = {DEFAULT_THRESHOLD}); free, it re-decodes nothing",
    )
    parser.add_argument("--limit", type=int, default=SUBSAMPLE, help="0 = the full test set")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="re-decode, ignoring cached hyps")
    parser.add_argument("--wav", nargs="+", type=Path, help="distractor tier: audio to check")
    parser.add_argument("--distractors", type=int, default=500)
    args = parser.parse_args()
    # argparse gives a bare `--post` an empty list, which reads as "no thresholds"
    # rather than "the shipped one". Absent stays absent (None -> []).
    args.post = [DEFAULT_THRESHOLD] if args.post == [] else (args.post or [])

    if args.tier == "tts":
        return run_tts(args.backend, alpha=args.boost)

    if args.tier == "distractor":
        wavs = args.wav or sorted((Path(__file__).parent / "audio").glob("*.wav"))
        pool_path = DE_DIR / "words" / "all_rare_words.txt"
        if not pool_path.exists():
            raise SystemExit(
                f"missing {pool_path} — uv run --group eval eval/bias_data.py --fetch german"
            )
        pool = pool_path.read_text(encoding="utf-8").split()
        terms = random.Random(args.seed).sample(pool, args.distractors)
        return run_distractor(args.backend, wavs, terms=terms, alpha=args.boost, post=args.post)

    rows = evaluate(
        args.tier,
        build_configs(args),
        limit=args.limit,
        seed=args.seed,
        backend_name=args.backend,
        force=args.force,
    )
    table = format_table(args.tier, rows, limit=args.limit, backend_name=args.backend)
    print()
    print(table)
    report = Path(str(REPORT).format(tier=args.tier))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(table)
    print(f"wrote {report}")

    ok = gate(rows) if args.tier == "english" else True
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
