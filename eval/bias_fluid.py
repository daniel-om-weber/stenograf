"""FluidAudio on our benchmark — the head-to-head against TypeWhisper's engine.

TypeWhisper is the closest thing to a competitor that ships glossary support on the
*same acoustic model we do* (Parakeet TDT 0.6b v3), so it is the one comparison where
the model can be held fixed and the biasing *mechanism* is the only variable. Its
Parakeet plugin does not implement biasing itself: it calls FluidAudio (Apache-2.0),
which is what this driver runs.

**The mechanisms differ in kind, not degree.** We bias the decoder *while it
transcribes* — a boosting tree over the token logits inside the greedy TDT loop
(``stenograf.asr.biasing``). FluidAudio lets the TDT decoder finish, then runs a
*second* acoustic model over the same audio (Parakeet CTC-110m), keyword-spots the
vocabulary against its per-frame posteriors, and rewrites words in the completed
transcript. Post-decode, alignment-anchored, acoustically-scored replacement. It is a
far better instrument than the find-and-replace every other dictation app ships, and
it is still downstream of the decision it is trying to influence.

**English only, and that is their constraint, not a choice of ours.** FluidAudio's
spotter is ``parakeet-ctc-110m``, whose tokenizer holds 1024 tokens and *zero*
non-ASCII ones (our TDT v3 vocab: 8192 tokens, 194 carrying umlauts/ß). German terms
tokenize into ``<unk>`` holes, and ``loadWithCtcTokens`` drops only *empty* token
lists — so a poisoned term is silently kept and scored as noise. TypeWhisper pairs
this English-only spotter with the multilingual transcriber with no language check.
Our German tier therefore has no opponent to run: it is not a number they lose, it is
a capability they do not have.

**Fairness.** Their encoder is CoreML int8, ours is MLX fp32, so the two baselines are
*different models* and their absolute WERs are not comparable. Only the relative move
each mechanism makes against **its own** unbiased baseline is — which is why this runs
their engine twice, with and without the vocabulary, and reports Δ against their
baseline, never against ours. Their knobs are set the way TypeWhisper actually runs
them: ``rescorerConfig(forVocabSize:)`` overrides the documented cbw 3.0 / minSim 0.52
to **4.5 / 0.55** at our 100-term lists, and ``marginSeconds`` is TypeWhisper's one
hardcoded knob (0.5, against FluidAudio's own 0.10 default).

**What it found (2026-07-13, 500 utts, N=100).** Two things, and the second matters
more than the first.

*As TypeWhisper ships it, the vocabulary destroys the transcript.* B-WER −75.3 %,
which looks like it laps us — bought with U-WER **+305.6 %** and **375 false
insertions**, altering 287 of 500 utterances. The rewrites are not subtle
(``glowing``→``unloving``, ``pound``→``compound``, ``words``→``awards``) and every one
we sampled snapped a correct common word onto a *distractor* — a term the benchmark
put in the list precisely because it is **not** spoken. It is not a tuning accident:
FluidAudio's own defaults give 374 and their documented ``cbw 3.0`` gives 362.

*But the mechanism is fine; the shipped configuration is not.* Forced to
``minSimilarity 0.85``, the same engine lands at **B-WER −32.2 %, U-WER +2.4 %, 8 false
insertions** — real parity with our −34.9 % / +0.0 % / 2. Post-decode CTC rescoring is
a legitimate instrument, and on accuracy alone this benchmark does not separate the two
approaches. What separates them is everything around the number: we get ours in one
pass, in-loop, with no second model, in a decoder that also streams, in a language they
cannot process at all.

Two defaults do the damage, and both are worst exactly where real users live:

- ``rescorerConfig(forVocabSize:)`` sets minSimilarity **0.60 above 100 terms, 0.55 for
  11–100, and 0.50 at ≤10** — the *smaller* the glossary, the *looser* the match. A real
  meeting glossary is 10–30 terms, i.e. their loosest setting.
- the **spotter rescue** ("acoustic single-word rescue"), on by default, is what wrecks
  small lists: with an oracle list of only words genuinely spoken it produces 762 false
  insertions and U-WER +1407 %, and ``--vocab-disable-spotter-rescue`` alone drops that
  to 104 / +72.6 %. minSimilarity does not touch it — 0.85 leaves the collapse intact.

**Read this fairly.** is21's "rare words" are ordinary English words (``frail``, ``idly``,
``holiness``), not the entities and jargon FluidAudio is built for — their published
99.3 % precision on earnings calls is plausibly true in that domain, and this benchmark
is harsher on their design than their target use case is. The claim here is bounded:
*on the standard contextual-biasing benchmark, at its standard list size, the
configuration TypeWhisper ships is destructive, and the mechanism underneath it is not.*

Usage:
    swift build -c release --product fluidaudiocli          # in a FluidAudio checkout
    uv run --group eval eval/bias_fluid.py --cli /path/to/.build/release/fluidaudiocli
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bias_score  # noqa: E402
from bias import SEED, SUBSAMPLE, load_audio, ref_path, subsample  # noqa: E402
from bias_data import BIAS_DIR  # noqa: E402
from common import OUT_DIR  # noqa: E402
from score import normalize  # noqa: E402

WORK = BIAS_DIR / "fluid"
REPORT = OUT_DIR / "bias-report-fluidaudio.md"

VOCAB_MARGIN = "0.5"
"""TypeWhisper's one hardcoded rescorer knob (``ParakeetPlugin.swift``). FluidAudio's
own default is 0.10; the CLI help advertising 0.5 is a doc bug. We match the app."""

OUR_HYPS = BIAS_DIR / "hyp" / "english"
"""Our arms are already decoded and cached by ``bias.py --tier english`` — the same
500 utterances, the same lists. Re-using them is not a shortcut: it guarantees both
systems are scored on byte-identical references by byte-identical code."""


@dataclass
class Arm:
    label: str
    hyps: dict[str, str]


def export(uttids: list[str], refs: dict[str, bias_score.RefUtt], work: Path) -> None:
    """Write each utterance's audio and its own biasing list to disk.

    One vocabulary file per utterance, because the is21 protocol gives every
    utterance its own list (its rare words + the shared distractor pool). Feeding one
    merged list would be a different, easier benchmark.
    """
    import soundfile as sf

    wav_dir, vocab_dir = work / "wav", work / "vocab"
    wav_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    oracle_dir = work / "vocab_oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    missing = [u for u in uttids if not (wav_dir / f"{u}.wav").exists()]
    if missing:
        audio = load_audio("english", set(missing))
        for uttid in missing:
            sf.write(wav_dir / f"{uttid}.wav", audio[uttid], 16_000, subtype="PCM_16")
    for uttid in uttids:
        path = vocab_dir / f"{uttid}.txt"
        if not path.exists():
            path.write_text("\n".join(refs[uttid].biasing_words) + "\n", encoding="utf-8")
        # The oracle list: the utterance's own rare words, nothing else.
        path = oracle_dir / f"{uttid}.txt"
        if not path.exists():
            path.write_text("\n".join(refs[uttid].rare_words) + "\n", encoding="utf-8")
    print(f"exported {len(uttids)} wavs + vocab files to {work}")


ARMS: dict[str, tuple[str | None, list[str]]] = {
    "baseline": (None, []),
    # TypeWhisper's actual configuration: FluidAudio's own `rescorerConfig(forVocabSize:)`
    # picks cbw 4.5 / minSimilarity 0.55 for a 100-term list, and the app overrides
    # marginSeconds to 0.5 (its one hardcoded knob).
    "typewhisper": ("vocab", ["--vocab-margin", VOCAB_MARGIN]),
    # FluidAudio's own defaults, untouched — margin 0.10, weight still auto. The control
    # for "is the damage TypeWhisper's override, or the mechanism?".
    "fluid-default": ("vocab", []),
    # And their *documented* weight, the gentlest setting the library advertises. If the
    # collapse survives this, it is not a tuning accident.
    "cbw-3.0": ("vocab", ["--vocab-cbw", "3.0"]),
    # The oracle list: *only* words the utterance actually contains, no distractors.
    # This is not a benchmark condition — no user can supply a glossary of exactly what
    # is about to be said — it is the diagnostic that separates "the mechanism is broken"
    # from "the mechanism assumes every term you list is spoken". The distinction decides
    # what the result means, because in a real meeting most of a glossary is *not* said in
    # any given utterance: the distractor condition IS the production condition.
    "oracle": ("vocab_oracle", ["--vocab-margin", VOCAB_MARGIN]),
    # Their auto-config is the suspect: `rescorerConfig(forVocabSize:)` hands a list of
    # >100 terms minSimilarity 0.60, 11–100 → 0.55, and **≤10 → 0.50** — so the *smaller*
    # the glossary, the *looser* the matching, which is backwards for the case that
    # matters (a real meeting glossary is ~10–30 terms). These arms take the knob away
    # from their auto-config and hand the mechanism its best possible shot.
    "min-sim-0.70": ("vocab", ["--vocab-min-similarity", "0.70", "--vocab-margin", VOCAB_MARGIN]),
    "min-sim-0.85": ("vocab", ["--vocab-min-similarity", "0.85", "--vocab-margin", VOCAB_MARGIN]),
    "oracle-0.85": (
        "vocab_oracle",
        ["--vocab-min-similarity", "0.85", "--vocab-margin", VOCAB_MARGIN],
    ),
    # The oracle list stays wrecked even at minSimilarity 0.85, so the threshold is not
    # what breaks small lists. The remaining suspect is the *spotter rescue* — an
    # acoustic single-word path that is on by default. These isolate it.
    "oracle-norescue": (
        "vocab_oracle",
        ["--vocab-disable-spotter-rescue", "--vocab-margin", VOCAB_MARGIN],
    ),
    "best-case": (
        "vocab",
        [
            "--vocab-min-similarity",
            "0.85",
            "--vocab-disable-spotter-rescue",
            "--vocab-margin",
            VOCAB_MARGIN,
        ],
    ),
}
"""Every arm we give them. Four shots at the target, not one: a benchmark that condemns
a system under a single configuration it never ships is not evidence, and one that
cannot say *why* it failed is not an explanation."""


def transcribe(cli: Path, work: Path, uttids: list[str], *, arm: str, jobs: int) -> dict[str, str]:
    """Run FluidAudio over every utterance under one arm of :data:`ARMS`.

    ``fluidaudiocli transcribe`` takes one file per process and reloads both CoreML
    model sets on every invocation, so this is embarrassingly parallel and pays the
    load cost N times regardless. A thread pool hides the load latency; the JSON
    output doubles as the cache, so an interrupted run resumes for free.
    """
    out_dir = work / "out" / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(uttid: str) -> None:
        out = out_dir / f"{uttid}.json"
        if out.exists():
            return
        cmd = [
            str(cli),
            "transcribe",
            str(work / "wav" / f"{uttid}.wav"),
            "--language",
            "en",
            "--output-json",
            str(out),
        ]
        vocab_dir, flags = ARMS[arm]
        if vocab_dir is not None:
            cmd += ["--custom-vocab", str(work / vocab_dir / f"{uttid}.txt"), *flags]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out.exists():
            raise SystemExit(
                f"fluidaudiocli failed on {uttid} (rc={result.returncode})\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )

    started = time.perf_counter()
    todo = [u for u in uttids if not (out_dir / f"{u}.json").exists()]
    print(f"  {arm}: {len(uttids) - len(todo)} cached, {len(todo)} to run …", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for index, _ in enumerate(pool.map(run, todo), start=1):
            if index % 25 == 0:
                rate = index / (time.perf_counter() - started)
                print(f"    {index}/{len(todo)} ({rate:.1f}/s)", flush=True)

    hyps = {}
    for uttid in uttids:
        hyps[uttid] = json.loads((out_dir / f"{uttid}.json").read_text())["text"].strip()
    return hyps


def ours(name: str, limit: int, seed: int) -> dict[str, str] | None:
    path = OUR_HYPS / f"parakeet.{name}.{limit}s{seed}.tsv"
    return bias_score.read_hyps(path) if path.exists() else None


def table(refs: dict[str, bias_score.RefUtt], arms: list[Arm]) -> str:
    """Each system against **its own** unbiased baseline — never against the other's.

    The two engines run different quantizations of the same model, so the absolute
    columns are context, not the comparison. The Δ columns are the comparison.
    """
    lines = [
        "## English — stenograf (in-loop logit boosting) vs FluidAudio (post-decode CTC rescoring)",
        "",
        f"is21 test-clean, {len(refs)} pinned utterances, N=100 biasing lists.",
        "",
        "| system | B-WER | ΔB vs own baseline | U-WER | ΔU | recall (loose) | false ins. |",
        "|---|---|---|---|---|---|---|",
    ]
    base: dict[str, bias_score.Report] = {}
    for arm in arms:
        report = bias_score.score(refs, arm.hyps, normalize=normalize)
        system = arm.label.split(" — ")[0]
        if "unbiased" in arm.label:
            base[system] = report
        ref = base.get(system)
        db = du = "—"
        if ref is not None and "unbiased" not in arm.label:
            db = f"{(report.b_wer.wer - ref.b_wer.wer) / ref.b_wer.wer:+.1%}"
            du = f"{(report.u_wer.wer - ref.u_wer.wer) / ref.u_wer.wer:+.1%}"
        lines.append(
            f"| {arm.label} | {report.b_wer.wer:.2%} | {db} | {report.u_wer.wer:.2%} | {du} "
            f"| {report.recall:.1%} ({report.recall_loose:.1%}) | {report.false_insertions} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", type=Path, required=True, help="path to fluidaudiocli")
    parser.add_argument("--limit", type=int, default=SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=100, help="biasing-list size")
    parser.add_argument("--jobs", type=int, default=4, help="concurrent fluidaudiocli processes")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()

    if not args.cli.exists():
        raise SystemExit(f"no fluidaudiocli at {args.cli} — swift build -c release first")

    path = ref_path("english", args.n)
    if not path.exists():
        raise SystemExit(f"missing {path} — uv run --group eval eval/bias_data.py --fetch is21")
    refs_all = bias_score.read_refs(path)
    uttids = subsample(refs_all, args.limit, args.seed, "english")
    refs = {u: refs_all[u] for u in uttids}

    export(uttids, refs, args.work)

    arms = []
    for label, hyps in (
        ("stenograf — unbiased", ours("baseline", args.limit, args.seed)),
        ("stenograf — in-loop boosting (boost=1.0)", ours("a1-u1-c1-n100", args.limit, args.seed)),
    ):
        if hyps is None:
            raise SystemExit(
                "our English arms are not cached — run: "
                "uv run --group eval eval/bias.py --tier english"
            )
        arms.append(Arm(label, hyps))

    print("running FluidAudio (TypeWhisper's engine) …")
    labels = {
        "baseline": "FluidAudio — unbiased",
        "typewhisper": "FluidAudio — vocabulary, as TypeWhisper runs it (cbw 4.5, margin 0.5)",
        "fluid-default": "FluidAudio — vocabulary, FluidAudio's own defaults (margin 0.10)",
        "cbw-3.0": "FluidAudio — vocabulary, documented weight (cbw 3.0)",
        "oracle": "FluidAudio — ORACLE list (only words actually spoken; diagnostic)",
        "min-sim-0.70": "FluidAudio — minSimilarity forced to 0.70 (auto-config overridden)",
        "min-sim-0.85": "FluidAudio — minSimilarity forced to 0.85 (auto-config overridden)",
        "oracle-0.85": "FluidAudio — ORACLE list + minSimilarity 0.85",
        "oracle-norescue": "FluidAudio — ORACLE list, spotter rescue OFF",
        "best-case": "FluidAudio — minSimilarity 0.85 + spotter rescue OFF (their best case)",
    }
    for arm in ARMS:
        hyps = transcribe(args.cli, args.work, uttids, arm=arm, jobs=args.jobs)
        arms.append(Arm(labels[arm], hyps))

    out = table(refs, arms)
    print()
    print(out)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(out, encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
