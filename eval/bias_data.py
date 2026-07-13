"""Fetch and derive the contextual-biasing benchmarks — no hand labeling anywhere.

Two benchmarks, one format (the is21 reference TSV: utterance id, text, the
reference's own rare words, the biasing list actually fed to the model), therefore
one scorer and one driver:

**English** is downloaded. ``facebookresearch/fbai-speech/is21_deep_bias`` (MIT)
publishes the per-utterance biasing lists for LibriSpeech at N = 100/500/1000/2000,
the 5k common-word list, the 209k distractor pool — and, decisively, the 44
hypothesis files and the result files its own scorer produced for the paper. Those
are the oracle ``tests/test_eval_bias.py`` pins our scorer against. English is in
scope for that one reason: it is the only language whose numbers are published. We
do not ship English transcription quality.

**German is derived**, because no German biasing benchmark exists (searched; it
plainly does not). Same recipe as the English artifact, from data that ships free:
MLS publishes its *training transcripts* as a standalone 92 MB text file, so a
frequency list costs one download and no audio. Top-5k words by training count are
"common"; the complement is the rare pool; each test utterance's list is its own
rare words ∪ N distractors drawn from that pool — exactly the is21 construction,
verified against their files (a list of N is len(own_rare) + N, and the draw is per
utterance).

The seed is pinned, so the benchmark is the same benchmark on every machine and
every run; a sweep whose lists moved underneath it would be comparing nothing.

Usage:
    uv run --group eval eval/bias_data.py --fetch is21     # English + the oracle
    uv run --group eval eval/bias_data.py --fetch german   # derive the German benchmark
    uv run --group eval eval/bias_data.py --fetch all --sizes 100 500 1000 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bias_score import RefUtt, write_refs  # noqa: E402
from common import OUT_DIR  # noqa: E402

BIAS_DIR = OUT_DIR / "bias"
IS21_DIR = BIAS_DIR / "is21"
DE_DIR = BIAS_DIR / "de"

IS21_RAW = "https://raw.githubusercontent.com/facebookresearch/fbai-speech/main/is21_deep_bias"
IS21_API = "https://api.github.com/repos/facebookresearch/fbai-speech/contents/is21_deep_bias"

MLS_REPO = "facebook/multilingual_librispeech"
MLS_TRAIN_TRANSCRIPTS = "data/mls_german/train/transcripts.txt"
MLS_TEST_TRANSCRIPTS = "data/mls_german/test/transcripts.txt"

COMMON_WORDS = 5_000
"""Words this frequent in training are "common"; the rest are rare. is21's cut,
kept for German so the two benchmarks are the same experiment in two languages."""

SEED = 20260713
"""Pinned: the distractor draw must not move between runs, or a sweep compares
list draws instead of settings."""

SIZES = (100, 500, 1000, 2000)
SPLITS = ("test-clean", "test-other")


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {dest.relative_to(BIAS_DIR)}", flush=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310 — a pinned https URL
        dest.write_bytes(response.read())
    return dest


def _list_dir(name: str) -> list[str]:
    with urllib.request.urlopen(f"{IS21_API}/{name}") as response:  # noqa: S310
        return [entry["name"] for entry in json.load(response)]


def fetch_is21(
    sizes: tuple[int, ...] = (100,), *, oracle: bool = True, force: bool = False
) -> None:
    """Download the English lists, and the published hyp/result files behind them.

    Only the N=100 reference is needed to *score* anything: the rare-word column —
    the B set — is byte-identical across list sizes, and only the fed list grows
    (verified against their files). The larger references are downloaded solely to
    *drive* the model at that N, so a scoring-only or oracle-only run never pays for
    the 62 MB one.
    """
    print(f"is21_deep_bias → {IS21_DIR}")
    for name in ("common_words_5k.txt", "all_rare_words.txt"):
        _download(f"{IS21_RAW}/words/{name}", IS21_DIR / "words" / name, force=force)

    for split in SPLITS:
        for size in sorted({100, *sizes}):  # 100 always: it carries the B set
            name = f"{split}.biasing_{size}.tsv"
            _download(f"{IS21_RAW}/ref/{name}", IS21_DIR / "ref" / name, force=force)

    if not oracle:
        return
    # The correctness oracle: their hypotheses, and the numbers their scorer got.
    for kind in ("hyp", "results"):
        for name in _list_dir(kind):
            _download(f"{IS21_RAW}/{kind}/{name}", IS21_DIR / kind / name, force=force)


def _read_transcripts(path: Path) -> dict[str, str]:
    """MLS transcripts: ``<utterance id>\\t<text>``, already lowercased and
    unpunctuated — the same surface form as LibriSpeech's references."""
    utts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        uttid, _, text = line.partition("\t")
        if text.strip():
            utts[uttid] = text.strip()
    return utts


def build_lists(
    utts: dict[str, str],
    common: set[str],
    pool: list[str],
    size: int,
    seed: int,
) -> list[RefUtt]:
    """The is21 construction: per utterance, ``own rare words ∪ size distractors``.

    Distractors are drawn per utterance, not once for the whole set — that is what
    makes N a real difficulty knob (the tree has to hold N competing phrases while
    decoding *this* utterance), and it is what their published files do.

    The rng is re-seeded per list size and stepped per utterance, so the lists for
    N=100 are not a prefix of the lists for N=500 and the two sizes are independent
    draws — while both stay reproducible from ``seed``.
    """
    rng = random.Random(f"{seed}:{size}")
    refs = []
    for uttid, text in utts.items():
        rare = sorted({word for word in text.split() if word not in common})
        distractors = rng.sample(pool, size)
        refs.append(
            RefUtt(
                uttid=uttid,
                text=text,
                rare_words=tuple(rare),
                biasing_words=tuple(sorted(set(rare) | set(distractors))),
            )
        )
    return refs


def derive_german(
    sizes: tuple[int, ...] = SIZES, *, seed: int = SEED, force: bool = False
) -> None:
    """Build the German benchmark from MLS's own transcripts. No audio downloaded."""
    from huggingface_hub import hf_hub_download

    print(f"MLS German → {DE_DIR}")
    print("  ↓ train transcripts (92 MB, no audio)", flush=True)
    train_path = Path(
        hf_hub_download(MLS_REPO, MLS_TRAIN_TRANSCRIPTS, repo_type="dataset")
    )
    test_path = Path(hf_hub_download(MLS_REPO, MLS_TEST_TRANSCRIPTS, repo_type="dataset"))

    counts: Counter[str] = Counter()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        _, _, text = line.partition("\t")
        counts.update(text.split())

    common = {word for word, _ in counts.most_common(COMMON_WORDS)}
    # The distractor pool, sorted for a reproducible sample() regardless of how the
    # Counter happened to hash. Rare *by frequency*, never by capitalization: every
    # German noun is capitalized, so every case-based proper-noun heuristic in the
    # English literature produces garbage here.
    pool = sorted(set(counts) - common)

    words_dir = DE_DIR / "words"
    words_dir.mkdir(parents=True, exist_ok=True)
    (words_dir / "common_words_5k.txt").write_text(
        "\n".join(word for word, _ in counts.most_common(COMMON_WORDS)) + "\n"
    )
    (words_dir / "all_rare_words.txt").write_text("\n".join(pool) + "\n")

    utts = _read_transcripts(test_path)
    tokens = sum(len(text.split()) for text in utts.values())
    rare_tokens = sum(
        sum(1 for word in text.split() if word not in common) for text in utts.values()
    )

    for size in sizes:
        refs = build_lists(utts, common, pool, size, seed)
        path = DE_DIR / "ref" / f"mls-de-test.biasing_{size}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_refs(path, refs)
        print(f"  → {path.relative_to(BIAS_DIR)}")

    print(
        f"  {len(utts)} utts, {tokens} tokens, vocab {len(counts)} "
        f"(train), distractor pool {len(pool)}"
    )
    # The density that makes German the better biasing benchmark, not the worse
    # one: compounding fattens the vocabulary tail, so more of the test set is
    # rare — and rare is exactly what biasing is for.
    print(f"  rare-token density: {rare_tokens / tokens:.1%} of test tokens")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", choices=("is21", "german", "all"), default="all")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-oracle", action="store_true", help="skip is21's hyp/result files")
    parser.add_argument("--force", action="store_true", help="re-download / rebuild")
    args = parser.parse_args()

    sizes = tuple(args.sizes)
    if args.fetch in ("is21", "all"):
        fetch_is21(sizes, oracle=not args.no_oracle, force=args.force)
    if args.fetch in ("german", "all"):
        derive_german(sizes, seed=args.seed, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
