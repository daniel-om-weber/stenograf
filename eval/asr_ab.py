"""Two speech models, the same audio, human references: which one is right more often.

Built to decide whether Parakeet Ultra replaces Parakeet-v3 (2026-11-26,
README.md "Parakeet Ultra"); it compares any two models the Parakeet backends
can load. Three sets, each answering what the others cannot:

``ami``
    English meetings with human transcripts: all four headsets of the twelve
    ES2004 / IS1009 / TS3003 sessions, which AMI's standard partition holds out
    for testing. Each speaker's words are cut into windows at silences over a
    second, so both models decode identical audio and crosstalk between the
    speaker's own turns never reaches them.
``fleurs``
    German with human transcripts — read Wikipedia sentences, the only German
    ground truth there is. Read speech flatters every model; compare the two
    columns with each other, never with meeting numbers.
``own``
    The private segments of ``manifest.json`` through the finalize path
    (``windows.decode_windowed``). No references exist, so this counts where
    the two models differ, and for each such place which one Whisper large-v3
    wrote too — triage only: agreement between models is not accuracy
    (2026-07-26, "consensus"), and Whisper taught Parakeet-v3.

A difference whose two sides are the same string once spaces are dropped
("d v d" / "dvd", "123456" / "1 2 3 4 5 6") is a spelling convention, not a
recognition error, and the ``fair`` score leaves it out: Ultra writes digits and
joined acronyms where v3 spells them, and strict scoring charges that to Ultra
against references that happen to spell.

Usage:
    uv run --group eval eval/asr_ab.py decode v3 --model mlx-community/parakeet-tdt-0.6b-v3
    uv run --group eval eval/asr_ab.py decode ultra            # the shipped default
    uv run --group eval eval/asr_ab.py decode ultra-onnx --backend parakeet-onnx --sets fleurs
    uv run --group eval eval/asr_ab.py score v3 ultra
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
import tarfile
import time
import unicodedata
import wave
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUDIO_DIR, OUT_DIR, load_manifest  # noqa: E402

AB_DIR = OUT_DIR / "asr-ab"
FLEURS_DIR = AUDIO_DIR / "fleurs-de"
FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/de_de"
AMI_TEST = [group + session for group in ("ES2004", "IS1009", "TS3003") for session in "abcd"]
SAMPLE_RATE = 16000

GAP_S = 1.0
"""A silence longer than this ends a window: the speaker's turn is over, and
what the headset picks up next is somebody else."""
MAX_SPAN_S = 25.0
PAD_S = 0.25

Row = dict[str, object]


# --- decode -----------------------------------------------------------------


def ami_windows() -> Iterator[tuple[str, str, Path, list[list[tuple[float, float, str]]]]]:
    """(meeting, agent, headset wav, windows of (start, end, word)) per speaker."""
    import ami

    annotations = ami._ami_annotations()
    agents = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    with zipfile.ZipFile(ami.RAW_DIR / ami.AMI_ANNOTATIONS_ZIP) as archive:
        for meeting in AMI_TEST:
            for agent, (headset, _name) in sorted(agents[meeting].items()):
                root = ElementTree.fromstring(archive.read(f"words/{meeting}.{agent}.words.xml"))
                words = sorted(
                    (float(w.get("starttime", 0)), float(w.get("endtime", 0)), w.text.strip())
                    for w in root.iter("w")
                    if not w.get("punc") and w.get("starttime") and w.get("endtime") and w.text
                )
                windows: list[list[tuple[float, float, str]]] = []
                for word in words:
                    current = windows[-1] if windows else None
                    if (
                        current is None
                        or word[0] - current[-1][1] > GAP_S
                        or word[1] - current[0][0] > MAX_SPAN_S
                    ):
                        windows.append([])
                    windows[-1].append(word)
                yield meeting, agent, ami._ami_headset(meeting, headset), windows


def decode_ami(transcribe: Callable[[np.ndarray], str]) -> list[Row]:
    rows: list[Row] = []
    for meeting, agent, wav, windows in ami_windows():
        with wave.open(str(wav)) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        for i, window in enumerate(windows):
            start = max(0, int((window[0][0] - PAD_S) * SAMPLE_RATE))
            end = min(len(pcm), int((window[-1][1] + PAD_S) * SAMPLE_RATE))
            rows.append(
                {
                    "id": f"{meeting}.{agent}.{i}",
                    "group": f"{meeting[:6]}.{agent}",
                    "seconds": window[-1][1] - window[0][0],
                    "ref": " ".join(word for _, _, word in window),
                    "hyp": transcribe(pcm[start:end]),
                }
            )
        print(f"  {meeting} {agent}: {len(windows)} windows", flush=True)
    return rows


def fetch_fleurs() -> list[tuple[str, str, str]]:
    """(sentence id, wav name, transcript) of FLEURS German test, downloaded once."""
    from urllib.request import urlretrieve

    FLEURS_DIR.mkdir(parents=True, exist_ok=True)
    table = FLEURS_DIR / "test.tsv"
    if not table.exists():
        urlretrieve(f"{FLEURS_URL}/test.tsv", table)
    if not (FLEURS_DIR / "test").is_dir():
        archive = FLEURS_DIR / "test.tar.gz"
        urlretrieve(f"{FLEURS_URL}/audio/test.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(FLEURS_DIR, filter="data")
        archive.unlink()
    clips: dict[str, tuple[str, str, str]] = {}
    with table.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            clips.setdefault(row[1], (row[0], row[1], row[2]))
    return list(clips.values())


def decode_fleurs(transcribe: Callable[[np.ndarray], str]) -> list[Row]:
    import soundfile

    rows: list[Row] = []
    for sentence, name, transcript in fetch_fleurs():
        samples, rate = soundfile.read(str(FLEURS_DIR / "test" / name), dtype="float32")
        assert rate == SAMPLE_RATE and samples.ndim == 1
        rows.append({"id": name, "group": sentence, "ref": transcript, "hyp": transcribe(samples)})
    return rows


def decode_own(asr, out_dir: Path) -> None:
    from windows import decode_windowed, read_mono16k

    from stenograf import assets
    from stenograf.vad import SileroVAD

    vad = SileroVAD(assets.fetch(assets.SILERO_VAD))
    out_dir.mkdir(parents=True, exist_ok=True)
    for segment in load_manifest():
        if not segment.wav_path.exists():
            continue
        segments, windows = decode_windowed(
            asr, vad, read_mono16k(segment.wav_path), segment.language
        )
        record = {"language": segment.language, "segments": segments, "windows": windows}
        (out_dir / f"{segment.id}.json").write_text(json.dumps(record, ensure_ascii=False))
        print(f"\n  {segment.id}: {len(windows)} windows", flush=True)


def decode(label: str, backend: str, model: str | None, sets: list[str]) -> None:
    from stenograf.asr import create_backend

    asr = create_backend(backend, **({"model_id": model} if model else {}))
    asr.load()
    out = AB_DIR / label
    out.mkdir(parents=True, exist_ok=True)

    def transcribe(samples: np.ndarray) -> str:
        return " ".join(segment.text for segment in asr.transcribe(samples, None))

    for name, run in (("ami", decode_ami), ("fleurs", decode_fleurs)):
        if name in sets:
            started = time.perf_counter()
            rows = run(transcribe)
            (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
            print(f"{label} {name}: {len(rows)} clips in {time.perf_counter() - started:.0f} s")
    if "own" in sets:
        decode_own(asr, out / "own")


# --- score ------------------------------------------------------------------


def difference_regions(ref: list[str], hyp: list[str]) -> list[tuple[int, int, int, int]]:
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def errors(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(strict, fair) word errors of ``hyp`` against ``ref``."""
    import jiwer

    if not ref or not hyp:
        return (max(len(ref), len(hyp)),) * 2
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    strict = out.substitutions + out.deletions + out.insertions
    convention = 0
    region: list[int] | None = None
    for chunk in [*out.alignments[0], None]:
        if chunk is not None and chunk.type != "equal":
            cost = max(
                chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx
            )
            if region is None:
                region = [chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, 0, 0]
            region[1], region[3], region[4] = chunk.ref_end_idx, chunk.hyp_end_idx, region[4] + cost
        elif region is not None:
            r0, r1, h0, h1, cost = region
            if "".join(ref[r0:r1]) == "".join(hyp[h0:h1]):
                convention += cost
            region = None
    return strict, strict - convention


def score_references(name: str, a: str, b: str, normalize: Callable[[str], str]) -> None:
    rows = {label: json.loads((AB_DIR / label / f"{name}.json").read_text()) for label in (a, b)}
    by_id = {label: {row["id"]: row for row in rows[label]} for label in (a, b)}
    ids = sorted(set(by_id[a]) & set(by_id[b]))
    counts: dict[str, dict[str, tuple[int, int, int]]] = {a: {}, b: {}}
    for label in (a, b):
        for i in ids:
            row = by_id[label][i]
            ref = normalize(row["ref"]).split()
            counts[label][i] = (*errors(ref, normalize(row["hyp"]).split()), len(ref))

    def wer(label: str, chosen: list[str], column: int) -> float:
        if not chosen:
            return 0.0
        table = np.array([counts[label][i] for i in chosen])
        return 100 * table[:, column].sum() / max(1, table[:, 2].sum())

    # Resampled by speaker (AMI) or sentence (FLEURS, read by several people):
    # clips of one group are not independent draws.
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(by_id[a][i]["group"], []).append(i)
    names = sorted(groups)
    rng = np.random.default_rng(0)
    draws = [
        [i for group in rng.choice(names, len(names)) for i in groups[group]] for _ in range(2000)
    ]

    def line(title: str, keep: Callable[[str], bool], column: int) -> None:
        chosen = [i for i in ids if keep(i)]
        deltas = [
            wer(b, kept, column) - wer(a, kept, column)
            for kept in ([i for i in draw if keep(i)] for draw in draws)
        ]
        low, high = np.percentile(deltas, [2.5, 97.5])
        words = sum(counts[a][i][2] for i in chosen)
        wer_a, wer_b = wer(a, chosen, column), wer(b, chosen, column)
        print(
            f"  {title:12s} {words:6d} words  {a} {wer_a:5.2f}  {b} {wer_b:5.2f}  "
            f"{wer_b - wer_a:+.2f} [{low:+.2f}, {high:+.2f}]"
        )

    print(f"{name}: {len(ids)} clips, {len(names)} groups; word error rate %, {b} minus {a} [95 %]")
    line("strict", lambda i: True, 0)
    line("fair", lambda i: True, 1)
    if name == "ami":
        for group in ("ES2004", "IS1009", "TS3003"):
            line(group, lambda i, group=group: i.startswith(group), 1)
        for low, high in ((0, 2), (2, 10), (10, 100)):
            line(
                f"{low}-{high} s",
                lambda i, low=low, high=high: low <= by_id[a][i]["seconds"] < high,
                1,
            )
    for label in (a, b):
        hyps = [by_id[label][i]["hyp"] for i in ids]
        foreign = sum(
            any(unicodedata.name(c, "").startswith(("CYRILLIC", "GREEK")) for c in hyp)
            for hyp in hyps
        )
        print(
            f"  {label}: {sum('<unk>' in hyp for hyp in hyps)} clips with <unk>, "
            f"{foreign} in another script, {sum(not hyp.strip() for hyp in hyps)} empty"
        )


def score_own(a: str, b: str) -> None:
    from score import normalize

    print(f"own segments: words, places the two differ (spelling only), Whisper sides with {a}/{b}")
    totals = np.zeros(5, dtype=int)
    for path in sorted((AB_DIR / a / "own").glob("*.json")):
        other = AB_DIR / b / "own" / path.name
        if not other.exists():
            continue
        words = [
            normalize(
                " ".join(
                    w["text"] for s in json.loads(p.read_text())["segments"] for w in s["words"]
                )
            ).split()
            for p in (path, other)
        ]
        regions = difference_regions(*words)
        real = [
            r for r in regions if "".join(words[0][r[0] : r[1]]) != "".join(words[1][r[2] : r[3]])
        ]
        sides = [0, 0]
        pivot = OUT_DIR / "whisper" / path.name
        if pivot.exists():
            whisper = f" {normalize(json.loads(pivot.read_text())['text'])} "
            for i1, i2, j1, j2 in real:
                # One word of context each side, or a deletion matches anywhere.
                found = [
                    f" {' '.join(words[0][max(0, i1 - 1) : i2 + 1])} " in whisper,
                    f" {' '.join(words[1][max(0, j1 - 1) : j2 + 1])} " in whisper,
                ]
                if found[0] != found[1]:
                    sides[found.index(True)] += 1
        row = [len(words[0]), len(words[1]), len(real), *sides]
        totals += row
        print(
            f"  {path.stem:13s} {row[0]:6d} {row[1]:6d}  {row[2]:4d} ({len(regions) - len(real)})"
            f"  {sides[0]:3d}/{sides[1]}"
        )
    print(
        f"  {'total':13s} {totals[0]:6d} {totals[1]:6d}  {totals[2]:4d}       "
        f"{totals[3]:3d}/{totals[4]}"
    )


def score(a: str, b: str) -> None:
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer,
        EnglishTextNormalizer,
    )

    for name, normalize in (("ami", EnglishTextNormalizer({})), ("fleurs", BasicTextNormalizer())):
        if all((AB_DIR / label / f"{name}.json").exists() for label in (a, b)):
            score_references(name, a, b, normalize)
    if all((AB_DIR / label / "own").is_dir() for label in (a, b)):
        score_own(a, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dec = commands.add_parser("decode")
    dec.add_argument("label")
    dec.add_argument("--backend", default="parakeet")
    dec.add_argument("--model", help="repo id or directory (default: the backend's own)")
    dec.add_argument("--sets", default="ami,fleurs,own")
    sco = commands.add_parser("score")
    sco.add_argument("a")
    sco.add_argument("b")
    args = parser.parse_args()
    if args.command == "decode":
        decode(args.label, args.backend, args.model, args.sets.split(","))
    else:
        score(args.a, args.b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
