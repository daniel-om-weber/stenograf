"""AMI's held-out evaluation meetings: Nemotron 3 diarization, scored for a reader.

The corpus harness (`eval/ami.py`) is built mostly from AMI training meetings,
which Nemotron 3 Diarization trained on, so its numbers flatter that model.
AMI's evaluation split — IS1009, ES2004, TS3003, sessions a–d — is the clean
comparison: 12 meetings, 36 channels in the harness topology (mic, loop, duo).
Results and caveats: `eval/README.md`, "The estimate path, and Nemotron 3
Diarization as the one diarizer"; open gates: `PLAN.md`.

The channels are built with `ami.py`'s own download/mixing/reference functions
but deliberately NOT added to ``ami.AMI_GROUPS``: ES2004/TS3003 have not had the
vetting pass that dict requires, and IS1009 (already in it) clips on every
headset at full length (``fetch`` records the probe per headset). The manifest
lives in ``eval/out/heldout/`` so the shipped matrices never see these channels.

Scoring is about what a reader of the transcript gets:

- **attribution** — the share of single-speaker reference time that
  ``pipeline.merge_words_turns``' rule (the covering turn, else the nearest
  turn) gives to the right speaker under the best one-to-one label mapping; a
  phantom speaker's time counts as wrong. It is a proxy on reference speech,
  not an ASR word run.
- **count** — exact, and also with "explained" extras: hypothesis speakers left
  over after that mapping whose speech is ≥ 60 % inside a meeting participant
  who is *not* on the channel (headset crosstalk the reference does not list).
- DER with its missed / false-alarm / confusion split (0.25 s collar), and, for
  Nemotron, the true count applied afterwards (keep the k most active speakers,
  give orphaned frames to the most probable kept one).

``score`` scores every ``turns_*.json`` in the output directory, so turn files
other diarizers left there stay comparable.

Usage (``eval/nemotron_arm.py run`` produces the reference Nemotron turns; the
``library`` arm runs ``stenograf.diarization.nemotron`` on the same channels,
and with ``--reference-features`` — written by ``eval/nemotron_arm.py
features`` — feeds its chunk loop the reference processor's features instead
of its own, which isolates the port from the front end)::

    uv run --group eval eval/heldout.py fetch
    uv run --group eval eval/heldout.py run --arm library [--reference-features DIR]
    uv run --group eval eval/heldout.py score [--detail]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

import ami
import numpy as np
from common import read_pcm16
from der import score_der
from rttm import Turn, parse_rttm
from scipy.optimize import linear_sum_assignment

OUT_DIR = Path(__file__).resolve().parent / "out" / "heldout"
GROUPS = {"IS1009": "abcd", "ES2004": "abcd", "TS3003": "abcd"}
"""AMI's evaluation-split scenario meetings (EN2002 is not a fixed-foursome series)."""
EXPLAINED_SHARE = 0.6
KINDS = ("mic", "duo", "loop")


def _polite_download(url: str, dst: Path) -> None:
    """The AMI mirror answers bursts with 403 (measured 2026-11-15): pause between
    files and back off on 403 instead of failing the whole fetch."""
    if dst.exists():
        return
    for attempt in range(8):
        try:
            _plain_download(url, dst)
            time.sleep(5)
            return
        except urllib.error.HTTPError as err:
            if err.code != 403 or attempt == 7:
                raise
            wait = 30 * (attempt + 1)
            print(f"  403, retrying in {wait} s", flush=True)
            time.sleep(wait)


_plain_download = ami._download


def cmd_fetch(out: Path) -> int:
    ami._download = _polite_download
    ami.AMI_GROUPS = {**ami.AMI_GROUPS, **GROUPS}  # so the annotation extract covers them
    annotations = ami._ami_annotations()
    meetings = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    channels, clipping = [], {}
    for group, sessions in GROUPS.items():
        for session in sessions:
            meeting = group + session
            speakers = {}
            for agent, (headset, name) in meetings[meeting].items():
                spans = ami.parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
                if not spans:
                    continue
                wav = ami._ami_headset(meeting, headset)
                pcm = np.abs(read_pcm16(wav).astype(np.int32))
                clipping[f"{meeting}.{name}"] = round(1e6 * float(np.mean(pcm >= 32767)), 1)
                speakers[name] = (wav, ami.merge_spans(spans))
            channels += ami._build_meeting(meeting, group, session, speakers)
            print(f"[{meeting}] {len(speakers)} speakers", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"id": c.id, "group": c.group, "session": c.session, "num_speakers": c.num_speakers}
        for c in channels
    ]
    (out / "channels.json").write_text(json.dumps(manifest, indent=1))
    (out / "clipping_per_million.json").write_text(json.dumps(clipping, indent=1))
    print(f"{len(channels)} channels → {out / 'channels.json'}")
    return 0


def cmd_library(out: Path, features: Path | None) -> int:
    """The library's Nemotron with no count: turns, plus probabilities so ``score``
    adds the count-capped row. Same file layout as ``eval/nemotron_arm.py run``."""
    from stenograf.diarization import nemotron

    current: dict[str, np.ndarray] = {}
    if features is not None:
        # The reference features are what nemotron_arm.py fed its loop, frame for frame.
        def reference_log_mel(audio, start=0, stop=None, **_window):
            return np.ascontiguousarray(current["features"][start:stop])

        nemotron.log_mel = reference_log_mel
    name = "nemotron_library" + ("_reference_features" if features is not None else "")
    diarizer = nemotron.NemotronDiarizer()
    path = out / f"turns_{name}.json"
    probs_dir = out / f"probs_{name}"
    probs_dir.mkdir(exist_ok=True)
    result = json.loads(path.read_text()) if path.exists() else {}
    for channel in json.loads((out / "channels.json").read_text()):
        cid = channel["id"]
        if cid in result:
            continue  # resumable
        pcm = read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav")
        if features is not None:
            current["features"] = np.load(features / f"{cid}.features.npy", mmap_mode="r")
        started = time.monotonic()
        probs = diarizer.probabilities(pcm)
        secs = time.monotonic() - started
        np.save(probs_dir / f"{cid}.npy", probs.astype(np.float16))
        turns = nemotron.turns_from_probs(probs)
        result[cid] = {
            "turns": [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns],
            "secs": round(secs, 2),
            "audio_s": len(pcm) / 16000,
        }
        path.write_text(json.dumps(result))
        print(f"{cid} {name}: {secs:.1f} s, {len({t.speaker for t in turns})} speakers", flush=True)
    return 0


def _activity(turns: list[dict], labels: list[str], n: int) -> np.ndarray:
    grid = np.zeros((len(labels), n), bool)
    for t in turns:
        grid[labels.index(t["speaker"]), int(t["start"] * 100) : int(t["end"] * 100)] = True
    return grid


def _as_dicts(turns: list[Turn]) -> list[dict]:
    return [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns]


def attribution(ref: list[Turn], hyp: list[dict]) -> float:
    end = max([t.end for t in ref] + [t["end"] for t in hyp])
    n = int(end * 100) + 2
    ref_labels = sorted({t.speaker for t in ref})
    hyp_labels = sorted({t["speaker"] for t in hyp})
    ref_grid = _activity(_as_dicts(ref), ref_labels, n)
    hyp_grid = _activity(hyp, hyp_labels, n)
    frames = np.nonzero(ref_grid.sum(0) == 1)[0]
    covered = hyp_grid.any(0)
    label = np.where(covered, hyp_grid.argmax(0), -1)
    idx = np.nonzero(covered)[0]
    if len(idx):  # an uncovered frame takes the nearest covered frame's speaker
        pos = np.searchsorted(idx, np.arange(n))
        left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
        right = idx[np.clip(pos, 0, len(idx) - 1)]
        here = np.arange(n)
        nearest = np.where(np.abs(here - left) <= np.abs(right - here), left, right)
        label = np.where(covered, label, label[nearest])
    agree = np.zeros((len(ref_labels), len(hyp_labels)))
    np.add.at(agree, (ref_grid[:, frames].argmax(0), label[frames]), 1)
    rows, cols = linear_sum_assignment(-agree)
    return float(agree[rows, cols].sum() / len(frames))


def explained_extras(cid: str, ref: list[Turn], hyp: list[dict]) -> int:
    meeting = cid.split(".")[0]
    on_channel = {t.speaker for t in ref}
    outsiders = [
        t
        for kind in ("mic", "loop")
        for t in parse_rttm(ami.AMI_REFS_DIR / f"{meeting}.{kind}.rttm")
        if t.speaker not in on_channel
    ]
    end = max([t.end for t in ref + outsiders] + [t["end"] for t in hyp])
    n = int(end * 100) + 2
    outside = _activity(_as_dicts(outsiders), sorted({t.speaker for t in outsiders}), n).any(0)
    ref_labels, hyp_labels = sorted(on_channel), sorted({t["speaker"] for t in hyp})
    ref_grid = _activity(_as_dicts(ref), ref_labels, n)
    hyp_grid = _activity(hyp, hyp_labels, n)
    overlap = (ref_grid[:, None, :] & hyp_grid[None, :, :]).sum(-1)
    _, mapped = linear_sum_assignment(-overlap)
    leftover = [j for j in range(len(hyp_labels)) if j not in set(mapped)]
    return sum(
        (hyp_grid[j] & outside).sum() >= EXPLAINED_SHARE * max(hyp_grid[j].sum(), 1)
        for j in leftover
    )


def count_capped(probs: np.ndarray, k: int) -> list[dict]:
    """The true count applied after the fact: keep the k most active speaker slots and
    give frames only dropped slots claimed to the most probable kept slot."""
    active = probs > 0.5
    keep = np.argsort(-active.sum(0))[:k]
    capped = np.zeros_like(active)
    capped[:, keep] = active[:, keep]
    orphan = active.any(1) & ~capped.any(1)
    best = keep[np.argmax(probs[:, keep], 1)]
    capped[np.nonzero(orphan)[0], best[orphan]] = True
    turns = []
    for slot in range(capped.shape[1]):
        edges = np.diff(np.concatenate([[0], capped[:, slot].astype(np.int8), [0]]))
        turns += [
            {"speaker": f"S{slot}", "start": s * 0.01, "end": e * 0.01}
            for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True)
        ]
    return turns


def _row(cid: str, k: int, ref: list[Turn], hyp: list[dict]) -> dict:
    speech: dict[str, float] = {}
    for t in hyp:
        speech[t["speaker"]] = speech.get(t["speaker"], 0.0) + t["end"] - t["start"]
    total = sum(speech.values()) or 1.0
    audible = {s for s, v in speech.items() if v > 0.01 * total}
    der = score_der(ref, [Turn(t["speaker"], t["start"], t["end"]) for t in hyp])
    return {
        "k": k,
        "found": len(speech),
        "found_1pct": len(audible),
        # Explained among the audible speakers only: a crosstalk slot under 1 %
        # of the speech is already out of found_1pct, and counting it here too
        # subtracted it twice.
        "explained": explained_extras(cid, ref, [t for t in hyp if t["speaker"] in audible]),
        "attribution": attribution(ref, hyp),
        "der": der.der,
        "miss": der.missed,
        "fa": der.false_alarm,
        "confusion": der.confusion,
    }


def cmd_score(out: Path, detail: bool) -> int:
    channels = {c["id"]: c for c in json.loads((out / "channels.json").read_text())}
    systems: dict[str, dict[str, dict]] = {}
    speed: dict[str, float] = {}
    for path in sorted(out.glob("turns_*.json")):
        name = path.stem.removeprefix("turns_")
        data = {cid: v for cid, v in json.loads(path.read_text()).items() if cid in channels}
        rows = {}
        for cid, v in data.items():
            ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
            rows[cid] = _row(cid, channels[cid]["num_speakers"], ref, v["turns"])
        systems[name] = rows
        hours = sum(v["audio_s"] for v in data.values()) / 3600
        speed[name] = sum(v["secs"] for v in data.values()) / hours if hours else float("nan")
        probs_dir = out / f"probs_{name}"
        if probs_dir.is_dir():
            capped = {}
            for cid in data:
                probs = np.load(probs_dir / f"{cid}.npy").astype(np.float32)
                k = channels[cid]["num_speakers"]
                ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
                capped[cid] = _row(cid, k, ref, count_capped(probs, k))
            systems[f"{name} + count cap"] = capped
            speed[f"{name} + count cap"] = speed[name]

    print(f"{'system':36s}" + "".join(f"{kind:>30s}" for kind in KINDS) + "  s/audio-h")
    print(f"{'':36s}" + f"{'count ok(expl)/n attr conf':>30s}" * len(KINDS))
    for name, rows in systems.items():
        cells = []
        for kind in KINDS:
            rs = [r for cid, r in rows.items() if cid.endswith(kind)]
            if not rs:
                cells.append(f"{'—':>30s}")
                continue
            exact = sum(r["found"] == r["k"] for r in rs)
            explained = sum(r["found_1pct"] - r["explained"] == r["k"] for r in rs)
            attr = 100 * np.mean([r["attribution"] for r in rs])
            conf = 100 * np.mean([r["confusion"] for r in rs])
            cells.append(f"{exact:2d}({explained:2d})/{len(rs):2d} {attr:5.1f}% {conf:4.1f}%")
        print(f"{name:36s}" + "".join(f"{c:>30s}" for c in cells) + f"  {speed[name]:8.0f}")
    print(
        "\ncount ok(expl) = exact count (also ignoring speakers with < 1 % of the speech and "
        "crosstalk-explained extras); attr = attribution, mean; conf = speaker confusion, mean."
        "\nCompare systems on the channels they share: an arm that has not run everywhere "
        "averages over fewer channels."
    )
    if detail:
        for cid in channels:
            for name, rows in systems.items():
                if cid in rows:
                    r = rows[cid]
                    print(
                        f"  {cid:14s} {name:34s} k={r['k']} found={r['found']}"
                        f"({r['found_1pct']}, {r['explained']} expl) DER={100 * r['der']:5.1f} "
                        f"miss={100 * r['miss']:4.1f} fa={100 * r['fa']:4.1f} "
                        f"conf={100 * r['confusion']:4.1f} attr={100 * r['attribution']:5.1f}"
                    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=["library"], default="library")
    run.add_argument("--reference-features", type=Path, help="library arm only")
    sub.add_parser("score").add_argument("--detail", action="store_true")
    args = parser.parse_args()
    if args.cmd == "fetch":
        return cmd_fetch(args.out)
    if args.cmd == "run":
        return cmd_library(args.out, args.reference_features)
    return cmd_score(args.out, args.detail)


if __name__ == "__main__":
    sys.exit(main())
