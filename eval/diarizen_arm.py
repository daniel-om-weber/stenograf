"""DiariZen meeting-base as a drop-in segmentation+clustering arm — the step-5 gate.

`PLAN-DIARIZATION.md` step 5: `BUT-FIT/diarizen-meeting-base` (MIT, ungated)
is the one shippable model the research record puts clearly above the pyannote
family, but its published 15.6 AMI-SDM is family-A scoring on mixed
single-stream audio with estimated counts — not comparable to our family-B
per-channel known-count numbers, so the adoption trigger ("still ≥2 pts DER or
a visible word-attribution gap") is decidable only by running it through this
harness. This arm does exactly the replacement the plan describes: DiariZen
produces the turns, everything else — word intersection, cluster embeddings
for re-ID, scoring — stays ours.

The model runs in an isolated venv (`~/.cache/diarizen-eval`, 2026-08-07:
torch 2.5.1 CPU + DiariZen@844f555 with its vendored pyannote fork — the fork
is mandatory, stock pyannote.audio lacks its clustering classes, and two
undeclared deps `psutil`/`accelerate` surface only at import). Nothing of it
touches the project environment; this script talks to it via subprocess. The
known speaker count *binds* on this checkpoint (AHC clustering honours
min=max=k; verified by a k-sweep with real repartitioning — but VBx-configured
DiariZen checkpoints silently ignore the count, so any checkpoint change must
re-verify). Telemetry: the vendored fork is pyannote 3.1.1, which predates
pyannote's telemetry; HF hub telemetry is disabled per invocation below.

Channel selection mirrors the harness: loop channels at the known count k
(straight k — the k+1-fold is a workaround for *our* estimator's over-split,
not part of the replacement), duo channels at k=2 (the near-field go/no-go the
plan requires: meeting-base is trained exclusively far-field). Words come from
the baseline matrix (diarization-independent). Requires the cached matrix
(`eval/ami.py run`) and the venv above.

Run (pilot: one loop channel first — validate before the long unattended set)::

    uv run --group eval eval/diarizen_arm.py --segments ES2003c.loop
    uv run --group eval eval/diarizen_arm.py            # all loops + duos
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import OUT_DIR, REFS_DIR, read_pcm16
from der import _load_words, score_attribution, score_der
from diarize import _words_json
from rttm import Turn, parse_rttm, write_rttm

from stenograf.asr.base import Word
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.loop import OwnDiarizer
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import merge_words_turns

BASELINE = "ami"
VENV_ROOT = Path.home() / ".cache" / "diarizen-eval"
RUNNER = VENV_ROOT / "run_diarizen.py"
PYTHON = VENV_ROOT / "venv" / "bin" / "python"


def diarizen_turns(wav: Path, num_speakers: int | None, rttm_out: Path) -> float:
    """Run the isolated DiariZen runner; return its wall-clock seconds."""
    env = os.environ | {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DO_NOT_TRACK": "1",
    }
    count = [] if num_speakers is None else ["--num-speakers", str(num_speakers)]
    started = time.monotonic()
    subprocess.run(
        [str(PYTHON), str(RUNNER), str(wav), str(rttm_out), *count],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated channel ids (default: loops + duos)")
    parser.add_argument("--out-name", default="ami-diarizen")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="leave the count unconstrained (checkpoint defaults, min 2 / max 8) — "
        "the research record measured forced counts *hurting* pyannote-family "
        "pipelines, so the known-count arm needs this control",
    )
    args = parser.parse_args()

    import ami

    if not PYTHON.exists():
        raise SystemExit(f"no DiariZen venv at {VENV_ROOT} — see module docstring")
    out_dir = OUT_DIR / "diar" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = OUT_DIR / "diar" / BASELINE
    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [
        c
        for c in ami.load_channels(include_duo=True)
        if c.num_speakers > 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        raise SystemExit("no matching multi-speaker channels")

    embedder = OwnDiarizer()
    for channel in channels:
        raw_rttm = out_dir / f"{channel.id}.diarizen-raw.rttm"
        count = None if args.estimate else channel.num_speakers
        seconds = diarizen_turns(channel.wav_path, count, raw_rttm)
        turns = [
            SpeakerTurn(speaker=t.speaker, start=t.start, end=t.end)
            for t in parse_rttm(raw_rttm)
        ]
        if not turns:
            print(f"[{channel.id}] DiariZen produced no turns", file=sys.stderr)
            return 1
        samples = read_pcm16(channel.wav_path)
        embeddings = cluster_embeddings(turns, samples, embedder.embed)

        base_words = base_dir / f"{channel.id}.words.json"
        words_available = base_words.exists()  # duos have no baseline ASR pass
        if words_available:
            words = [
                Word(text=w["text"], start=w["start"], end=w["end"])
                for w in json.loads(base_words.read_text())["words"]
            ]
            entries = merge_words_turns(words, turns)
            (out_dir / f"{channel.id}.words.json").write_text(
                json.dumps(_words_json(entries), ensure_ascii=False, indent=2)
            )
        write_rttm(
            out_dir / f"{channel.id}.rttm",
            [Turn(t.speaker, t.start, t.end) for t in turns],
            channel.id,
        )
        (out_dir / f"{channel.id}.emb.json").write_text(
            json.dumps({k: [float(x) for x in v] for k, v in embeddings.items()})
        )

        (out_dir / f"{channel.id}.meta.json").write_text(
            json.dumps({"rtf": seconds / (len(samples) / 16000)})
        )
        print(f"[{channel.id}] done in {seconds:.0f}s")

    rows = []
    for rttm_path in sorted(out_dir.glob("*.rttm")):
        if rttm_path.name.endswith(".diarizen-raw.rttm"):
            continue
        channel_id = rttm_path.name.removesuffix(".rttm")
        ref = parse_rttm(REFS_DIR / "ami" / f"{channel_id}.rttm")
        arm = score_der(ref, parse_rttm(rttm_path))
        base_rttm = base_dir / f"{channel_id}.rttm"
        base = score_der(ref, parse_rttm(base_rttm)) if base_rttm.exists() else None
        meta_path = out_dir / f"{channel_id}.meta.json"
        rtf = json.loads(meta_path.read_text())["rtf"] if meta_path.exists() else None
        row = {"id": channel_id, "arm": arm, "base": base, "rtf": rtf}
        arm_words = out_dir / f"{channel_id}.words.json"
        chan_base_words = base_dir / f"{channel_id}.words.json"
        if arm_words.exists() and chan_base_words.exists():
            row["attr"] = score_attribution(_load_words(arm_words), ref)
            row["base_attr"] = score_attribution(_load_words(chan_base_words), ref)
        rows.append(row)

    mode = (
        "count unconstrained (checkpoint defaults)"
        if args.estimate
        else "known count bound (min=max=k)"
    )
    lines = [
        f"# DiariZen meeting-base arm vs production baseline (diarizen_arm.py, {args.out_name})",
        "",
        f"{mode}; turns from DiariZen, words/embeddings/scoring ours.",
        "",
        "| channel | DER | Δ vs base | miss | fa | conf | attribution | Δ attr | RTF |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d_der = f"{(r['arm'].der - r['base'].der) * 100:+.1f}" if r["base"] else "—"
        attr = f"{r['attr'].accuracy:.1%}" if "attr" in r else "—"
        d_attr = (
            f"{(r['attr'].accuracy - r['base_attr'].accuracy) * 100:+.1f}"
            if "attr" in r
            else "—"
        )
        a = r["arm"]
        rtf = f"{r['rtf']:.2f}" if r["rtf"] is not None else "—"
        lines.append(
            f"| {r['id']} | {a.der:.1%} | {d_der} | {a.missed:.1%} | {a.false_alarm:.1%} "
            f"| {a.confusion:.1%} | {attr} | {d_attr} | {rtf} |"
        )
    report = OUT_DIR / f"diar-{args.out_name}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
