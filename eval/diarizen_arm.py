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

Three modes, because the replacement boundary decides what is fair
(2026-08-09 review):

- ``kplus1fold`` (primary): DiariZen at k+1, folded to k by
  ``pipeline.fold_excess_clusters`` — the fold lives in the pipeline layer
  step 5 does NOT replace, and exact-k is a configuration our own stack loses
  4.4 pts with (ward@k 17.6 vs 13.2 %), so comparing DiariZen@k to
  production@k+1+fold would hand our side an advantage the product would not
  have. The fold's 0.8 gate was calibrated on ward clusters; the report's
  fold-audit table re-measures the same/cross pair similarities on DiariZen
  clusters rather than assuming the band transfers.
- ``exactk``: the straight-k ablation (isolates what the fold contributes on
  DiariZen clusters).
- ``estimate``: count unconstrained — DiariZen exactly as published
  (`recipes/diar_ssl` passes no counts; config min 2 / max 8). This is a
  DiariZen-internal control for "does forcing the count hurt it" — its report
  carries no Δ-vs-base columns, because the production baseline consumed true
  counts and our current estimate seat (stenodiar/speakrs) has no post-ward
  baseline artifacts.

Decision rule, pre-registered before the matrix landed (per-channel variance
is huge — baseline 5.0–42.1 % DER — so a small mean delta is one unlucky
channel, not a verdict): adoption consideration requires the best fair
DiariZen arm to beat production on paired per-channel ΔDER in BOTH mean and
median by ≥2 pts, with a sign-test-supported win/loss record, no worst-channel
regression story that survives leave-one-group-out over the five groups, and
word attribution weighing above DER where they disagree; the duo channels are
the near-field go/no-go (meeting-base trained far-field only). A win here is
still not adoption — it opens the ONNX-export program plus the re-runs steps
2.3/2.4 oblige; a loss declines step 5 with these numbers.

The model runs in an isolated venv (`~/.cache/diarizen-eval`, 2026-08-07:
torch 2.5.1 CPU + DiariZen@844f555 with its vendored pyannote fork — the fork
is mandatory, stock pyannote.audio lacks its clustering classes, and two
undeclared deps `psutil`/`accelerate` surface only at import). Nothing of it
touches the project environment; this script talks to it via subprocess. The
known speaker count *binds* on this checkpoint (AHC clustering honours
min=max=k; verified by a k-sweep with real repartitioning — but VBx-configured
DiariZen checkpoints silently ignore the count, so any checkpoint change must
re-verify), and the report's `bound` column records the realized cluster count
anyway: pyannote's AHC falls back with only a printed warning when it cannot
reach k, and the runner's stdout lands in `{id}.runner.log` rather than the
void. Telemetry: the vendored fork is pyannote 3.1.1, which predates
pyannote's telemetry; HF hub telemetry is disabled and, since the snapshots
are cached, `HF_HUB_OFFLINE=1` removes the hub round-trip per invocation
entirely.

Channel selection mirrors the harness: loop channels (known count k), duo
channels at k=2 (the near-field go/no-go). Words come from the baseline
matrix (diarization-independent; loops only — duos carry no ASR pass, so they
score DER-only against `ami-duo-ward`, the production ward+fold arm rebuilt
from the freeze). Requires the cached matrix (`eval/ami.py run`), the duo
baseline (`eval/loop_arm.py --cluster ward --include-duo --out-name
ami-duo-ward --segments <duos>`), and the venv above.

Two measured shortcuts, both parity-checked, not assumed (2026-08-09):

- ``--device mps``: estimate mode is turn-identical to CPU (ES2003a.loop:
  same 140 turns / 5 speakers, cross-DER 0.000 %); known-count mode is not
  byte-exact but immaterial (IS1009a.loop at k+1: cross-DER 0.064 %, ΔDER
  0.019 pts, same turn count) — at RTF 0.06 vs CPU's 1.22, the difference
  between a ~1.5 h and a 25 h matrix.
- the stage-1 cache (segmentations + chunk embeddings are k-independent):
  fresh-path RTTM byte-matches the pre-cache runner, cache-path RTTM
  byte-matches the fresh path (ES2003c.loop, k=3, MPS); a cache hit turns a
  126 s channel into 0.6 s, so every arm after the first is nearly free.

Re-verify both on any torch or checkpoint change.

Run (pilot one channel first; a finished channel is skipped on re-run, so the
matrix resumes after interruption; ``--rescore`` rebuilds reports alone)::

    uv run --group eval eval/diarizen_arm.py --mode kplus1fold --device mps
    uv run --group eval eval/diarizen_arm.py --mode exactk --device mps
    uv run --group eval eval/diarizen_arm.py --mode estimate --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from common import OUT_DIR, REFS_DIR, read_pcm16
from der import _load_words, score_attribution, score_der
from diarize import _words_json
from rttm import Turn, parse_rttm, write_rttm

from stenograf.asr.base import Word
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.loop import OwnDiarizer
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import fold_excess_clusters, merge_words_turns

BASELINE = "ami"
DUO_BASELINE = "ami-duo-ward"
VENV_ROOT = Path.home() / ".cache" / "diarizen-eval"
RUNNER = VENV_ROOT / "run_diarizen.py"
PYTHON = VENV_ROOT / "venv" / "bin" / "python"
STAGE1_DIR = VENV_ROOT / "stage1"

MODES = ("kplus1fold", "exactk", "estimate")


def diarizen_turns(
    wav: Path,
    num_speakers: int | None,
    rttm_out: Path,
    device: str | None,
    stage1: Path,
    log_out: Path,
) -> float:
    """Run the isolated DiariZen runner; return its wall-clock seconds."""
    env = os.environ | {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }
    count = [] if num_speakers is None else ["--num-speakers", str(num_speakers)]
    dev = [] if device is None else ["--device", device]
    started = time.monotonic()
    with log_out.open("w") as log:
        subprocess.run(
            [
                str(PYTHON),
                str(RUNNER),
                str(wav),
                str(rttm_out),
                *count,
                *dev,
                "--stage1-cache",
                str(stage1),
            ],
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return time.monotonic() - started


def base_dir_for(channel_id: str) -> Path:
    name = DUO_BASELINE if channel_id.endswith(".duo") else BASELINE
    return OUT_DIR / "diar" / name


def ref_overlap_seconds(turns: list[Turn], ref: list[Turn]) -> dict[str, dict[str, float]]:
    """Per hypothesis label: seconds of overlap with each reference speaker."""
    out: dict[str, dict[str, float]] = {}
    for t in turns:
        acc = out.setdefault(t.speaker, {})
        for r in ref:
            sec = min(t.end, r.end) - max(t.start, r.start)
            if sec > 0:
                acc[r.speaker] = acc.get(r.speaker, 0.0) + sec
    return out


def fold_audit_row(out_dir: Path, channel_id: str) -> str | None:
    """What the k+1→k fold merged on DiariZen clusters, judged by the reference."""
    raw_rttm = out_dir / f"{channel_id}.diarizen-raw.rttm"
    raw_emb_path = out_dir / f"{channel_id}.raw-emb.json"
    folded_rttm = out_dir / f"{channel_id}.rttm"
    if not (raw_rttm.exists() and raw_emb_path.exists() and folded_rttm.exists()):
        return None
    raw = parse_rttm(raw_rttm)
    folded_labels = {t.speaker for t in parse_rttm(folded_rttm)}
    gone = sorted({t.speaker for t in raw} - folded_labels)
    if not gone:
        return f"| {channel_id} | — | no fold (bound < k+1) | — | — |"
    emb = {k: np.asarray(v) for k, v in json.loads(raw_emb_path.read_text()).items()}
    ref = parse_rttm(REFS_DIR / "ami" / f"{channel_id}.rttm")
    dominant = {
        label: max(overlaps, key=overlaps.get) if overlaps else "∅"
        for label, overlaps in ref_overlap_seconds(raw, ref).items()
    }
    cells = []
    verdicts = []
    for label in gone:
        partners = {
            other: float(emb[label] @ emb[other])
            for other in emb
            if other != label and other not in gone
        }
        partner, sim = max(partners.items(), key=lambda kv: kv[1])
        same = dominant.get(label) == dominant.get(partner)
        cells.append(f"{label}→{partner} @ {sim:.3f}")
        verdicts.append("same" if same else "CROSS")
    return (
        f"| {channel_id} | {', '.join(gone)} | {', '.join(cells)} "
        f"| {', '.join(verdicts)} | {len(folded_labels)} |"
    )


def run_channel(channel, mode: str, out_dir: Path, device: str | None, embedder) -> None:
    raw_rttm = out_dir / f"{channel.id}.diarizen-raw.rttm"
    done_marker = out_dir / f"{channel.id}.rttm"
    if done_marker.exists() and raw_rttm.exists():
        print(f"[{channel.id}] already done — skipped", flush=True)
        return

    if mode == "kplus1fold":
        count = channel.num_speakers + 1
    elif mode == "exactk":
        count = channel.num_speakers
    else:
        count = None

    seconds = None
    if not raw_rttm.exists():
        seconds = diarizen_turns(
            channel.wav_path,
            count,
            raw_rttm,
            device,
            STAGE1_DIR / f"{channel.id}.npz",
            out_dir / f"{channel.id}.runner.log",
        )
    turns = [
        SpeakerTurn(speaker=t.speaker, start=t.start, end=t.end) for t in parse_rttm(raw_rttm)
    ]
    if not turns:
        raise RuntimeError("DiariZen produced no turns")
    samples = read_pcm16(channel.wav_path)
    embeddings = cluster_embeddings(turns, samples, embedder.embed)

    if mode == "kplus1fold":
        (out_dir / f"{channel.id}.raw-emb.json").write_text(
            json.dumps({k: [float(x) for x in v] for k, v in embeddings.items()})
        )
        folded = fold_excess_clusters(turns, embeddings, channel.num_speakers)
        turns, embeddings = list(folded[0]), folded[1]

    base_words = base_dir_for(channel.id) / f"{channel.id}.words.json"
    if base_words.exists():  # duos have no baseline ASR pass
        words = [
            Word(text=w["text"], start=w["start"], end=w["end"])
            for w in json.loads(base_words.read_text())["words"]
        ]
        entries = merge_words_turns(words, turns)
        (out_dir / f"{channel.id}.words.json").write_text(
            json.dumps(_words_json(entries), ensure_ascii=False, indent=2)
        )
    (out_dir / f"{channel.id}.emb.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in embeddings.items()})
    )
    if seconds is not None:
        (out_dir / f"{channel.id}.meta.json").write_text(
            json.dumps(
                {
                    "rtf": seconds / (len(samples) / 16000),
                    "device": device or "cpu",
                    "requested": count,
                }
            )
        )
    # The processed rttm doubles as the resume marker — it must be written last.
    write_rttm(
        out_dir / f"{channel.id}.rttm",
        [Turn(t.speaker, t.start, t.end) for t in turns],
        channel.id,
    )
    stamp = f"{seconds:.0f}s" if seconds is not None else "reused raw"
    print(f"[{channel.id}] done in {stamp}", flush=True)


def write_report(out_name: str, mode: str, channels_by_id: dict) -> Path:
    out_dir = OUT_DIR / "diar" / out_name
    rows = []
    for rttm_path in sorted(out_dir.glob("*.rttm")):
        if rttm_path.name.endswith(".diarizen-raw.rttm"):
            continue
        channel_id = rttm_path.name.removesuffix(".rttm")
        ref = parse_rttm(REFS_DIR / "ami" / f"{channel_id}.rttm")
        arm = score_der(ref, parse_rttm(rttm_path))
        base_rttm = base_dir_for(channel_id) / f"{channel_id}.rttm"
        base = (
            score_der(ref, parse_rttm(base_rttm))
            if mode != "estimate" and base_rttm.exists()
            else None
        )
        raw_path = out_dir / f"{channel_id}.diarizen-raw.rttm"
        bound = len({t.speaker for t in parse_rttm(raw_path)}) if raw_path.exists() else None
        meta_path = out_dir / f"{channel_id}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        runner_log = out_dir / f"{channel_id}.runner.log"
        cache_hit = runner_log.exists() and "cache hit" in runner_log.read_text()
        row = {
            "id": channel_id,
            "arm": arm,
            "base": base,
            "rtf": "cache" if cache_hit else meta.get("rtf"),
            "bound": bound,
            "k": channels_by_id[channel_id].num_speakers if channel_id in channels_by_id else None,
        }
        arm_words = out_dir / f"{channel_id}.words.json"
        chan_base_words = base_dir_for(channel_id) / f"{channel_id}.words.json"
        if arm_words.exists() and chan_base_words.exists() and mode != "estimate":
            row["attr"] = score_attribution(_load_words(arm_words), ref)
            row["base_attr"] = score_attribution(_load_words(chan_base_words), ref)
        rows.append(row)

    mode_line = {
        "kplus1fold": "DiariZen at k+1, folded to k by the production "
        "`fold_excess_clusters` (the pipeline layer step 5 keeps)",
        "exactk": "known count bound straight (min=max=k) — the no-fold ablation",
        "estimate": "count unconstrained (checkpoint defaults, min 2 / max 8) — "
        "DiariZen as published; no Δ columns: the baseline consumed true counts, "
        "so a Δ here would compare unlike seats",
    }[mode]
    lines = [
        f"# DiariZen meeting-base arm vs production baseline (diarizen_arm.py, {out_name})",
        "",
        f"{mode_line}; turns from DiariZen, words/embeddings/scoring ours. "
        f"Duo rows score DER-only against `{DUO_BASELINE}` (no ASR pass on duos).",
        "",
        "| channel | k | bound | DER | Δ vs base | miss | fa | conf | attribution | Δ attr | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d_der = f"{(r['arm'].der - r['base'].der) * 100:+.1f}" if r["base"] else "—"
        attr = f"{r['attr'].accuracy:.1%}" if "attr" in r else "—"
        d_attr = (
            f"{(r['attr'].accuracy - r['base_attr'].accuracy) * 100:+.1f}" if "attr" in r else "—"
        )
        a = r["arm"]
        if r["rtf"] == "cache":
            rtf = "(cache)"  # stage-1 reused — wall-clock says nothing about the model
        elif r["rtf"] is not None:
            rtf = f"{r['rtf']:.2f}"
        else:
            rtf = "—"
        bound = "—" if r["bound"] is None else str(r["bound"])
        k = "—" if r["k"] is None else str(r["k"])
        lines.append(
            f"| {r['id']} | {k} | {bound} | {a.der:.1%} | {d_der} | {a.missed:.1%} "
            f"| {a.false_alarm:.1%} | {a.confusion:.1%} | {attr} | {d_attr} | {rtf} |"
        )

    def summarize(pick, label):
        picked = [r for r in rows if pick(r) and r["base"]]
        if not picked:
            return
        deltas = sorted((r["arm"].der - r["base"].der) * 100 for r in picked)
        mean = sum(deltas) / len(deltas)
        median = statistics.median(deltas)
        wins = sum(1 for d in deltas if d < 0)
        lines.append(
            f"- **{label}** (n={len(picked)}): paired ΔDER mean {mean:+.1f}, "
            f"median {median:+.1f}, DiariZen wins {wins}/{len(deltas)}, "
            f"worst {deltas[-1]:+.1f}, best {deltas[0]:+.1f}"
        )

    lines.append("")
    summarize(lambda r: r["id"].endswith(".loop"), "loops")
    summarize(lambda r: r["id"].endswith(".duo"), "duos (near-field go/no-go)")

    if mode == "kplus1fold":
        lines += [
            "",
            "## Fold audit on DiariZen clusters (is 0.8 still the right gate?)",
            "",
            "The folded label's best partner and their cosine; `CROSS` means the",
            "reference assigns the two clusters to different speakers, i.e. the",
            "fold merged real speakers. (The partner shown is the best *surviving*",
            "cluster — which is the fold's actual choice in both branches: the",
            "gated max-pair keeps the longer side, the duration spare merges into",
            "its most-similar survivor.)",
            "",
            "| channel | folded | merge @ cosine | ref verdict | final clusters |",
            "|---|---|---|---|---|",
        ]
        for rttm_path in sorted(out_dir.glob("*.rttm")):
            if rttm_path.name.endswith(".diarizen-raw.rttm"):
                continue
            row = fold_audit_row(out_dir, rttm_path.name.removesuffix(".rttm"))
            if row:
                lines.append(row)

    report = OUT_DIR / f"diar-{out_name}.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="kplus1fold")
    parser.add_argument("--segments", help="comma-separated channel ids (default: loops + duos)")
    parser.add_argument("--out-name", help="default: ami-diarizen[-exactk|-est] by mode")
    parser.add_argument("--device", help="torch device for the runner (parity note above)")
    parser.add_argument(
        "--rescore", action="store_true", help="rebuild the report from existing artifacts only"
    )
    args = parser.parse_args()

    import ami

    out_name = args.out_name or {
        "kplus1fold": "ami-diarizen",
        "exactk": "ami-diarizen-exactk",
        "estimate": "ami-diarizen-est",
    }[args.mode]

    if not PYTHON.exists():
        raise SystemExit(f"no DiariZen venv at {VENV_ROOT} — see module docstring")
    out_dir = OUT_DIR / "diar" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [
        c
        for c in ami.load_channels(include_duo=True)
        if c.num_speakers > 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        raise SystemExit("no matching multi-speaker channels")

    failures = []
    if not args.rescore:
        embedder = OwnDiarizer()
        for channel in channels:
            try:
                run_channel(channel, args.mode, out_dir, args.device, embedder)
            except Exception as exc:  # noqa: BLE001 — one bad channel must not kill the matrix
                failures.append(channel.id)
                print(f"[{channel.id}] FAILED: {exc}", file=sys.stderr, flush=True)

    report = write_report(out_name, args.mode, {c.id: c for c in channels})
    print(f"wrote {report}", flush=True)
    if failures:
        print(f"failures: {', '.join(failures)}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
