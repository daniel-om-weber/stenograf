"""Systematic old-vs-new comparison of re-transcribed meetings.

The regression harness that caught both 2026-07-19 VAD retunes (see
README.md "window-length study"): old = the transcript stored in
``~/Documents/Meetings/<m>/transcript.json`` at recording time, new = a
fresh ``steno transcribe <audio.wav> --no-notes --no-diarization`` output
per meeting in ``--new-dir``. Compares per channel (Local* = mic,
Remote* = system), aligns normalized word streams, classifies every
difference region (added / removed / changed), and — where the study
produced Whisper hypotheses for the same channel audio — referees each
region textually against the Whisper pivot.

Interpretation notes, learned the hard way: a batch re-decode of the SAME
code reproduces a stored (live-reused) transcript almost exactly (measured
−27 words / 10.5 k), so old-vs-new diffs are attributable to the code
change — but confirm with a pre-change-batch control (git worktree at the
old commit) before blaming a specific edit, and treat symmetric one-word
filler churn as noise. Net word LOSS concentrated in `removed` regions is
the signature that a change moved window bounds (greedy TDT tail
instability).

Run:
    uv run --group eval python eval/retranscribe_compare.py --new-dir DIR
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from score import normalize  # noqa: E402

MEETINGS_DIR = Path.home() / "Documents/Meetings"
WHISPER_DIR = Path(__file__).parent / "out" / "whisper"

# meeting dir -> whisper segment-id prefix (channel suffix -mic/-sys appended)
PIVOT_IDS = {
    "meeting-20260713-130220": "en-0713",
    "meeting-20260714-150401": "de-0714",
    "meeting-20260717-133034": "de-0717s",
}
MEETINGS = [
    "meeting-20260713-130220",
    "meeting-20260714-150401",
    "meeting-20260714-160212",
    "meeting-20260715-090130",
    "meeting-20260717-133034",
    "meeting-20260717-135929",
]
INTERJECTION_MAX_WORDS = 3
REF_PAD_S = 0.25


def channel_words(transcript: dict, prefix: str) -> list[dict]:
    """Word dicts (with display/norm/time) of one channel, in time order."""
    words = []
    for entry in transcript["entries"]:
        if not entry["speaker"].startswith(prefix):
            continue
        for w in entry["words"]:
            norm = normalize(w["text"])
            if norm:
                words.append({"d": w["text"], "n": norm, "s": w["start"], "e": w["end"]})
    words.sort(key=lambda w: w["s"])
    return words


def other_labels(transcript: dict) -> set[str]:
    return {
        e["speaker"]
        for e in transcript["entries"]
        if not e["speaker"].startswith(("Local", "Remote"))
    }


def load_pivot(meeting: str, chan: str) -> list[dict] | None:
    prefix = PIVOT_IDS.get(meeting)
    if prefix is None:
        return None
    path = WHISPER_DIR / f"{prefix}-{chan}.json"
    if not path.exists():
        return None
    out = []
    for seg in json.loads(path.read_text())["segments"]:
        for w in seg["words"]:
            norm = normalize(w["text"])
            if norm:
                out.append({"n": norm, "s": w["start"], "e": w["end"]})
    return out


def ref_text(pivot: list[dict], t0: float, t1: float) -> str:
    return " ".join(
        w["n"] for w in pivot if t0 - REF_PAD_S <= (w["s"] + w["e"]) / 2 <= t1 + REF_PAD_S
    )


def merge_regions(opcodes, max_join=2):
    """Merge non-equal opcodes separated by <= max_join equal tokens."""
    regions = []
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            continue
        if regions and i1 - regions[-1][1] <= max_join:
            regions[-1][1], regions[-1][3] = i2, j2
        else:
            regions.append([i1, i2, j1, j2])
    return regions


def region_span(old, new, i1, i2, j1, j2):
    ts = [w["s"] for w in old[i1:i2]] + [w["s"] for w in new[j1:j2]]
    te = [w["e"] for w in old[i1:i2]] + [w["e"] for w in new[j1:j2]]
    return (min(ts), max(te)) if ts else (0.0, 0.0)


def wer_or_1(ref: str, hyp: str) -> float:
    if not ref:
        return -1.0
    return jiwer.wer(ref, hyp) if hyp else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-dir",
        type=Path,
        required=True,
        help="directory holding one steno-transcribe output folder per meeting "
        "(same folder names as under ~/Documents/Meetings)",
    )
    args = parser.parse_args()
    global SCRATCH
    SCRATCH = args.new_dir
    grand = {
        "old_words": 0,
        "new_words": 0,
        "added": 0,
        "removed": 0,
        "changed": 0,
        "added_words": 0,
        "removed_words": 0,
        "interjections": 0,
        "echoish_removed": 0,
        "new_better": 0,
        "old_better": 0,
        "tie": 0,
        "no_ref": 0,
    }
    examples = {"recovered": [], "new_wins": [], "old_wins": [], "removed": []}
    rows = []
    for meeting in MEETINGS:
        old_path = MEETINGS_DIR / meeting / "transcript.json"
        new_path = SCRATCH / meeting / "transcript.json"
        if not new_path.exists():
            print(f"[skip] {meeting}: no re-transcription")
            continue
        old_t = json.loads(old_path.read_text())
        new_t = json.loads(new_path.read_text())
        for t, name in ((old_t, "old"), (new_t, "new")):
            labels = other_labels(t)
            if labels:
                print(f"[note] {meeting} {name}: non-channel speaker labels {sorted(labels)}")
        stats = {k: 0 for k in grand}
        for chan, prefix in (("mic", "Local"), ("sys", "Remote")):
            old = channel_words(old_t, prefix)
            new = channel_words(new_t, prefix)
            pivot = load_pivot(meeting, chan)
            stats["old_words"] += len(old)
            stats["new_words"] += len(new)
            other_new_norm = " ".join(
                w["n"] for w in channel_words(new_t, "Remote" if prefix == "Local" else "Local")
            )
            sm = difflib.SequenceMatcher(
                a=[w["n"] for w in old], b=[w["n"] for w in new], autojunk=False
            )
            for i1, i2, j1, j2 in merge_regions(sm.get_opcodes()):
                o_txt = " ".join(w["n"] for w in old[i1:i2])
                n_txt = " ".join(w["n"] for w in new[j1:j2])
                t0, t1 = region_span(old, new, i1, i2, j1, j2)
                loc = f"{meeting[8:]}/{chan}@{t0:.0f}s"
                if not o_txt:
                    stats["added"] += 1
                    stats["added_words"] += j2 - j1
                    if j2 - j1 <= INTERJECTION_MAX_WORDS:
                        stats["interjections"] += 1
                    if len(examples["recovered"]) < 12:
                        examples["recovered"].append(f"{loc}: +“{n_txt}”")
                    continue
                if not n_txt:
                    stats["removed"] += 1
                    stats["removed_words"] += i2 - i1
                    if o_txt in other_new_norm:
                        stats["echoish_removed"] += 1
                    if len(examples["removed"]) < 8:
                        examples["removed"].append(f"{loc}: -“{o_txt[:80]}”")
                    continue
                stats["changed"] += 1
                if pivot is None:
                    continue
                ref = ref_text(pivot, t0, t1)
                if not ref:
                    stats["no_ref"] += 1
                    continue
                wo, wn = wer_or_1(ref, o_txt), wer_or_1(ref, n_txt)
                if wn < wo:
                    stats["new_better"] += 1
                    if len(examples["new_wins"]) < 10:
                        examples["new_wins"].append(
                            f"{loc}: “{o_txt[:60]}” → “{n_txt[:60]}” (ref “{ref[:60]}”)"
                        )
                elif wo < wn:
                    stats["old_better"] += 1
                    if len(examples["old_wins"]) < 10:
                        examples["old_wins"].append(
                            f"{loc}: “{o_txt[:60]}” → “{n_txt[:60]}” (ref “{ref[:60]}”)"
                        )
                else:
                    stats["tie"] += 1
        for k in grand:
            grand[k] += stats[k]
        rows.append((meeting, stats))

    print("\n| Meeting | old words | new words | Δ | added | interj. | removed | changed |")
    print("|---|---|---|---|---|---|---|---|")
    for meeting, s in rows:
        print(
            f"| {meeting[8:]} | {s['old_words']} | {s['new_words']} "
            f"| {s['new_words'] - s['old_words']:+d} | {s['added']} | {s['interjections']} "
            f"| {s['removed']} | {s['changed']} |"
        )
    g = grand
    print(
        f"| **total** | {g['old_words']} | {g['new_words']} | {g['new_words'] - g['old_words']:+d} "
        f"| {g['added']} | {g['interjections']} | {g['removed']} | {g['changed']} |"
    )
    judged = g["new_better"] + g["old_better"] + g["tie"]
    print(
        f"\nWhisper-refereed changed regions (3 meetings with pivot): "
        f"new better {g['new_better']}, old better {g['old_better']}, tie {g['tie']} "
        f"({judged} judged, {g['no_ref']} no-ref)"
    )
    print(
        f"Removed regions whose text appears on the other channel (echo-dedup?): "
        f"{g['echoish_removed']} of {g['removed']}"
    )
    for title, key in (
        ("Recovered speech (additions)", "recovered"),
        ("Refereed improvements", "new_wins"),
        ("Refereed regressions", "old_wins"),
        ("Removed text", "removed"),
    ):
        if examples[key]:
            print(f"\n{title}:")
            for ex in examples[key]:
                print(f"  - {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
