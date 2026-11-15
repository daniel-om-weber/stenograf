"""Does a median over a profile's entries shrug off one bad entry where the mean does not?

A profile stores up to eight per-meeting voice entries and is matched by the
mean of their cosines (``voiceprints.SpeakerProfile.similarity``). One bad
entry moves a mean by (bad − good) / n; a median ignores it once n ≥ 3. The
question is whether that buys anything measurable and costs nothing on clean
profiles. Rules compared: mean (shipped), median, mean without the lowest
score (n ≥ 3), max (control — `store_v2.py` measured it worse, 2026-08-06).

AMI's held-out meetings, Nemotron's library turns with the true count applied
(clean clusters), embeddings as production computes them. Leave one session
out: a person's profile holds their genuine mic and loop clusters from the
other three sessions; every genuine cluster of the held-out session (all
channels) is a trial, at full length and from its first 3 s of clean speech.
All four participants are enrolled, so a wrong name is a trial named as
someone else. Then one bad entry joins one profile at a time:

- **wrong person** — another participant's cluster from a non-held-out session;
- **phantom** — a no-count crosstalk cluster carrying the person's label
  (``assign_guard.py``'s rule), from a non-held-out session;
- **weak sample** — one of the person's own 2 s samples scoring below the
  naming threshold against the clean profile.

Counted at the naming threshold: the person's own trials named right, and
other people's trials named as this person.

    uv run --group eval eval/profile_median.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import ami
import assign_guard as guard
import heldout
import numpy as np
from common import read_pcm16
from naming_gate import clean_spans, embed_spans, truncate_spans
from rttm import Turn, parse_rttm

from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.embedding import VoiceEmbedder
from stenograf.diarization.nemotron import turns_from_probs

OUT = heldout.OUT_DIR
NAMING = 0.56
SESSIONS = "abcd"


def _mean(s: np.ndarray) -> float:
    return float(s.mean())


def _median(s: np.ndarray) -> float:
    return float(np.median(s))


def _drop_lowest(s: np.ndarray) -> float:
    return float(np.sort(s)[1:].mean()) if len(s) >= 3 else float(s.mean())


def _max(s: np.ndarray) -> float:
    return float(s.max())


RULES: dict[str, Callable[[np.ndarray], float]] = {
    "mean": _mean,
    "median": _median,
    "mean w/o lowest": _drop_lowest,
    "max": _max,
}


def channel_clusters(channel: dict, capped: bool, embedder: VoiceEmbedder) -> list[dict]:
    """Labelled clusters of one channel; genuine ones also get a 3 s and 2 s vector."""
    cid = channel["id"]
    probs = np.load(guard.PROBS / f"{cid}.npy").astype(np.float32)
    turns = turns_from_probs(probs, channel["num_speakers"] if capped else None)
    cache = OUT / ("assign_guard_capped" if capped else "collapse_embeddings") / f"{cid}.npz"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache) as stored:
            vectors = {k: stored[k] for k in stored.files}
    else:
        vectors = embedder.cluster_embeddings(turns, read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav"))
        np.savez(cache, **vectors)
    ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
    outside = guard.off_channel(cid, ref)
    by: dict[str, list[SpeakerTurn]] = {}
    for t in turns:
        by.setdefault(t.speaker, []).append(t)
    spans = clean_spans(by)
    pcm = None
    out = []
    for label, cl in sorted(by.items()):
        who = ami.dominant_speaker([Turn(t.speaker, t.start, t.end) for t in cl], ref)
        if who is None or label not in vectors:
            continue
        if guard.share(cl, outside) >= heldout.EXPLAINED_SHARE:
            kind = "phantom"
        elif guard.share(cl, [t for t in ref if t.speaker == who]) >= guard.PURE:
            kind = "genuine"
        else:
            continue
        short = {}
        if kind == "genuine":
            if pcm is None:
                pcm = read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav")
            for budget in (3.0, 2.0):
                short[budget] = embed_spans(
                    pcm, truncate_spans(spans[label], budget), embedder.embed
                )
        out.append(
            {
                "id": f"{cid}/{label}",
                "group": channel["group"],
                "session": channel["session"],
                "kind": channel["id"].rsplit(".", 1)[1],
                "who": who,
                "class": kind,
                "vec": vectors[label],
                "short": short,
            }
        )
    return out


def named(trial: np.ndarray, profiles: dict[str, list[np.ndarray]], rule) -> str | None:
    scores = {n: rule(np.array([trial @ e for e in entries])) for n, entries in profiles.items()}
    best = max(scores, key=lambda n: scores[n])
    return best if scores[best] >= NAMING else None


def main() -> int:
    channels = json.loads((OUT / "channels.json").read_text())
    embedder = VoiceEmbedder()
    clean = [x for c in channels for x in channel_clusters(c, True, embedder)]
    phantoms = [
        x for c in channels for x in channel_clusters(c, False, embedder) if x["class"] == "phantom"
    ]
    genuine = [x for x in clean if x["class"] == "genuine"]
    sizes: list[int] = []
    # (rule, condition, trial length) → [own named right, own trials, others named as P, ...]
    tally: dict[tuple[str, str, str], list[int]] = {}

    def score(rule_name, condition, profiles, owner, trials):
        rule = RULES[rule_name]
        for length, pick in (("full", lambda t: t["vec"]), ("3 s", lambda t: t["short"][3.0])):
            row = tally.setdefault((rule_name, condition, length), [0, 0, 0, 0])
            for t in trials:
                vec = pick(t)
                if vec is None:
                    continue
                who = named(vec, profiles, rule)
                if t["who"] == owner:
                    row[0] += who == owner
                    row[1] += 1
                else:
                    row[2] += who == owner
                    row[3] += 1

    for group in sorted({x["group"] for x in genuine}):
        for held in SESSIONS:
            pool = [x for x in genuine if x["group"] == group and x["session"] != held]
            trials = [x for x in genuine if x["group"] == group and x["session"] == held]
            profiles = {}
            for x in pool:
                if x["kind"] in ("mic", "loop"):
                    profiles.setdefault(x["who"], []).append(x["vec"])
            if not trials or len(profiles) < 2:
                continue
            for owner, entries in profiles.items():
                sizes.append(len(entries))
                if len(entries) < 3:
                    continue  # mean and median agree below three entries
                bad: list[tuple[str, np.ndarray]] = []
                others = [x for x in pool if x["who"] != owner and x["kind"] in ("mic", "loop")]
                bad += [("wrong person", x["vec"]) for x in others[:3]]
                bad += [
                    ("phantom", x["vec"])
                    for x in phantoms
                    if x["group"] == group and x["session"] != held and x["who"] == owner
                ]
                for x in pool:
                    weak = x["short"].get(2.0)
                    if (
                        x["who"] == owner
                        and weak is not None
                        and np.mean([weak @ e for e in entries]) < NAMING
                    ):
                        bad.append(("weak sample", weak))
                for rule_name in RULES:
                    score(rule_name, "clean", profiles, owner, trials)
                    for condition, vec in bad:
                        poisoned = {**profiles, owner: [*entries, vec]}
                        score(rule_name, condition, poisoned, owner, trials)
                        score(rule_name, f"{condition}, clean", profiles, owner, trials)

    print(
        f"profile sizes (entries): {dict(sorted({n: sizes.count(n) for n in sizes}.items()))}"
        " — only profiles of 3 or more are scored below"
    )
    for length in ("full", "3 s"):
        print(
            f"\ntrials at {length}: own trials named right, clean → with the bad entry"
            " (others named as the owner, clean → bad)"
        )
        for condition in ("wrong person", "phantom", "weak sample"):
            print(f"  {condition}:")
            for rule_name in RULES:
                good = tally[(rule_name, f"{condition}, clean", length)]
                bad = tally[(rule_name, condition, length)]
                print(
                    f"    {rule_name:16s} {100 * good[0] / good[1]:5.1f} → "
                    f"{100 * bad[0] / bad[1]:5.1f} %  "
                    f"({100 * good[2] / max(good[3], 1):4.1f} → "
                    f"{100 * bad[2] / max(bad[3], 1):4.1f} %)  n={bad[1]}"
                )
        print("  clean, every profile:")
        for rule_name in RULES:
            row = tally[(rule_name, "clean", length)]
            print(
                f"    {rule_name:16s} {100 * row[0] / row[1]:5.1f} %  "
                f"({100 * row[2] / max(row[3], 1):4.1f} %)  n={row[1]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
