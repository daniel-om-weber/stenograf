"""Can estimate-mode over-splits be recovered downstream, and by what rule?

The estimator splits every single-speaker corpus channel 2–3 ways
(`solo_arms.py`, −22 pts word attribution). Recovery candidates, each measured
here on estimate-mode diarization of *all twenty* corpus channels:

- ``reid 1:1`` — today's resolve: one cluster per profile, so a split speaker
  keeps at most one of their clusters named and the rest stay separate.
- ``reid N:1`` — merge-at-naming: every cluster over threshold on the same
  profile takes its name, so the split collapses for an enrolled speaker.
- ``self``    — profile-free analog: clusters whose embeddings are mutually
  ≥ the same threshold merge with each other (union-find), covering the
  speaker who has no profile. The plan's other unprofiled candidate — the
  estimator's own confidence — does not exist: stenodiar emits turns only.

Discriminator stats per channel (est k vs true k, top-cluster dominance
share, pairwise cluster-embedding similarities) say whether any rule can
separate "actually one speaker, split" from "actually several speakers"
without hurting the multi-speaker channels. A final section measures
merge-at-naming's stranger risk on the *known-count* loop clusters, where
every loop contains one unenrolled participant.

AMI sessions b–d have enrolled galleries (session ``a`` is the enrollment
source and is excluded from the re-ID arms); ICSI channels have no profiles at
all — they are the unprofiled case by construction. Word times are read back
from the matrix's hypotheses (run ``eval/ami.py run`` first); estimate-mode
turns and embeddings are cached under ``out/diar/ami-est/``.

    uv run --group eval eval/split_recovery.py [--sherpa-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from common import OUT_DIR, read_pcm16
from der import Word, score_attribution
from rttm import Turn, parse_rttm, write_rttm

from stenograf.asr.base import Word as PipelineWord
from stenograf.pipeline import merge_words_turns
from stenograf.voiceprints import DEFAULT_THRESHOLD

EST_DIR = OUT_DIR / "diar" / "ami-est"


def _resolve(
    embeddings: dict[str, np.ndarray],
    gallery: dict[str, np.ndarray],
    *,
    exclusive: bool,
) -> dict[str, str]:
    """Greedy cluster→profile naming, mirroring ``SpeakerReID.resolve``.

    ``exclusive=True`` is the shipped one-to-one semantics; ``False`` lets any
    number of clusters take the same profile name (merge-at-naming)."""
    scored = sorted(
        (
            (score, cluster, name)
            for cluster, vector in embeddings.items()
            for name, enrolled in gallery.items()
            if (score := float(vector @ enrolled)) >= DEFAULT_THRESHOLD
        ),
        reverse=True,
    )
    mapping: dict[str, str] = {}
    claimed: set[str] = set()
    for _score, cluster, name in scored:
        if cluster in mapping or (exclusive and name in claimed):
            continue
        mapping[cluster] = name
        claimed.add(name)
    return mapping


def _self_merge(embeddings: dict[str, np.ndarray]) -> dict[str, str]:
    """Union-find over clusters with pairwise cosine ≥ the naming threshold."""
    parent = {c: c for c in embeddings}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a, b in combinations(sorted(embeddings), 2):
        if float(embeddings[a] @ embeddings[b]) >= DEFAULT_THRESHOLD:
            parent[find(b)] = find(a)
    return {c: find(c) for c in embeddings}


def _attribution(words: list[PipelineWord], turns: list[Turn], relabel: dict[str, str], ref):
    relabelled = [Turn(relabel.get(t.speaker, t.speaker), t.start, t.end) for t in turns]
    entries = merge_words_turns(list(words), relabelled)
    labelled = [Word(w.text, w.start, w.end, e.speaker) for e in entries for w in e.words]
    return score_attribution(labelled, ref)


def _load_word_times(path: Path) -> list[PipelineWord]:
    record = json.loads(path.read_text())
    return [PipelineWord(text=w["text"], start=w["start"], end=w["end"]) for w in record["words"]]


def _pairwise(embeddings: dict[str, np.ndarray]) -> list[float]:
    return [
        float(embeddings[a] @ embeddings[b]) for a, b in combinations(sorted(embeddings), 2)
    ]


def _est_result(channel, estimator, embed) -> tuple[list[Turn], dict[str, np.ndarray]]:
    """Estimate-mode turns + cluster embeddings, cached under ``ami-est/``.

    The two caches invalidate independently: deleting only ``.emb.json`` (an
    embedding-computation change) re-embeds the cached turns without re-running
    the estimator, so the turns stay fixed across embedding experiments."""
    from stenograf.diarization.base import SpeakerTurn
    from stenograf.diarization.sherpa import cluster_embeddings

    rttm_path = EST_DIR / f"{channel.id}.rttm"
    emb_path = EST_DIR / f"{channel.id}.emb.json"
    if rttm_path.exists() and emb_path.exists():
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads(emb_path.read_text()).items()
        }
        return parse_rttm(rttm_path), embeddings
    if rttm_path.exists():
        turns = parse_rttm(rttm_path)
        embeddings = cluster_embeddings(
            [SpeakerTurn(t.speaker, t.start, t.end) for t in turns],
            read_pcm16(channel.wav_path),
            embed,
        )
    else:
        result = estimator.diarize_with_embeddings(read_pcm16(channel.wav_path), None)
        turns = [Turn(t.speaker, t.start, t.end) for t in result.turns]
        embeddings = result.embeddings
        EST_DIR.mkdir(parents=True, exist_ok=True)
        write_rttm(rttm_path, turns, channel.id)
    emb_path.write_text(
        json.dumps({k: [float(x) for x in v] for k, v in embeddings.items()})
    )
    return turns, embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sherpa-only",
        action="store_true",
        help="estimate with sherpa even if the stenodiar helper is built",
    )
    args = parser.parse_args()

    import ami
    from diarize import _build_diarizer

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    sherpa = SherpaOnnxDiarizer()
    estimator = _build_diarizer(sherpa_only=args.sherpa_only)
    galleries = ami.build_galleries(sherpa.embed)
    hyp_dir = OUT_DIR / "diar" / "ami"

    est_rows, risk_rows = [], []
    for channel in ami.load_channels():
        words_path = hyp_dir / f"{channel.id}.words.json"
        if not words_path.exists():
            print(f"  no matrix output for {channel.id} — run eval/ami.py run", file=sys.stderr)
            continue
        ref = parse_rttm(channel.ref_path)
        words = _load_word_times(words_path)
        turns, embeddings = _est_result(channel, estimator, sherpa.embed)
        gallery = galleries.get(channel.group) if channel.session != ami.ENROLL_SESSION else None

        durations = {c: sum(t.end - t.start for t in turns if t.speaker == c) for c in embeddings}
        total = sum(durations.values()) or 1.0
        dominance = max(durations.values(), default=0.0) / total
        sims = _pairwise(embeddings)
        merged = _self_merge(embeddings)

        arms = {
            "est": _attribution(words, turns, {}, ref),
            "self": _attribution(words, turns, merged, ref),
        }
        named = {}
        if gallery:
            for arm, exclusive in (("reid 1:1", True), ("reid N:1", False)):
                named[arm] = _resolve(embeddings, gallery, exclusive=exclusive)
                arms[arm] = _attribution(words, turns, named[arm], ref)
        est_rows.append(
            (channel, len(embeddings), dominance, sims, len(set(merged.values())), arms)
        )
        print(
            f"[{channel.id}] true k={channel.num_speakers} est k={len(embeddings)} "
            f"dom={dominance:.0%} minsim={min(sims, default=1.0):.2f} "
            + "  ".join(f"{name} {a.accuracy:.1%}" for name, a in arms.items()),
            flush=True,
        )

        # Stranger risk of N:1 on the KNOWN-count clusters (the shipped path).
        if gallery and channel.num_speakers > 1:
            known_emb = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in json.loads((hyp_dir / f"{channel.id}.emb.json").read_text()).items()
            }
            known_turns = parse_rttm(hyp_dir / f"{channel.id}.rttm")
            for arm, exclusive in (("1:1", True), ("N:1", False)):
                mapping = _resolve(known_emb, gallery, exclusive=exclusive)
                wrong = stranger = 0
                for cluster, name in mapping.items():
                    truth = ami.dominant_speaker(
                        [t for t in known_turns if t.speaker == cluster], ref
                    )
                    if truth not in gallery:
                        stranger += 1
                    elif truth != name:
                        wrong += 1
                risk_rows.append((channel.id, arm, len(mapping), wrong, stranger))

    lines = [
        "# Split recovery — estimate-mode arms + the discriminator stats",
        "",
        f"Threshold {DEFAULT_THRESHOLD} everywhere (naming and self-merge). Attribution",
        "is Hungarian-mapped, so it scores *grouping*; naming correctness is the",
        "risk table. `self k` = clusters left after profile-free self-merge.",
        "",
        "| Channel | true k | est k | dom | min sim | max sim | self k "
        "| est | self | reid 1:1 | reid N:1 |",
        "|---" * 11 + "|",
    ]
    for channel, est_k, dominance, sims, self_k, arms in est_rows:
        reid1 = arms.get("reid 1:1")
        reidn = arms.get("reid N:1")
        lines.append(
            f"| {channel.id} | {channel.num_speakers} | {est_k} | {dominance:.0%} "
            f"| {min(sims, default=1.0):.2f} | {max(sims, default=1.0):.2f} | {self_k} "
            f"| {arms['est'].accuracy:.1%} | {arms['self'].accuracy:.1%} "
            f"| {reid1.accuracy:.1%} | {reidn.accuracy:.1%} |"
            if reid1 and reidn
            else f"| {channel.id} | {channel.num_speakers} | {est_k} | {dominance:.0%} "
            f"| {min(sims, default=1.0):.2f} | {max(sims, default=1.0):.2f} | {self_k} "
            f"| {arms['est'].accuracy:.1%} | {arms['self'].accuracy:.1%} | — | — |"
        )
    lines += [
        "",
        "## Merge-at-naming risk on known-count clusters",
        "",
        "Named clusters whose dominant reference speaker is a different enrolled",
        "person (*wrong*) or the unenrolled participant (*stranger named*).",
        "",
        "| Channel | arm | named | wrong | stranger named |",
        "|---|---|---|---|---|",
    ]
    for channel_id, arm, named_n, wrong, stranger in risk_rows:
        lines.append(f"| {channel_id} | {arm} | {named_n} | {wrong} | {stranger} |")

    report = OUT_DIR / "diar-split-recovery.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
