"""Embedding-model A/B: candidates vs the shipped ERes2Net-base, whole gate.

`PLAN-DIARIZATION.md` step 3: the shipped embedder's one measured weakness is
the short-turn cliff (2 s EER 3.28 % vs ERes2NetV2's 1.48 % in the research
record). The embedding model sits in *two* seats — clustering inside sherpa's
diarization and the re-ID scoring — so a candidate must be measured in both:
a model that names better but clusters worse loses where users read.

Per candidate: the full 40-channel matrix diarization into
``out/diar/ami-<tag>/`` (child process per loop channel, mirroring
``ami.run``), then DER + word attribution against the corpus refs, then
same-group naming trials (galleries re-enrolled with the candidate, the
2026-08-07 convention) at full duration and 3 s / 2 s truncation. The
baseline arm re-scores the cached shipped-model matrix through the identical
code path. Cross-model numbers compare at FAR-anchored operating points
(DIR@FAR0, EER) — raw thresholds are model-specific by design
(``voiceprints.py`` binds profiles to the model id).

Gate (the plan's): DER and DIR@FAR0 both non-regressing, short-turn naming
improved. One winner ships; the loser is declined with its numbers.

Models land in ``eval/audio/models/`` (gitignored). Run::

    uv run --group eval eval/embedder_ab.py            # all candidates
    uv run --group eval eval/embedder_ab.py --tags v2  # subset
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from common import AUDIO_DIR, OUT_DIR, read_pcm16
from der import _load_words, score_attribution, score_der
from reid_score import Trial, dir_at_far, eer_point
from rttm import parse_rttm

MODELS_DIR = AUDIO_DIR / "models"

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
CANDIDATES: dict[str, str] = {
    # 2026-08-07 model hunt: sherpa-hosted files only (they carry the metadata
    # SpeakerEmbeddingExtractor hard-requires; raw ModelScope exports don't).
    # English/VoxCeleb ERes2NetV2 was never released (3D-Speaker #208: no plan
    # to), so the v2 candidate is the Chinese-200k model — a language-domain
    # change this A/B measures rather than assumes. ResNet34-LM (CC-BY-4.0,
    # dim 256) is the pyannote-3 pairing. Conditional third candidate if both
    # fail the gate: self-export of iic/speech_eres2net_large_sv_en_voxceleb
    # (EN, EER 0.57 %, unsupported by sherpa's exporter — unproven path).
    "v2zh": f"{_SHERPA}/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "resnet34lm": f"{_SHERPA}/wespeaker_en_voxceleb_resnet34_LM.onnx",
}

BASELINE_TAG = "base"
BUDGETS = (None, 3.0, 2.0)


def fetch_model(tag: str, url: str) -> Path:
    dst = MODELS_DIR / f"{tag}.onnx"
    if not dst.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        print(f"fetching {url}")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.rename(dst)
    return dst


def ensure_matrix(tag: str, model: Path | None) -> Path:
    """Diarize all channels with ``model`` into ``out/diar/ami-<tag>`` unless
    cached; per-loop-channel child processes, mics batched (ami.run's shape)."""
    import ami

    out_dir = OUT_DIR / "diar" / ("ami" if model is None else f"ami-{tag}")
    channels = ami.load_channels()
    missing = [c for c in channels if not (out_dir / f"{c.id}.rttm").exists()]
    if not missing:
        return out_dir
    if model is None:
        raise SystemExit("baseline matrix incomplete — run `eval/ami.py run` first")
    eval_dir = Path(__file__).parent

    def phase(ids: list[str]) -> None:
        subprocess.run(
            [
                sys.executable,
                "diarize.py",
                "--ami",
                "--segments",
                ",".join(ids),
                "--embedding",
                str(model),
                "--out-name",
                f"ami-{tag}",
            ],
            cwd=eval_dir,
            check=True,
        )

    mics = [c.id for c in missing if c.num_speakers == 1]
    if mics:
        phase(mics)
    for channel in (c for c in missing if c.num_speakers > 1):
        phase([channel.id])
    return out_dir


def matrix_scores(out_dir: Path) -> tuple[float, float, float]:
    """(mean loop DER, mean loop word-attribution, mean mic attribution)."""
    import ami

    ders, attrs, mic_attrs = [], [], []
    for channel in ami.load_channels():
        ref = parse_rttm(channel.ref_path)
        hyp = parse_rttm(out_dir / f"{channel.id}.rttm")
        words = _load_words(out_dir / f"{channel.id}.words.json")
        attribution = score_attribution(words, ref)
        if channel.num_speakers > 1:
            ders.append(score_der(ref, hyp).der)
            attrs.append(attribution.accuracy)
        else:
            mic_attrs.append(attribution.accuracy)
    return (
        float(np.mean(ders)),
        float(np.mean(attrs)),
        float(np.mean(mic_attrs)),
    )


def naming_trials(out_dir: Path, embed) -> dict[float | None, list[Trial]]:
    """Same-group naming trials from ``out_dir``'s clusters under ``embed``'s
    galleries, full plus truncated."""
    import ami
    from naming_gate import embed_spans, load_clusters, truncate_spans

    galleries = ami.build_galleries(embed)
    channels = [c for c in ami.load_channels() if c.session != ami.ENROLL_SESSION]
    by_channel: dict[str, list] = {}
    for cluster in load_clusters(out_dir, channels):
        by_channel.setdefault(cluster.channel_id, []).append(cluster)
    group_of = {c.id: c.group for c in channels}

    trials: dict[float | None, list[Trial]] = {b: [] for b in BUDGETS}
    for channel_id, clusters in sorted(by_channel.items()):
        gallery = galleries.get(group_of[channel_id]) or {}
        if not gallery:
            continue
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads((out_dir / f"{channel_id}.emb.json").read_text()).items()
        }
        pcm = read_pcm16(ami.CHANNELS_DIR / f"{channel_id}.wav")
        for cluster in clusters:
            for budget in BUDGETS:
                vector = (
                    embeddings.get(cluster.label)
                    if budget is None
                    else embed_spans(pcm, truncate_spans(cluster.spans, budget), embed)
                )
                if vector is None:
                    continue
                scores = {n: float(vector @ e) for n, e in gallery.items()}
                trials[budget].append(Trial(cluster.name, cluster.true_speaker, scores))
    return trials


def main() -> int:
    from stenograf.diarization.loop import OwnDiarizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", help="comma-separated candidate tags (default: all)")
    args = parser.parse_args()
    wanted = set(args.tags.split(",")) if args.tags else None

    arms: dict[str, Path | None] = {BASELINE_TAG: None}
    for tag, url in CANDIDATES.items():
        if wanted is None or tag in wanted:
            arms[tag] = fetch_model(tag, url)

    lines = [
        "## Embedding-model A/B (embedder_ab.py)",
        "",
        "| arm | loop DER | loop attribution | mic attribution "
        "| DIR@FAR0 full (thr) | DIR@FAR0 3 s | DIR@FAR0 2 s | EER point full |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, model in arms.items():
        out_dir = ensure_matrix(tag, model)
        der, attr, mic_attr = matrix_scores(out_dir)
        embed = OwnDiarizer(embedding_model=model).embed
        trials = naming_trials(out_dir, embed)
        strict = {b: dir_at_far(trials[b], 0.0) for b in BUDGETS}
        eer = eer_point(trials[None])
        lines.append(
            f"| {tag} | {der:.1%} | {attr:.1%} | {mic_attr:.1%} "
            f"| {strict[None].dir_:.1%} ({strict[None].threshold:.3f}) "
            f"| {strict[3.0].dir_:.1%} | {strict[2.0].dir_:.1%} "
            f"| {eer.dir_:.1%} / {eer.far:.1%} / {eer.frr:.1%} @ {eer.threshold:.3f} |"
        )
        known = sum(1 for t in trials[None] if t.known)
        lines.append(
            f"|  | ({len(trials[None])} trials: {known} known) |  |  |  |  |  |  |"
        )

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-embedder-ab.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
