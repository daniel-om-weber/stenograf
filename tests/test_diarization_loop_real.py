"""Real-backend OwnDiarizer test — the production diarization loop, no fakes.

Every other diarizer test uses ``FakeDiarizer`` or fake extractors; this one
drives the actual ONNX pipeline (pyannote segmentation-3.0 via onnxruntime +
eres2net embeddings via sherpa's extractor) — the surface speaker re-ID
extends. Precedent for why it matters: the MLX thread-stream bug showed a
real backend can break what every mocked test passes green.

It is **gated**: it runs only when sherpa-onnx is installed, both diarization
models are already cached, and a real eval clip is present (the audio is
gitignored private meeting content, the models are a multi-hundred-MB download).
A fresh checkout / CI therefore skips it; run it on a dev machine that has done
the eval setup. Assertions are structural (turns are well-formed, sorted, in
bounds; the known count caps clusters; the threshold estimate path runs)
rather than accuracy numbers — DER scoring is ``eval/der.py``'s job.
"""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stenograf import assets
from stenograf.diarization.base import SpeakerTurn

CLIP = Path(__file__).resolve().parents[1] / "eval" / "audio" / "de-1.wav"
CLIP_SECONDS = 40  # long enough that a second speaker appears in de-1


def _sherpa_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return True


def _models_cached() -> bool:
    return (
        assets.cached_path(assets.PYANNOTE_SEGMENTATION) is not None
        and assets.cached_path(assets.SPEAKER_EMBEDDING) is not None
    )


pytestmark = pytest.mark.skipif(
    not (_sherpa_available() and _models_cached() and CLIP.exists()),
    reason="real loop test needs sherpa-onnx, cached diarization models, and eval audio",
)


def _load_clip() -> np.ndarray:
    with wave.open(str(CLIP), "rb") as wv:
        rate = wv.getframerate()
        frames = wv.readframes(min(wv.getnframes(), CLIP_SECONDS * rate))
    return np.frombuffer(frames, dtype=np.int16)


@pytest.fixture(scope="module")
def diarized() -> SimpleNamespace:
    """Run the real loop once (inference is the slow part) and stash results."""
    from stenograf.diarization.loop import OwnDiarizer

    pcm = _load_clip()
    diarizer = OwnDiarizer()

    known2 = diarizer.diarize(pcm, num_speakers=2)
    known3 = diarizer.diarize(pcm, num_speakers=3)
    estimated = diarizer.diarize(pcm, num_speakers=None)
    embedded = diarizer.diarize_with_embeddings(pcm, num_speakers=2)

    return SimpleNamespace(
        duration=len(pcm) / 16000,
        known2=known2,
        known3=known3,
        estimated=estimated,
        embedded=embedded,
    )


def _assert_well_formed(turns: list[SpeakerTurn], duration: float) -> None:
    assert turns, "real speech should yield at least one turn"
    assert all(isinstance(t, SpeakerTurn) for t in turns)
    assert turns == sorted(turns, key=lambda t: t.start)  # diarize returns sorted
    for t in turns:
        assert 0.0 <= t.start < t.end <= duration + 0.5  # within the clip, non-empty


def test_known_count_returns_well_formed_capped_turns(diarized):
    _assert_well_formed(diarized.known2, diarized.duration)
    # maxclust: requesting 2 can never emit more than 2 speakers.
    assert len({t.speaker for t in diarized.known2}) <= 2
    _assert_well_formed(diarized.known3, diarized.duration)
    assert len({t.speaker for t in diarized.known3}) <= 3


def test_estimation_runs_unconstrained(diarized):
    # num_speakers=None → the reference complete-linkage threshold cut
    # (production reaches it on stenodiar-less installs). It must run without
    # raising and produce well-formed turns.
    _assert_well_formed(diarized.estimated, diarized.duration)


def test_embeddings_are_per_cluster_and_normalized(diarized):
    # diarize_with_embeddings returns the same turns plus one L2-normalized
    # mean voice embedding per cluster that had embeddable audio.
    result = diarized.embedded
    assert [t.speaker for t in result.turns] == [t.speaker for t in diarized.known2]
    assert result.embeddings, "real speech should yield at least one cluster embedding"
    clusters = {t.speaker for t in result.turns}
    for speaker, vector in result.embeddings.items():
        assert speaker in clusters  # keyed by the run's cluster labels
        assert vector.shape == (192,)  # eres2net embedding dim
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_distinct_clusters_have_distinct_embeddings(diarized):
    # If two clusters were embedded they must not be near-identical vectors —
    # a guard against every cluster collapsing to the same embedding.
    vectors = list(diarized.embedded.embeddings.values())
    if len(vectors) < 2:
        pytest.skip("only one cluster embedded in this clip")
    assert float(vectors[0] @ vectors[1]) < 0.99


def test_reid_round_trips_real_cluster_embeddings(diarized):
    # Re-ID end-to-end on REAL eres2net vectors (fakes can't catch a
    # dimension/geometry/threshold surprise): enrol each cluster as a profile,
    # then the resolver must re-identify each cluster as itself. A self-match is
    # cosine 1.0 — the maximum — so with the one-to-one constraint every cluster
    # maps back to its own name.
    from stenograf.voiceprints import ProfileStore, SpeakerReID

    embeddings = diarized.embedded.embeddings
    model = assets.SPEAKER_EMBEDDING.name
    store = ProfileStore(profiles=[])
    for cluster, vector in embeddings.items():
        store.enroll(f"person-{cluster}", vector, model)

    mapping = SpeakerReID(store, model).resolve(embeddings)
    assert mapping == {cluster: f"person-{cluster}" for cluster in embeddings}

    # Scoped to the model: a query under a different embedding model matches nothing.
    assert SpeakerReID(store, "some-other-model.onnx").resolve(embeddings) == {}
