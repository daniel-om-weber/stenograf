"""Speaker-profile store + cosine re-ID (Phase 3, Task 1b).

Pure unit tests on synthetic unit vectors — no models, no audio. The real
embedding path (``diarize_with_embeddings``) is covered by
``test_diarization_sherpa.py``; here we test the store, the model-bound scoping,
the cosine threshold, persistence, and the one-to-one cluster→profile matching.
"""

from __future__ import annotations

import numpy as np
import pytest

from stenograf.profiles import (
    DEFAULT_THRESHOLD,
    ProfileStore,
    SpeakerProfile,
    SpeakerReID,
    default_store_path,
)

MODEL = "eres2net-voxceleb-16k.onnx"
OTHER_MODEL = "resnet293-lm.onnx"


def vec(*components: float) -> np.ndarray:
    """A (non-normalized) embedding; the store normalizes on the way in."""
    return np.asarray(components, dtype=np.float32)


def unit(*components: float) -> np.ndarray:
    v = vec(*components)
    return v / np.linalg.norm(v)


# A small orthonormal-ish basis: near-parallel vectors match, orthogonal ones don't.
DANIEL = unit(1.0, 0.0, 0.0)
DANIEL_AGAIN = unit(0.97, 0.24, 0.0)  # cosine ~0.97 with DANIEL
ANNA = unit(0.0, 1.0, 0.0)
CARL = unit(0.0, 0.0, 1.0)


class TestSpeakerProfile:
    def test_similarity_is_cosine_and_normalizes_inputs(self):
        p = SpeakerProfile("Daniel", MODEL, DANIEL)
        assert p.similarity(vec(5.0, 0.0, 0.0)) == pytest.approx(1.0)  # scale-invariant
        assert p.similarity(ANNA) == pytest.approx(0.0, abs=1e-6)


class TestProfileStore:
    def test_enroll_normalizes_and_matches_itself(self):
        store = ProfileStore(profiles=[])
        store.enroll("Daniel", vec(3.0, 0.0, 0.0), MODEL)  # unnormalized input
        (profile,) = store.for_model(MODEL)
        assert np.linalg.norm(profile.embedding) == pytest.approx(1.0)
        matched = store.match(DANIEL, MODEL)
        assert matched is not None and matched[0].name == "Daniel"

    def test_no_match_below_threshold(self):
        store = ProfileStore(profiles=[SpeakerProfile("Daniel", MODEL, DANIEL)])
        assert store.match(ANNA, MODEL) is None  # cosine 0 < 0.5

    def test_match_is_model_scoped(self):
        # A vector only compares against profiles from the *same* embedding model,
        # even if the raw numbers would match perfectly.
        store = ProfileStore(profiles=[SpeakerProfile("Daniel", OTHER_MODEL, DANIEL)])
        assert store.match(DANIEL, MODEL) is None
        assert store.match(DANIEL, OTHER_MODEL) is not None

    def test_match_returns_best_of_several(self):
        store = ProfileStore(
            profiles=[
                SpeakerProfile("Daniel", MODEL, DANIEL),
                SpeakerProfile("Anna", MODEL, ANNA),
            ]
        )
        result = store.match(unit(0.9, 0.4, 0.0), MODEL)
        assert result is not None
        profile, score = result
        assert profile.name == "Daniel"
        assert 0.5 <= score <= 1.0

    def test_enroll_rejects_duplicate_name_per_model(self):
        store = ProfileStore(profiles=[SpeakerProfile("Daniel", MODEL, DANIEL)])
        with pytest.raises(ValueError):
            store.enroll("Daniel", ANNA, MODEL)
        # Same name under a different model is fine (disjoint namespaces).
        store.enroll("Daniel", ANNA, OTHER_MODEL)

    def test_reinforce_blends_toward_new_observation(self):
        store = ProfileStore(profiles=[])
        p = store.enroll("Daniel", DANIEL, MODEL)
        before = p.similarity(DANIEL_AGAIN)
        updated = store.reinforce(p, DANIEL_AGAIN)
        assert updated.samples == 2
        assert np.linalg.norm(updated.embedding) == pytest.approx(1.0)
        # The mean moved toward the new sample, so similarity to it went up.
        assert updated.similarity(DANIEL_AGAIN) > before
        # The store now holds the updated profile, not the original.
        assert store.for_model(MODEL)[0].samples == 2

    def test_rename_and_remove(self):
        store = ProfileStore(profiles=[])
        p = store.enroll("Speaker 1", DANIEL, MODEL)
        renamed = store.rename(p, "Daniel")
        assert store.get("Daniel", MODEL) is not None
        assert store.get("Speaker 1", MODEL) is None
        store.remove(renamed)
        assert store.for_model(MODEL) == []

    def test_rename_rejects_collision(self):
        store = ProfileStore(profiles=[])
        store.enroll("Daniel", DANIEL, MODEL)
        anna = store.enroll("Anna", ANNA, MODEL)
        with pytest.raises(ValueError):
            store.rename(anna, "Daniel")


class TestPersistence:
    def test_roundtrip_preserves_profiles(self, tmp_path):
        path = tmp_path / "profiles.json"
        store = ProfileStore(path)
        store.enroll("Daniel", DANIEL, MODEL)
        store.enroll("Anna", ANNA, MODEL, samples=3)
        store.save()

        loaded = ProfileStore.load(path)
        names = {p.name: p for p in loaded.for_model(MODEL)}
        assert set(names) == {"Daniel", "Anna"}
        assert names["Anna"].samples == 3
        assert names["Daniel"].similarity(DANIEL) == pytest.approx(1.0)

    def test_missing_file_is_empty_store(self, tmp_path):
        store = ProfileStore.load(tmp_path / "absent.json")
        assert store.profiles() == []

    def test_save_is_atomic_no_partials(self, tmp_path):
        path = tmp_path / "profiles.json"
        store = ProfileStore(path)
        store.enroll("Daniel", DANIEL, MODEL)
        store.save()
        # No leftover temp file, only the final store.
        assert [p.name for p in tmp_path.iterdir()] == ["profiles.json"]

    def test_default_store_path_uses_data_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
        assert default_store_path() == tmp_path / "profiles.json"

    def test_data_dir_windows_default(self, tmp_path, monkeypatch):
        from stenograf import profiles

        monkeypatch.delenv("STENOGRAF_DATA", raising=False)
        monkeypatch.setattr(profiles.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert profiles.data_dir() == tmp_path / "stenograf"


class TestSpeakerReID:
    def _store(self) -> ProfileStore:
        return ProfileStore(
            profiles=[
                SpeakerProfile("Daniel", MODEL, DANIEL),
                SpeakerProfile("Anna", MODEL, ANNA),
            ]
        )

    def test_resolves_matching_clusters_to_names(self):
        reid = SpeakerReID(self._store(), MODEL)
        mapping = reid.resolve({"S0": DANIEL_AGAIN, "S1": ANNA})
        assert mapping == {"S0": "Daniel", "S1": "Anna"}

    def test_unmatched_cluster_is_omitted(self):
        # CARL matches no profile → absent from the mapping; the caller keeps S<n>.
        reid = SpeakerReID(self._store(), MODEL)
        mapping = reid.resolve({"S0": DANIEL, "S1": CARL})
        assert mapping == {"S0": "Daniel"}

    def test_matching_is_one_to_one(self):
        # Two clusters both closest to Daniel: only the better one may take him;
        # the other cannot collapse onto the same profile.
        reid = SpeakerReID(self._store(), MODEL)
        mapping = reid.resolve({"S0": DANIEL, "S1": DANIEL_AGAIN})
        assert list(mapping.values()).count("Daniel") == 1
        assert mapping.get("S0") == "Daniel"  # the exact match wins the tie-break
        assert "S1" not in mapping  # loser stays a raw cluster

    def test_empty_when_no_profiles_for_model(self):
        reid = SpeakerReID(self._store(), OTHER_MODEL)  # store has none under this model
        assert reid.resolve({"S0": DANIEL}) == {}

    def test_empty_embeddings(self):
        reid = SpeakerReID(self._store(), MODEL)
        assert reid.resolve({}) == {}

    def test_threshold_override(self):
        # A strict threshold rejects an otherwise-good match.
        reid = SpeakerReID(self._store(), MODEL, threshold=0.999)
        assert reid.resolve({"S0": DANIEL_AGAIN}) == {}


def test_default_threshold_matches_plan():
    assert DEFAULT_THRESHOLD == 0.5
