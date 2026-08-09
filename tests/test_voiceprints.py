"""Speaker-profile store + cosine re-ID.

Pure unit tests on synthetic unit vectors — no models, no audio. The real
embedding path (``diarize_with_embeddings``) is covered by
``test_diarization_sherpa.py``; here we test the store, the model-bound scoping,
the cosine threshold, persistence (v1 migration included), and
merge-at-naming cluster→profile matching.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stenograf.voiceprints import (
    DEFAULT_THRESHOLD,
    MAX_EMBEDDINGS,
    MeetingEmbedding,
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


def profile(name: str, model: str, *vectors: np.ndarray) -> SpeakerProfile:
    """A profile holding one entry per given (already-normalized) vector."""
    return SpeakerProfile(name, model, tuple(MeetingEmbedding(v) for v in vectors))


# A small orthonormal-ish basis: near-parallel vectors match, orthogonal ones don't.
DANIEL = unit(1.0, 0.0, 0.0)
DANIEL_AGAIN = unit(0.97, 0.24, 0.0)  # cosine ~0.97 with DANIEL
ANNA = unit(0.0, 1.0, 0.0)
CARL = unit(0.0, 0.0, 1.0)


class TestSpeakerProfile:
    def test_similarity_is_cosine_and_normalizes_inputs(self):
        p = profile("Daniel", MODEL, DANIEL)
        assert p.similarity(vec(5.0, 0.0, 0.0)) == pytest.approx(1.0)  # scale-invariant
        assert p.similarity(ANNA) == pytest.approx(0.0, abs=1e-6)

    def test_similarity_averages_scores_over_the_stored_set(self):
        # Score averaging: the mean of the per-meeting cosines, NOT the cosine
        # against a mean vector — DANIEL and ANNA average to 0.5 toward DANIEL.
        p = profile("Daniel", MODEL, DANIEL, ANNA)
        assert p.similarity(DANIEL) == pytest.approx(0.5, abs=1e-6)


class TestProfileStore:
    def test_enroll_normalizes_and_matches_itself(self):
        store = ProfileStore(profiles=[])
        store.enroll("Daniel", vec(3.0, 0.0, 0.0), MODEL)  # unnormalized input
        (enrolled,) = store.for_model(MODEL)
        (entry,) = enrolled.embeddings
        assert np.linalg.norm(entry.vector) == pytest.approx(1.0)
        assert entry.date is not None  # stamped with the meeting date (today)
        matched = store.match(DANIEL, MODEL)
        assert matched is not None and matched[0].name == "Daniel"

    def test_no_match_below_threshold(self):
        store = ProfileStore(profiles=[profile("Daniel", MODEL, DANIEL)])
        assert store.match(ANNA, MODEL) is None  # cosine 0 < 0.5

    def test_match_is_model_scoped(self):
        # A vector only compares against profiles from the *same* embedding model,
        # even if the raw numbers would match perfectly.
        store = ProfileStore(profiles=[profile("Daniel", OTHER_MODEL, DANIEL)])
        assert store.match(DANIEL, MODEL) is None
        assert store.match(DANIEL, OTHER_MODEL) is not None

    def test_match_returns_best_of_several(self):
        store = ProfileStore(
            profiles=[
                profile("Daniel", MODEL, DANIEL),
                profile("Anna", MODEL, ANNA),
            ]
        )
        result = store.match(unit(0.9, 0.4, 0.0), MODEL)
        assert result is not None
        matched, score = result
        assert matched.name == "Daniel"
        assert 0.5 <= score <= 1.0

    def test_enroll_rejects_duplicate_name_per_model(self):
        store = ProfileStore(profiles=[profile("Daniel", MODEL, DANIEL)])
        with pytest.raises(ValueError):
            store.enroll("Daniel", ANNA, MODEL)
        # Same name under a different model is fine (disjoint namespaces).
        store.enroll("Daniel", ANNA, OTHER_MODEL)

    def test_reinforce_appends_a_meeting_embedding(self):
        store = ProfileStore(profiles=[])
        p = store.enroll("Daniel", DANIEL, MODEL)
        before = p.similarity(DANIEL_AGAIN)
        updated = store.reinforce(p, DANIEL_AGAIN)
        assert len(updated.embeddings) == 2
        # The stored set now contains the new sample, so its score went up.
        assert updated.similarity(DANIEL_AGAIN) > before
        # The store now holds the updated profile, not the original.
        assert len(store.for_model(MODEL)[0].embeddings) == 2

    def test_reinforce_evicts_the_oldest_meeting_beyond_the_cap(self):
        store = ProfileStore(profiles=[])
        p = store.enroll("Daniel", DANIEL, MODEL, date="2026-01-01")
        for i in range(MAX_EMBEDDINGS):
            p = store.reinforce(p, DANIEL_AGAIN, date=f"2026-02-{i + 1:02d}")
        assert len(p.embeddings) == MAX_EMBEDDINGS
        assert all(e.date != "2026-01-01" for e in p.embeddings)  # oldest evicted

    def test_reinforce_evicts_undated_migrations_first(self):
        # A v1-migrated entry (date None) goes before any dated meeting.
        p = SpeakerProfile("Daniel", MODEL, (MeetingEmbedding(DANIEL, date=None),))
        store = ProfileStore(profiles=[p])
        for i in range(MAX_EMBEDDINGS):
            p = store.reinforce(p, DANIEL_AGAIN, date=f"2026-02-{i + 1:02d}")
        assert all(e.date is not None for e in p.embeddings)

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
        daniel = store.enroll("Daniel", DANIEL, MODEL, date="2026-08-01")
        store.reinforce(daniel, DANIEL_AGAIN, date="2026-08-06")
        store.enroll("Anna", ANNA, MODEL)
        store.save()

        loaded = ProfileStore.load(path)
        names = {p.name: p for p in loaded.for_model(MODEL)}
        assert set(names) == {"Daniel", "Anna"}
        assert [e.date for e in names["Daniel"].embeddings] == ["2026-08-01", "2026-08-06"]
        assert names["Anna"].similarity(ANNA) == pytest.approx(1.0)

    def test_v1_store_migrates_on_load(self, tmp_path):
        # A v1 file holds one running-mean "embedding" per profile; it becomes
        # the profile's first stored entry (undated) and matching still works.
        path = tmp_path / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": [
                        {
                            "name": "Daniel",
                            "embedding_model": MODEL,
                            "embedding": [float(x) for x in DANIEL],
                            "samples": 3,
                        }
                    ],
                }
            )
        )
        store = ProfileStore.load(path)
        (migrated,) = store.for_model(MODEL)
        assert [e.date for e in migrated.embeddings] == [None]
        matched = store.match(DANIEL_AGAIN, MODEL)
        assert matched is not None and matched[0].name == "Daniel"
        store.save()  # persists in the v2 shape
        saved = json.loads(path.read_text())
        assert saved["version"] == 2
        assert "embeddings" in saved["profiles"][0]

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
        from stenograf import paths

        monkeypatch.delenv("STENOGRAF_DATA", raising=False)
        monkeypatch.setattr(paths.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert paths.data_dir() == tmp_path / "stenograf"


class TestSpeakerReID:
    def _store(self) -> ProfileStore:
        return ProfileStore(
            profiles=[
                profile("Daniel", MODEL, DANIEL),
                profile("Anna", MODEL, ANNA),
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

    def test_split_speaker_merges_at_naming(self):
        # Two clusters both over threshold on Daniel: both take his name — an
        # over-split speaker is made whole by the profile, not kept apart.
        reid = SpeakerReID(self._store(), MODEL)
        mapping = reid.resolve({"S0": DANIEL, "S1": DANIEL_AGAIN})
        assert mapping == {"S0": "Daniel", "S1": "Daniel"}

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
    # 0.56, measured 2026-08-07 on the ward-clustering matrix
    # (eval/threshold_pick.py; curves in eval/README.md).
    assert DEFAULT_THRESHOLD == 0.56


class TestAssign:
    """Enroll-or-reinforce: the store operation behind `steno profiles assign`."""

    def test_new_name_enrolls(self):
        store = ProfileStore(profiles=[])
        p, created = store.assign("Anna", ANNA, MODEL, date="2026-08-06")
        assert created
        assert p.embeddings[0].date == "2026-08-06"
        assert store.get("Anna", MODEL) is p

    def test_existing_name_reinforces(self):
        store = ProfileStore(profiles=[])
        store.assign("Anna", ANNA, MODEL, date="2026-08-01")
        p, created = store.assign("Anna", ANNA, MODEL, date="2026-08-06")
        assert not created
        assert len(p.embeddings) == 2

    def test_scoped_to_the_embedding_model(self):
        store = ProfileStore(profiles=[profile("Anna", OTHER_MODEL, ANNA)])
        p, created = store.assign("Anna", ANNA, MODEL)
        assert created  # the other-model profile is a different vector space
        assert len(store.profiles()) == 2


class TestMeetingVoiceprints:
    """The per-meeting sidecar `steno profiles assign` reads."""

    def test_roundtrip_normalized(self, tmp_path):
        from stenograf.voiceprints import (
            load_meeting_voiceprints,
            write_meeting_voiceprints,
        )

        speakers = {"Remote-1": np.array([3.0, 4.0], np.float32), "Daniel": DANIEL}
        write_meeting_voiceprints(tmp_path, speakers, MODEL, date="2026-08-06")
        loaded = load_meeting_voiceprints(tmp_path)
        assert loaded is not None
        assert loaded.embedding_model == MODEL
        assert loaded.date == "2026-08-06"
        assert set(loaded.speakers) == {"Remote-1", "Daniel"}
        np.testing.assert_allclose(loaded.speakers["Remote-1"], [0.6, 0.8])

    def test_missing_sidecar_is_none(self, tmp_path):
        from stenograf.voiceprints import load_meeting_voiceprints

        assert load_meeting_voiceprints(tmp_path) is None
