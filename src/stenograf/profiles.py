"""Cross-meeting speaker re-identification: a local profile store + cosine match.

A speaker *profile* is a named mean voice embedding saved across meetings, so a
cluster the diarizer finds in this meeting can be matched to "Daniel" enrolled
from an earlier one (PLAN.md §2 "Cross-meeting speaker re-ID"). The diarizer only
labels voices *within* one run (``S0``/``S1``…); the profile store is what carries
identity *between* runs.

Two facts shape the design:

- **Embeddings are model-bound.** A vector only means anything relative to the
  model that produced it, so every profile records its embedding-model id and a
  match is only ever attempted between vectors from the *same* model. Swapping the
  embedding model (PLAN.md's ResNet293-LM upgrade path) simply starts a fresh,
  disjoint set of profiles rather than silently mis-matching.
- **Profiles are precious user data, not a re-downloadable cache.** The store
  lives in the platform *data* dir (separate from ``models.cache_dir``) and writes
  atomically, so a crash mid-save never corrupts the library.

The store and the cosine match live here, in the core — deliberately *not* in the
diarizer (PLAN.md §2, [[phase3-verified-library-constraints]]): sherpa's
``OfflineSpeakerDiarization`` exposes no embeddings, so the diarizer's job ends at
``diarize_with_embeddings`` handing back a per-cluster mean vector; turning those
into names is this module's job.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD = 0.5
"""Cosine similarity at or above which a cluster is deemed the same speaker as a
stored profile. ~0.5 is PLAN.md's starting point for the shipped eres2net
embedding. It stays at this default rather than being empirically tuned: tuning
needs the hand-labelled 0d reference data, which is not being produced. Override
per run with ``--reid-threshold`` (``steno start``/``transcribe``)."""

_STORE_VERSION = 1


# eq=False: the default field-wise __eq__/__hash__ a frozen dataclass generates
# both break on the ``embedding`` ndarray field — ``==`` raises "truth value of an
# array is ambiguous" and ``hash`` raises "unhashable type: ndarray". Identity
# semantics are what the store actually uses (``remove``/``_replace`` match by
# ``is``, names are unique per model), and they keep a profile safe to put in a set
# or dict key — which a Phase-4 web UI will do (PLAN.md §5 Phase 3→4 audit).
@dataclass(frozen=True, eq=False)
class SpeakerProfile:
    """A named voice, identified by a unit-norm mean embedding under one model.

    ``samples`` counts how many enrolments were averaged into ``embedding`` so the
    mean can be extended incrementally (:meth:`ProfileStore.reinforce`) without
    re-reading past audio.
    """

    name: str
    embedding_model: str
    embedding: np.ndarray  # float32, L2-normalized
    samples: int = 1

    def similarity(self, embedding: np.ndarray) -> float:
        """Cosine similarity to another embedding (both treated as unit vectors)."""
        return float(_unit(self.embedding) @ _unit(np.asarray(embedding, dtype=np.float32)))

    def _to_json(self) -> dict:
        return {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "embedding": [float(x) for x in self.embedding],
            "samples": self.samples,
        }

    @staticmethod
    def _from_json(data: Mapping) -> SpeakerProfile:
        return SpeakerProfile(
            name=data["name"],
            embedding_model=data["embedding_model"],
            embedding=_unit(np.asarray(data["embedding"], dtype=np.float32)),
            samples=int(data.get("samples", 1)),
        )


class ProfileStore:
    """A local, model-scoped library of :class:`SpeakerProfile` s.

    Load with :meth:`load` (a missing file is an empty store), mutate with
    :meth:`enroll`/:meth:`rename`/:meth:`remove`/:meth:`reinforce`, and persist
    with :meth:`save` (atomic). Matching (:meth:`match`) is always scoped to a
    single embedding-model id; the cross-run relabelling that consumes it is
    :class:`SpeakerReID`.
    """

    def __init__(
        self,
        path: Path | None = None,
        profiles: list[SpeakerProfile] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.threshold = threshold
        self._profiles: list[SpeakerProfile] = list(profiles or [])

    # ---- persistence ------------------------------------------------------

    @classmethod
    def load(
        cls, path: Path | None = None, *, threshold: float = DEFAULT_THRESHOLD
    ) -> ProfileStore:
        """Load a store from ``path`` (default location if omitted); empty if absent."""
        path = Path(path) if path is not None else default_store_path()
        if not path.exists():
            return cls(path, threshold=threshold)
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = [SpeakerProfile._from_json(p) for p in data.get("profiles", [])]
        return cls(path, profiles, threshold=threshold)

    def save(self) -> None:
        """Write the store to ``self.path`` atomically (temp file + replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _STORE_VERSION, "profiles": [p._to_json() for p in self._profiles]},
            ensure_ascii=False,
            indent=2,
        )
        with tempfile.NamedTemporaryFile(
            "w", dir=self.path.parent, suffix=".part", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    # ---- reads ------------------------------------------------------------

    def profiles(self) -> list[SpeakerProfile]:
        """Every profile in the store, regardless of model."""
        return list(self._profiles)

    def for_model(self, model: str) -> list[SpeakerProfile]:
        """Profiles produced by embedding model ``model`` — the only ones a vector
        from that model may be compared against."""
        return [p for p in self._profiles if p.embedding_model == model]

    def get(self, name: str, model: str) -> SpeakerProfile | None:
        for p in self._profiles:
            if p.name == name and p.embedding_model == model:
                return p
        return None

    def match(
        self, embedding: np.ndarray, model: str, *, threshold: float | None = None
    ) -> tuple[SpeakerProfile, float] | None:
        """Best profile for ``embedding`` under ``model`` with cosine ≥ threshold.

        Returns ``(profile, score)`` or ``None`` if nothing clears the bar.
        Considers only same-model profiles; :class:`SpeakerReID` layers the
        one-cluster-to-one-profile constraint on top for a whole run.
        """
        threshold = self.threshold if threshold is None else threshold
        vector = _unit(np.asarray(embedding, dtype=np.float32))
        best: tuple[SpeakerProfile, float] | None = None
        for profile in self.for_model(model):
            score = float(profile.embedding @ vector)
            if score >= threshold and (best is None or score > best[1]):
                best = (profile, score)
        return best

    # ---- writes -----------------------------------------------------------

    def enroll(
        self, name: str, embedding: np.ndarray, model: str, *, samples: int = 1
    ) -> SpeakerProfile:
        """Add a new profile. Names are unique per model (a name is a person)."""
        if self.get(name, model) is not None:
            raise ValueError(f"a profile named {name!r} already exists for model {model!r}")
        profile = SpeakerProfile(
            name=name,
            embedding_model=model,
            embedding=_unit(np.asarray(embedding, dtype=np.float32)),
            samples=samples,
        )
        self._profiles.append(profile)
        return profile

    def reinforce(self, profile: SpeakerProfile, embedding: np.ndarray) -> SpeakerProfile:
        """Fold a new observation into ``profile``'s mean (sample-weighted).

        Lets a re-matched cluster strengthen an existing profile over meetings
        without retaining any past audio. Returns the updated profile (the store
        is mutated in place)."""
        vector = _unit(np.asarray(embedding, dtype=np.float32))
        blended = _unit(profile.embedding * profile.samples + vector)
        updated = replace(profile, embedding=blended, samples=profile.samples + 1)
        self._replace(profile, updated)
        return updated

    def rename(self, profile: SpeakerProfile, new_name: str) -> SpeakerProfile:
        """Rename a profile (the Task 1c "name this unmatched speaker" action)."""
        if new_name != profile.name and self.get(new_name, profile.embedding_model) is not None:
            raise ValueError(
                f"a profile named {new_name!r} already exists for model {profile.embedding_model!r}"
            )
        updated = replace(profile, name=new_name)
        self._replace(profile, updated)
        return updated

    def remove(self, profile: SpeakerProfile) -> None:
        self._profiles = [p for p in self._profiles if p is not profile]

    def _replace(self, old: SpeakerProfile, new: SpeakerProfile) -> None:
        self._profiles = [new if p is old else p for p in self._profiles]


class SpeakerReID:
    """Resolves a run's diarization clusters to stored profile names.

    Given the per-cluster mean embeddings from ``diarize_with_embeddings``, returns
    a mapping ``cluster label → profile name`` for the clusters that match a stored
    profile. The matching is **one-to-one**: two clusters the diarizer kept apart
    are two distinct speakers, so they can never collapse onto the same profile —
    the highest-scoring pairs are assigned first (greedy), and a profile or cluster
    already claimed is skipped. Clusters with no embedding or no over-threshold
    match are simply absent from the result; the caller keeps its own label (the
    channel-coarse ``Local-N``/``Remote-M`` template) for those.
    """

    def __init__(
        self,
        store: ProfileStore,
        model: str,
        *,
        threshold: float | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.threshold = store.threshold if threshold is None else threshold

    def resolve(self, embeddings: Mapping[str, np.ndarray]) -> dict[str, str]:
        profiles = self.store.for_model(self.model)
        if not profiles or not embeddings:
            return {}
        units = {
            cluster: _unit(np.asarray(v, dtype=np.float32)) for cluster, v in embeddings.items()
        }
        scored = [
            (score, cluster, p.name)
            for cluster, vec in units.items()
            for p in profiles
            if (score := float(p.embedding @ vec)) >= self.threshold
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        mapping: dict[str, str] = {}
        claimed: set[str] = set()
        for _score, cluster, name in scored:
            if cluster in mapping or name in claimed:
                continue
            mapping[cluster] = name
            claimed.add(name)
        return mapping


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def data_dir() -> Path:
    """Directory for precious user data (speaker profiles, settings), distinct
    from the model cache: ``$STENOGRAF_DATA`` if set, else the platform data
    dir (``%APPDATA%`` on Windows — added with Phase 6, before any Windows
    release, so no pre-existing installs need migrating)."""
    if override := os.environ.get("STENOGRAF_DATA"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "stenograf"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "stenograf"
    xdg = os.environ.get("XDG_DATA_HOME", "~/.local/share")
    return Path(xdg).expanduser() / "stenograf"


def default_store_path() -> Path:
    return data_dir() / "profiles.json"
