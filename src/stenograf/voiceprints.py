"""Cross-meeting speaker re-identification: a local profile store + cosine match.

A speaker *profile* is a named set of per-meeting voice embeddings, so a
cluster the diarizer finds in this meeting can be matched to "Daniel" enrolled
from an earlier one. The diarizer only
labels voices *within* one run (``S0``/``S1``…); the profile store is what carries
identity *between* runs.

Two facts shape the design:

- **Embeddings are model-bound.** A vector only means anything relative to the
  model that produced it, so every profile records its embedding-model id and a
  match is only ever attempted between vectors from the *same* model. Swapping the
  embedding model simply starts a fresh,
  disjoint set of profiles rather than silently mis-matching.
- **Profiles are precious user data, not a re-downloadable cache.** The store
  lives in the platform *data* dir (:func:`stenograf.paths.data_dir`, separate
  from the model cache) and writes atomically, so a crash mid-save never
  corrupts the library.

The store and the cosine match live here, in the core — deliberately *not* in the
diarizer ([[phase3-verified-library-constraints]]): sherpa's
``OfflineSpeakerDiarization`` exposes no embeddings, so the diarizer's job ends at
``diarize_with_embeddings`` handing back a per-cluster mean vector; turning those
into names is this module's job.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date as _date
from pathlib import Path

import numpy as np

from stenograf.audio import l2_normalize
from stenograf.output import atomic_write_text
from stenograf.paths import data_dir

DEFAULT_THRESHOLD = 0.56
"""Cosine similarity at or above which a cluster is deemed the same speaker as
a stored profile. Picked from measured curves (2026-08-07, five-group corpus
harness, `eval/threshold_pick.py`, re-run after the ward clustering swap): the
smallest threshold rejecting every measured stranger in BOTH enrollment arms
at every measured duration, with margin — strangers top out at 0.531
(cross-group diagnostic; 0.529 same-group), so 0.56 keeps a 0.029 margin and
holds FAR 0 % / FRR 0 % at full duration in both arms. The previous 0.62 was
calibrated against complete-linkage clusters, whose impure-enrollment leak
(strangers to 0.860) ward removed; on today's curves it pays 13.7 pts DIR at
3 s of clean speech and 5.9 at 2 s for no FAR gain. Known solo/1:1 matches
score 0.870+, far above. Override per run with ``--reid-threshold``
(``steno start``/``transcribe``)."""

_STORE_VERSION = 2

MAX_EMBEDDINGS = 8
"""Per-meeting embeddings a profile retains; adding one beyond the cap evicts
the oldest meeting first. Bounds the store's growth and match cost while
keeping several meetings' worth of channel and style variety to average
over."""

MEETING_VOICEPRINTS_NAME = "voiceprints.json"
"""Per-meeting sidecar in the meeting folder: each speaker's voice embedding
under the label the transcript shows. What makes ``steno profiles assign``
possible after the audio is gone — the correction's enrollment material,
measured as good as a clean sample (``eval/README.md``, 2026-08-06). Written
whenever the meeting's speaker machinery ran (the diarization switch):
diarized channels contribute per-cluster embeddings, solo channels one over
their transcribed speech — so a 1:1 counterpart is assignable too. With the
switch off there is no sidecar; enroll from a sample instead."""


@dataclass(frozen=True, eq=False)
class MeetingEmbedding:
    """One meeting's voice embedding: a unit-norm vector and the meeting date.

    The date (ISO ``YYYY-MM-DD``; ``None`` for embeddings migrated from a v1
    store) orders eviction when a profile exceeds :data:`MAX_EMBEDDINGS`.
    """

    vector: np.ndarray  # float32, L2-normalized
    date: str | None = None


# eq=False: the default field-wise __eq__/__hash__ a frozen dataclass generates
# both break on ndarray fields — ``==`` raises "truth value of an
# array is ambiguous" and ``hash`` raises "unhashable type: ndarray". Identity
# semantics are what the store actually uses (``remove``/``_replace`` match by
# ``is``, names are unique per model), and they keep a profile safe to put in a set
# or dict key.
@dataclass(frozen=True, eq=False)
class SpeakerProfile:
    """A named voice: up to :data:`MAX_EMBEDDINGS` per-meeting embeddings under
    one model, matched by *score averaging* — the mean cosine against the
    stored set — rather than by collapsing the set to a mean vector. Modern
    speaker embeddings measurably lose accuracy when averaged into one vector
    (the i-vector-era rule is retracted; 2.05 % vs 2.85 % EER for score vs
    embedding averaging in the research record, ``PLAN-DIARIZATION.md``); on
    our own harness the two are within single-trial noise, so the store keeps
    the form the literature favors and later steps need — per-meeting entries
    are what rename-once enrollment appends to and what gated updates can
    evict without poisoning a mean.
    """

    name: str
    embedding_model: str
    embeddings: tuple[MeetingEmbedding, ...]

    def similarity(self, embedding: np.ndarray) -> float:
        """Mean cosine similarity against the stored per-meeting embeddings."""
        other = l2_normalize(np.asarray(embedding, dtype=np.float32))
        return float(np.mean([float(e.vector @ other) for e in self.embeddings]))

    def _to_json(self) -> dict:
        return {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "embeddings": [
                {"vector": [float(x) for x in e.vector], "date": e.date}
                for e in self.embeddings
            ],
        }

    @staticmethod
    def _from_json(data: Mapping) -> SpeakerProfile:
        if "embedding" in data:  # v1: one running mean becomes the first entry
            entries = [{"vector": data["embedding"], "date": None}]
        else:
            entries = data["embeddings"]
        return SpeakerProfile(
            name=data["name"],
            embedding_model=data["embedding_model"],
            embeddings=tuple(
                MeetingEmbedding(
                    vector=l2_normalize(np.asarray(e["vector"], dtype=np.float32)),
                    date=e.get("date"),
                )
                for e in entries
            ),
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
        payload = json.dumps(
            {"version": _STORE_VERSION, "profiles": [p._to_json() for p in self._profiles]},
            ensure_ascii=False,
            indent=2,
        )
        atomic_write_text(self.path, payload)

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
        Considers only same-model profiles; :class:`SpeakerReID` applies this
        per cluster for a whole run.
        """
        threshold = self.threshold if threshold is None else threshold
        best: tuple[SpeakerProfile, float] | None = None
        for profile in self.for_model(model):
            score = profile.similarity(embedding)
            if score >= threshold and (best is None or score > best[1]):
                best = (profile, score)
        return best

    # ---- writes -----------------------------------------------------------

    def enroll(
        self, name: str, embedding: np.ndarray, model: str, *, date: str | None = None
    ) -> SpeakerProfile:
        """Add a new profile from one meeting's embedding. Names are unique per
        model (a name is a person); ``date`` is the meeting's ISO date (today
        when omitted)."""
        if self.get(name, model) is not None:
            raise ValueError(f"a profile named {name!r} already exists for model {model!r}")
        profile = SpeakerProfile(
            name=name,
            embedding_model=model,
            embeddings=(_entry(embedding, date),),
        )
        self._profiles.append(profile)
        return profile

    def reinforce(
        self, profile: SpeakerProfile, embedding: np.ndarray, *, date: str | None = None
    ) -> SpeakerProfile:
        """Add another meeting's embedding to ``profile``'s stored set.

        Lets a re-matched cluster strengthen an existing profile over meetings
        without retaining any past audio. Beyond :data:`MAX_EMBEDDINGS` the
        oldest-dated entry is evicted (undated v1 migrations first). Returns
        the updated profile (the store is mutated in place)."""
        entries = [*profile.embeddings, _entry(embedding, date)]
        if len(entries) > MAX_EMBEDDINGS:
            oldest = min(
                range(len(entries)),
                key=lambda i: (entries[i].date is not None, entries[i].date or "", i),
            )
            del entries[oldest]
        updated = replace(profile, embeddings=tuple(entries))
        self._replace(profile, updated)
        return updated

    def assign(
        self, name: str, embedding: np.ndarray, model: str, *, date: str | None = None
    ) -> tuple[SpeakerProfile, bool]:
        """Enroll ``name`` from ``embedding``, or reinforce the existing profile.

        The store operation behind "this speaker is NAME": one gesture must
        work whether NAME is new or already enrolled, so the caller never has
        to know. Returns ``(profile, created)``."""
        existing = self.get(name, model)
        if existing is None:
            return self.enroll(name, embedding, model, date=date), True
        return self.reinforce(existing, embedding, date=date), False

    def rename(self, profile: SpeakerProfile, new_name: str) -> SpeakerProfile:
        """Rename a profile (the "name this unmatched speaker" action)."""
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


def _entry(embedding: np.ndarray, date: str | None) -> MeetingEmbedding:
    return MeetingEmbedding(
        vector=l2_normalize(np.asarray(embedding, dtype=np.float32)),
        date=date if date is not None else _date.today().isoformat(),
    )


class SpeakerReID:
    """Resolves a run's diarization clusters to stored profile names.

    Given the per-cluster mean embeddings from ``diarize_with_embeddings``, returns
    a mapping ``cluster label → profile name`` for the clusters that match a stored
    profile. Every cluster independently takes its best over-threshold profile, so
    several clusters may resolve to the same name — that **is** the over-split
    recovery (merge-at-naming): a speaker the diarizer split in two is made whole
    by the profile both halves match. The one-to-one constraint this replaces
    measured strictly worse on the corpus harness (2026-08-06, `eval/README.md`):
    it left every profiled split unrecovered and forced an over-split cluster onto
    a *wrong* profile when the right one was already claimed, while its apparent
    stranger protection was incidental — the threshold, not exclusivity, is the
    false-accept control. Clusters with no embedding or no over-threshold match
    are simply absent from the result; the caller keeps its own label (the
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
        mapping: dict[str, str] = {}
        for cluster, vector in embeddings.items():
            best = self.store.match(vector, self.model, threshold=self.threshold)
            if best is not None:
                mapping[cluster] = best[0].name
        return mapping


def default_store_path() -> Path:
    return data_dir() / "profiles.json"


@dataclass(frozen=True)
class MeetingVoiceprints:
    """One meeting's :data:`MEETING_VOICEPRINTS_NAME` sidecar, loaded."""

    embedding_model: str
    date: str | None
    speakers: dict[str, np.ndarray]


def write_meeting_voiceprints(
    out_dir: Path, speakers: Mapping[str, np.ndarray], model: str, *, date: str | None
) -> Path:
    """Write the meeting's speaker embeddings next to its transcript (atomic)."""
    path = out_dir / MEETING_VOICEPRINTS_NAME
    payload = json.dumps(
        {
            "embedding_model": model,
            "date": date,
            "speakers": {
                label: [float(x) for x in vector] for label, vector in speakers.items()
            },
        },
        ensure_ascii=False,
    )
    atomic_write_text(path, payload)
    return path


def load_meeting_voiceprints(meeting_dir: Path) -> MeetingVoiceprints | None:
    """The meeting folder's voiceprints, or ``None`` when the meeting has none
    (pre-sidecar meetings, and meetings without a diarized channel)."""
    path = meeting_dir / MEETING_VOICEPRINTS_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MeetingVoiceprints(
        embedding_model=data["embedding_model"],
        date=data.get("date"),
        speakers={
            label: l2_normalize(np.asarray(vector, dtype=np.float32))
            for label, vector in data["speakers"].items()
        },
    )
