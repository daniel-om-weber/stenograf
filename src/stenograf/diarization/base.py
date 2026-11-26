"""Diarization interface (who-spoke-when within one channel).

Diarization runs per channel: never across mic + system together, so it only
has to separate voices *within* a channel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    """Cluster label local to this run, e.g. ``"S0"``; mapping to display
    labels (``Local-1`` / ``Remote-2`` / profile names) happens in the core."""
    start: float
    end: float


@dataclass(frozen=True)
class DiarizationResult:
    """Turns plus, optionally, a mean voice embedding per cluster.

    The embedding lets the core match a run's clusters against saved speaker
    profiles for cross-meeting re-ID. Each value is an
    L2-normalized mean embedding keyed by the same cluster label used in
    ``turns`` (e.g. ``"S0"``); a cluster with too little clean audio to embed is
    simply absent from the mapping. Backends that cannot produce embeddings
    return an empty mapping, so callers must treat it as best-effort."""

    turns: list[SpeakerTurn]
    embeddings: dict[str, np.ndarray]


@dataclass(frozen=True)
class SpeakerSplit:
    """A speaker the diarizer found who turned out to be two people, split.

    ``speakers`` gives each segment handed to :meth:`Diarizer.split_speakers`
    its label after the split; ``turns`` are the turns relabelled to match;
    ``halves`` maps each new label to the one it came from; ``embeddings``
    holds the voice of every label in ``halves`` (keys and values) under the
    same rule as :class:`DiarizationResult`'s, and replaces their old ones."""

    speakers: list[str]
    turns: list[SpeakerTurn]
    halves: dict[str, str]
    embeddings: dict[str, np.ndarray]


Reader = Callable[[int, int], np.ndarray]
"""``read(lo, hi)``: samples ``[lo, hi)`` of one channel as float32 16 kHz."""


class DiarizationStream(ABC):
    """One channel's diarization, run while the meeting is still recorded.

    The meeting's streamer thread calls :meth:`advance` as audio arrives; the
    stop diarizes through :meth:`resumed`, which picks up from the work done
    and must give the speakers the offline diarizer gives on the same audio.
    The channel's audio is read only through ``read`` and only below
    ``num_samples`` — the meeting's store is append-only, so what was read
    never changes."""

    @abstractmethod
    def samples_needed(self) -> int | None:
        """How many samples the channel must hold before :meth:`advance` has
        work; ``None`` = it never will (nothing to stream)."""

    @abstractmethod
    def advance(self, read: Reader, num_samples: int, limit: int | None = None) -> int:
        """Do up to ``limit`` units of the work the channel's first
        ``num_samples`` samples already determine; returns how many ran."""

    @abstractmethod
    def interrupt(self) -> None:
        """Abandon an :meth:`advance` running on another thread, as soon as
        possible, without losing the work before it (the meeting stopped)."""

    @abstractmethod
    def resumed(self) -> Diarizer:
        """The diarizer the stop runs on this channel's finished audio."""

    @abstractmethod
    def close(self) -> None:
        """Release what the stream holds; idempotent."""


class Diarizer(ABC):
    def stream(self) -> DiarizationStream | None:
        """A :class:`DiarizationStream` for one channel of a live meeting, or
        ``None`` when this diarizer only runs at the end (the default)."""
        return None

    @abstractmethod
    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        """Segment mono 16 kHz PCM by speaker. ``num_speakers=None`` leaves
        the count to the diarizer; a stated one caps it."""

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        """Diarize and also return a per-cluster voice embedding for re-ID.

        The default runs :meth:`diarize` and returns no embeddings, which is
        what a test double that only needs turns wants; the real diarizer
        overrides it with :class:`~stenograf.diarization.embedding.VoiceEmbedder`."""
        return DiarizationResult(self.diarize(samples, num_speakers), {})

    def split_speakers(
        self,
        samples: np.ndarray,
        turns: list[SpeakerTurn],
        segments: list[tuple[str, float, float]],
        num_speakers: int | None = None,
    ) -> SpeakerSplit | None:
        """Split any of ``turns``' speakers who is two people, judged on the
        transcript's ``segments`` (``(speaker, start, end)`` under the turns'
        labels); ``None`` when nobody splits. The default never splits."""
        return None

    def take_voice_failure(self) -> Exception | None:
        """Why the last :meth:`diarize_with_embeddings` came back with turns
        but no voices, then ``None`` until the next failure.

        A voice model that cannot load or run costs the speakers their names,
        never their separation, so the diarizer keeps the turns and leaves the
        telling to the caller, which owns the user's status surface. ``None``
        by default: a diarizer that does not embed has nothing to report."""
        return None

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        """One voice embedding over a single-speaker channel's speech ``turns``.

        A solo channel never diarizes — one stated speaker has nothing to
        separate — but a meeting with speaker identity on still needs its
        voice, for re-ID naming and for post-meeting enrollment. No
        segmentation runs: the turns are the transcript's own speech spans.
        ``None`` when the backend cannot embed (the default) or the audio
        yields no embeddable span."""
        return None
