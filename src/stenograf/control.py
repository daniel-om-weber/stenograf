"""Reverse-control channel: correcting a meeting after it was captured.

The forward path is capture → finalize → transcript. This module is the *reverse*
path — the one defined way a UI feeds a correction back into a live meeting:

- the finalize pass auto-detected the wrong language, or the user knows the exact
  speaker count → :meth:`MeetingSession.refinalize` re-runs the finalize pass with
  the override, reusing the already-loaded backends (never reloading a model);
- a speaker is mislabelled ``Remote-2`` → :meth:`MeetingSession.rename_speaker`
  relabels that speaker across the transcript, a pure display change.

Because the full audio is retained in RAM and the backends stay warm, both
corrections are cheap: re-finalize is seconds, rename is instant. This replaces
the informal ``stop_callback`` the TUI passed around — a :class:`MeetingSession`
is the single typed object a view (the TUI today, the web server in Stage C)
drives a meeting through: :meth:`stop` it, :meth:`refinalize` it, rename its
speakers. The archived-meeting twin (correcting a meeting after its process is
gone) is :class:`~stenograf.archive` Stage B4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from stenograf.capture.base import AudioFrame, Channel
from stenograf.pipeline import rename_entry_speaker
from stenograf.recording import read_channels
from stenograf.session import SessionStore, plan_channels

if TYPE_CHECKING:
    from stenograf.archive import MeetingArchive, MeetingRecord
    from stenograf.config import Language
    from stenograf.pipeline import SpeakerResolver
    from stenograf.session import MeetingRecorder
    from stenograf.transcript import Transcript


@dataclass(frozen=True)
class FinalizeRequest:
    """A correction to apply on the next finalize; every field is optional.

    ``None`` means *keep whatever the session already resolved* — so an empty
    request re-runs finalize unchanged, and a request that sets only
    ``remote_speakers`` corrects that one parameter while a previously locked
    language and the other channel's count stay put. The counts follow the same
    0–8 validation as :class:`~stenograf.config.MeetingProfile` (a request that
    would leave a meeting with zero speakers raises when applied).
    """

    local_speakers: int | None = None
    remote_speakers: int | None = None
    language: Language | None = None
    reid: bool | None = None
    """Toggle cross-meeting speaker re-ID for this finalize: ``True`` enables it
    (a no-op if the session has no resolver), ``False`` disables it, ``None``
    keeps the session's current setting. Toggling is sticky across refinalizes."""


class MeetingSession:
    """A live meeting a view can correct: the recorder + its retained store + text.

    Holds the warm :class:`~stenograf.session.MeetingRecorder` (backends loaded
    once), the in-RAM :class:`~stenograf.session.SessionStore` the meeting filled,
    and the ``transcript`` currently shown. :meth:`refinalize` overrides the
    recorder's profile in place and re-runs its finalize — the same backend
    objects, so no model is reloaded; :meth:`rename_speaker` relabels one speaker
    on the current transcript without re-running anything. Both update
    :attr:`transcript` and return it. :meth:`stop` is the formalized capture-stop
    hook (the old ``stop_callback``), a no-op once the meeting has finalized.

    Construct it around a meeting: create it before finalize with ``stop`` wired
    (the web ``/ws`` stop path) and ``transcript=None``, or after finalize with
    the returned transcript for the correction-only archive/reader flow.
    """

    def __init__(
        self,
        recorder: MeetingRecorder,
        store: SessionStore,
        *,
        transcript: Transcript | None = None,
        stop: Callable[[], None] | None = None,
    ) -> None:
        self.recorder = recorder
        self.store = store
        self.transcript = transcript
        self._stop = stop
        # Hold the resolver even while re-ID is toggled off, so ``reid=True`` can
        # switch it back on later (the recorder's own ``reid`` gets set to None).
        self._resolver: SpeakerResolver | None = recorder.reid

    def stop(self) -> None:
        """End capture (the one defined stop path); harmless once finalized."""
        if self._stop is not None:
            self._stop()

    def refinalize(self, request: FinalizeRequest) -> Transcript:
        """Re-run the finalize pass with ``request``'s overrides applied.

        Overrides are applied *in place* on the recorder and are sticky: the
        session now reflects the corrected parameters, so a later empty request
        re-finalizes with them still in effect. An explicit language is recorded
        as such (``explicit`` provenance); a previously auto-detected one is kept
        and stays ``detected``. The already-loaded ASR/diarizer are reused — this
        never reloads a model. Updates and returns :attr:`transcript`.
        """
        overrides: dict[str, object] = {}
        if request.local_speakers is not None:
            overrides["local_speakers"] = request.local_speakers
        if request.remote_speakers is not None:
            overrides["remote_speakers"] = request.remote_speakers
        if request.language is not None:
            overrides["language"] = request.language
            # Also drive the ASR/LID language directly: the recorder may hold a
            # locked auto-detected language from an earlier finalize, and the
            # profile alone would not override it.
            self.recorder.language = request.language
        if overrides:
            self.recorder.profile = replace(self.recorder.profile, **overrides)
        if request.reid is not None:
            self.recorder.reid = self._resolver if request.reid else None
        self.transcript = self.recorder.finalize(self.store)
        return self.transcript

    def rename_speaker(self, old: str, new: str) -> Transcript:
        """Relabel the speaker ``old`` to ``new`` across the current transcript.

        A pure display rename (timestamps, text, words untouched); labels other
        than ``old`` pass through. Requires a finalized transcript — there is
        nothing to rename before the first finalize. Updates and returns
        :attr:`transcript`.
        """
        if self.transcript is None:
            raise ValueError("no transcript to rename yet; finalize the meeting first")
        renamed = rename_entry_speaker(self.transcript.entries, old, new)
        self.transcript = replace(self.transcript, entries=renamed)
        return self.transcript


class AudioUnavailable(Exception):
    """Re-finalizing an archived meeting needs its audio, which was not retained.

    stenograf keeps audio in RAM only; a meeting recorded without ``--record-audio``
    has no WAV to re-run the finalize pass over once its process is gone. The text
    is still fully editable — :meth:`ArchivedMeeting.rename_speaker` always works —
    but re-diarize / re-transcribe cannot (PLAN.md §5 Stage B4)."""

    def __init__(self, meeting_id: str) -> None:
        super().__init__(
            f"meeting {meeting_id!r} has no retained audio to re-finalize "
            "(record with --record-audio to enable archived re-finalize)"
        )
        self.meeting_id = meeting_id


class ArchivedMeeting:
    """Reverse control over a meeting whose live process is gone (Stage B4).

    The archived twin of :class:`MeetingSession`: it loads a meeting's transcript
    from the archive and applies the same two corrections, persisting each to disk
    under the meeting's stable id (:meth:`MeetingArchive.rewrite`).

    - :meth:`rename_speaker` **always** works — it is a pure relabel of the loaded
      transcript, needing no audio.
    - :meth:`refinalize` works **only when** :meth:`MeetingRecord.has_audio` — the
      one predicate gating everything that contradicts the in-RAM-only guarantee.
      It rehydrates a :class:`~stenograf.session.SessionStore` from the recorded
      WAV and re-runs the finalize pass; without a recording it raises
      :class:`AudioUnavailable`. Targets a live-captured ``--record-audio`` meeting
      (mic/system WAV); an imported non-recording source is not re-finalizable here.
    """

    def __init__(
        self,
        archive: MeetingArchive,
        record: MeetingRecord,
        *,
        transcript: Transcript | None = None,
    ) -> None:
        self.archive = archive
        self.record = record
        self.transcript = (
            transcript if transcript is not None else archive.load_transcript(record.id)
        )

    def rename_speaker(self, old: str, new: str) -> Transcript:
        """Relabel a speaker across the transcript and persist it under the same id.

        Always available (no audio needed): relabel the loaded transcript, rewrite
        the managed transcript files, and refresh the index record. Updates and
        returns :attr:`transcript`.
        """
        self.transcript = replace(
            self.transcript, entries=rename_entry_speaker(self.transcript.entries, old, new)
        )
        self.record = self.archive.rewrite(self.record, self.transcript)
        return self.transcript

    def refinalize(self, request: FinalizeRequest, *, recorder: MeetingRecorder) -> Transcript:
        """Re-run the finalize pass over the recorded audio with ``request`` applied.

        Requires :meth:`MeetingRecord.has_audio` — else :class:`AudioUnavailable`.
        Rehydrates the per-channel store from the recorded WAV (the meeting's own
        captured channels), anchors the freshly-loaded ``recorder`` to this
        meeting's archived profile and language, then delegates to a
        :class:`MeetingSession` so the exact same override/provenance rules as the
        live path apply. The result is written back under the same id.

        ``recorder`` carries the (re)loaded backends and, for a re-ID toggle, the
        resolver; its profile/language are overwritten from the archive, so the
        caller need not reconstruct them.
        """
        if not self.record.has_audio():
            raise AudioUnavailable(self.record.id)
        assert self.record.audio_path is not None  # guaranteed by has_audio()
        recorder.profile = self.transcript.profile
        recorder.language = self.transcript.language
        channels = [plan.channel for plan in plan_channels(self.transcript.profile)]
        store = _store_from_channels(read_channels(self.record.audio_path, channels))
        session = MeetingSession(recorder, store, transcript=self.transcript)
        self.transcript = session.refinalize(request)
        self.record = self.archive.rewrite(self.record, self.transcript)
        return self.transcript


def _store_from_channels(channels: dict[Channel, np.ndarray]) -> SessionStore:
    """Build a :class:`~stenograf.session.SessionStore` from per-channel int16 PCM,
    each channel anchored at ``t=0`` (the recording's shared clock)."""
    store = SessionStore(set(channels))
    for channel, samples in channels.items():
        if len(samples):
            store.append(AudioFrame(channel=channel, timestamp=0.0, samples=samples))
    return store
