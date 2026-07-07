"""Meeting configuration.

Every parameter follows one resolution order: explicit user setting >
auto-detected value > safe default. ``None`` always means "determine
automatically".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Language(StrEnum):
    GERMAN = "de"
    ENGLISH = "en"


class Provenance(StrEnum):
    """Where a resolved meeting parameter's value came from.

    Mirrors the resolution order (explicit user setting > auto-detected value >
    safe default): once a parameter is filled, its value alone no longer says
    whether the user set it, the finalize pass detected it, or it fell to a
    default. Recording the provenance next to the value keeps that distinction
    on the persisted transcript (PLAN.md §5 Task 3b)."""

    EXPLICIT = "explicit"  # the user set it
    DETECTED = "detected"  # the finalize pass auto-detected/estimated it
    DEFAULT = "default"  # neither given nor detected — left at a safe default


@dataclass(frozen=True)
class ResolvedValue:
    """A resolved parameter value paired with how it was arrived at."""

    value: object | None
    provenance: Provenance


def resolve_value(explicit: object | None, detected: object | None) -> ResolvedValue:
    """Resolve one parameter by the standard order and tag its provenance.

    ``explicit`` is the user-supplied value (``None`` = unspecified); ``detected``
    is what the finalize pass found (``None`` = nothing detected). An explicit
    value always wins; else a detected value; else the parameter is unresolved and
    left at a default. ``0`` is a real explicit/detected value (e.g. a listen-only
    channel with 0 local speakers), so ``None`` — not falsiness — marks "absent".
    """
    if explicit is not None:
        return ResolvedValue(explicit, Provenance.EXPLICIT)
    if detected is not None:
        return ResolvedValue(detected, Provenance.DETECTED)
    return ResolvedValue(None, Provenance.DEFAULT)


@dataclass(frozen=True)
class ResolvedParameters:
    """The meeting parameters as finally resolved, each tagged with provenance.

    Written back onto the finalized :class:`~stenograf.transcript.Transcript` so a
    reader (or a re-run) can tell a user-set value from a detected one — the plan's
    "Detected: German, 2 remote speakers" editability made durable. ``speakers`` is
    keyed by channel (``mic``/``system`` for a meeting, ``audio`` for a file
    transcribe); meeting-mode provenance is deferred with mode auto-detection
    (PLAN.md §5 Task 3b)."""

    language: ResolvedValue
    speakers: dict[str, ResolvedValue] = field(default_factory=dict)


class MeetingMode(StrEnum):
    ONLINE = "online"  # 1 local speaker, N remote
    HYBRID = "hybrid"  # N local speakers, M remote
    IN_ROOM = "in_room"  # N local speakers, no remote audio


@dataclass(frozen=True)
class MeetingProfile:
    """User-provided meeting parameters; ``None`` fields are auto-detected."""

    language: Language | None = None
    local_speakers: int | None = None
    remote_speakers: int | None = None
    glossary: tuple[str, ...] = ()
    """Domain terms to snap the finalized transcript to (Parakeet has no
    decode-time biasing, so vocabulary is a text post-correction — see
    ``stenograf.glossary`` and PLAN.md §5 Task 2b)."""
    attendee_names: tuple[str, ...] = ()
    """Participant names, corrected like the glossary (also token-by-token)."""
    speaker_profile_store: Path | None = field(default=None)
    """Override for the cross-meeting re-ID profile store; ``None`` = default store."""
    title: str | None = None
    """Human-readable meeting title. Surfaced by the meeting archive record and fed
    to the notes-enhancement prompt (PLAN.md §5 Stage A2); ``None`` = untitled."""

    def __post_init__(self) -> None:
        for name in ("local_speakers", "remote_speakers"):
            count = getattr(self, name)
            if count is not None and not 0 <= count <= 8:
                raise ValueError(f"{name} must be between 0 and 8, got {count}")
        if self.local_speakers == 0 and self.remote_speakers == 0:
            raise ValueError("a meeting needs at least one speaker")
        # Normalize the free-form fields so the profile stays hashable/serializable
        # regardless of what the caller passed (a list of terms, a str path).
        object.__setattr__(self, "glossary", tuple(self.glossary))
        object.__setattr__(self, "attendee_names", tuple(self.attendee_names))
        if self.speaker_profile_store is not None:
            object.__setattr__(self, "speaker_profile_store", Path(self.speaker_profile_store))
        if self.title is not None:
            # Collapse a blank/whitespace-only title to the single "untitled" form
            # so an empty string and ``None`` don't read as two different states.
            object.__setattr__(self, "title", self.title.strip() or None)

    @property
    def mode(self) -> MeetingMode | None:
        """Meeting mode implied by the speaker counts; ``None`` if undetermined."""
        if self.remote_speakers == 0:
            return MeetingMode.IN_ROOM
        if self.local_speakers is None or self.remote_speakers is None:
            return None
        if self.local_speakers <= 1:
            return MeetingMode.ONLINE
        return MeetingMode.HYBRID

    @property
    def needs_system_audio(self) -> bool:
        """The system-audio tap is only started when remote audio can exist."""
        return self.mode is not MeetingMode.IN_ROOM
