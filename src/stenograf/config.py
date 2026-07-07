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
