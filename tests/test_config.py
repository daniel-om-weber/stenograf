from pathlib import Path

import pytest

from stenograf.config import (
    Language,
    MeetingMode,
    MeetingProfile,
    Provenance,
    ResolvedValue,
    resolve_value,
)


def test_online_mode():
    profile = MeetingProfile(local_speakers=1, remote_speakers=3)
    assert profile.mode is MeetingMode.ONLINE
    assert profile.needs_system_audio


def test_hybrid_mode():
    profile = MeetingProfile(local_speakers=3, remote_speakers=2)
    assert profile.mode is MeetingMode.HYBRID
    assert profile.needs_system_audio


def test_in_room_mode_skips_system_audio():
    profile = MeetingProfile(local_speakers=4, remote_speakers=0)
    assert profile.mode is MeetingMode.IN_ROOM
    assert not profile.needs_system_audio


def test_unknown_counts_mean_auto():
    profile = MeetingProfile(language=Language.GERMAN)
    assert profile.mode is None
    assert profile.needs_system_audio  # tap stays available until mode is known


def test_in_room_known_without_local_count():
    assert MeetingProfile(remote_speakers=0).mode is MeetingMode.IN_ROOM


def test_speaker_count_bounds():
    with pytest.raises(ValueError):
        MeetingProfile(local_speakers=9)
    with pytest.raises(ValueError):
        MeetingProfile(local_speakers=0, remote_speakers=0)


def test_vocab_fields_default_empty():
    profile = MeetingProfile()
    assert profile.glossary == ()
    assert profile.attendee_names == ()
    assert profile.speaker_profile_store is None


def test_vocab_fields_are_coerced_and_hashable():
    # Lists are coerced to tuples (so the frozen profile stays hashable) and a
    # str store path becomes a Path.
    profile = MeetingProfile(
        glossary=["Kubernetes", "Greifswald"],
        attendee_names=["Daniel"],
        speaker_profile_store="/tmp/store.json",
    )
    assert profile.glossary == ("Kubernetes", "Greifswald")
    assert profile.attendee_names == ("Daniel",)
    assert profile.speaker_profile_store == Path("/tmp/store.json")
    hash(profile)  # must not raise


def test_title_defaults_to_none():
    assert MeetingProfile().title is None


def test_title_is_stripped_and_blank_collapses_to_none():
    assert MeetingProfile(title="  Weekly sync  ").title == "Weekly sync"
    assert MeetingProfile(title="   ").title is None
    assert MeetingProfile(title="").title is None


def test_resolve_value_explicit_wins_over_detected():
    assert resolve_value(Language.GERMAN, Language.ENGLISH) == ResolvedValue(
        Language.GERMAN, Provenance.EXPLICIT
    )


def test_resolve_value_falls_back_to_detected():
    assert resolve_value(None, 3) == ResolvedValue(3, Provenance.DETECTED)


def test_resolve_value_default_when_neither():
    assert resolve_value(None, None) == ResolvedValue(None, Provenance.DEFAULT)


def test_resolve_value_zero_is_a_real_value_not_absent():
    # A listen-only channel (0 local speakers) is an explicit 0, not "unspecified".
    assert resolve_value(0, None) == ResolvedValue(0, Provenance.EXPLICIT)
    assert resolve_value(None, 0) == ResolvedValue(0, Provenance.DETECTED)
