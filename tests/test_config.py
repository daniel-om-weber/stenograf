import pytest

from stenograf.config import Language, MeetingMode, MeetingProfile


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
