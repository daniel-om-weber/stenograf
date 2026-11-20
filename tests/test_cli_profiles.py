"""`steno profiles`: voiceprint enrollment, listing, and re-ID relabeling."""


from click.testing import CliRunner
from conftest import (
    CliASR,
    EnrollmentDiarizer,
    write_wav,
)

from stenograf import cli, loaders
from stenograf.diarization.base import SpeakerTurn

# ---- speaker profiles ------------------------------------------------------


def _patch_diarizer(monkeypatch, diarizer):
    monkeypatch.setattr(loaders, "load_diarizer", lambda: diarizer)


def test_profiles_list_empty(tmp_path, monkeypatch):
    result = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert result.exit_code == 0, result.output
    assert "no speaker profiles yet" in result.output


def test_profiles_enroll_then_list(tmp_path, monkeypatch):
    _patch_diarizer(monkeypatch, EnrollmentDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "daniel.wav"
    write_wav(audio)

    enroll = CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])
    assert enroll.exit_code == 0, enroll.output
    assert "enrolled 'Daniel'" in enroll.output

    listing = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Daniel" in listing.output
    assert "(1 sample)" in listing.output


def test_profiles_enroll_duplicate_then_reinforce(tmp_path, monkeypatch):
    _patch_diarizer(monkeypatch, EnrollmentDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "a.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])

    dup = CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])
    assert dup.exit_code != 0
    assert "--reinforce" in dup.output  # points the user at the right flag

    again = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Daniel", str(audio), "--reinforce"]
    )
    assert again.exit_code == 0, again.output
    assert "2 samples" in again.output


def test_profiles_enroll_reinforce_refuses_a_mismatched_voice_until_forced(
    tmp_path, monkeypatch
):
    # The same guard `profiles assign` has: a wrong voice in a one-entry
    # profile drags every later match.
    audio = tmp_path / "a.wav"
    write_wav(audio)
    _patch_diarizer(monkeypatch, EnrollmentDiarizer({"S0": [1.0, 0, 0]}))
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])

    _patch_diarizer(monkeypatch, EnrollmentDiarizer({"S0": [0, 1.0, 0]}))
    refused = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Daniel", str(audio), "--reinforce"]
    )
    assert refused.exit_code != 0
    assert "--force" in refused.output
    listing = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "(1 sample)" in listing.output

    forced = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Daniel", str(audio), "--reinforce", "--force"]
    )
    assert forced.exit_code == 0, forced.output
    assert "2 samples" in forced.output


def test_profiles_enroll_multispeaker_needs_speaker_choice(tmp_path, monkeypatch):
    diar = EnrollmentDiarizer(
        {"S0": [1.0, 0, 0], "S1": [0, 1.0, 0]},
        turns=[SpeakerTurn("S0", 0.0, 2.0), SpeakerTurn("S1", 2.0, 3.0)],
    )
    _patch_diarizer(monkeypatch, diar)
    audio = tmp_path / "m.wav"
    write_wav(audio)

    ambiguous = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Anna", str(audio), "--speakers", "2"]
    )
    assert ambiguous.exit_code != 0
    assert "S0" in ambiguous.output and "S1" in ambiguous.output  # lists the choices

    chosen = CliRunner().invoke(
        cli.main, ["profiles", "enroll", "Anna", str(audio), "--speakers", "2", "--speaker", "S1"]
    )
    assert chosen.exit_code == 0, chosen.output


def test_profiles_rename_and_remove(tmp_path, monkeypatch):
    _patch_diarizer(monkeypatch, EnrollmentDiarizer({"S0": [1.0, 0, 0]}))
    audio = tmp_path / "a.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Speaker 1", str(audio)])

    renamed = CliRunner().invoke(cli.main, ["profiles", "rename", "Speaker 1", "Daniel"])
    assert renamed.exit_code == 0, renamed.output
    after = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Daniel" in after.output and "Speaker 1" not in after.output

    removed = CliRunner().invoke(cli.main, ["profiles", "remove", "Daniel", "--yes"])
    assert removed.exit_code == 0, removed.output
    assert "no speaker profiles yet" in CliRunner().invoke(cli.main, ["profiles", "list"]).output


def test_transcribe_reid_relabels_enrolled_speaker(tmp_path, monkeypatch):
    # End-to-end: enroll Daniel, then a diarized transcribe relabels his cluster
    # to "Daniel" instead of the generic "Speaker 1"; --no-reid restores it.
    diar = EnrollmentDiarizer({"S0": [1.0, 0, 0]})
    _patch_diarizer(monkeypatch, diar)
    audio = tmp_path / "m.wav"
    write_wav(audio)
    CliRunner().invoke(cli.main, ["profiles", "enroll", "Daniel", str(audio)])

    monkeypatch.setattr(
        loaders,
        "load_backends",
        lambda *, need_diarizer, asr_backend=None, asr_ep=None, announce=None, **_: (
            CliASR(),
            None,
            diar,
        ),
    )
    reid = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--speakers", "2", "--out", str(tmp_path)]
    )
    assert reid.exit_code == 0, reid.output
    assert "re-ID: 1 profile(s) active" in reid.output
    assert "Daniel" in (tmp_path / "transcript.md").read_text()

    # The --no-reid re-run replaces the transcript in place — the --force flow.
    no_reid = CliRunner().invoke(
        cli.main,
        [
            "transcribe",
            str(audio),
            "--speakers",
            "2",
            "--no-reid",
            "--out",
            str(tmp_path),
            "--force",
        ],
    )
    assert no_reid.exit_code == 0, no_reid.output
    assert "re-ID:" not in no_reid.output
    md = (tmp_path / "transcript.md").read_text()
    assert "Daniel" not in md and "Speaker 1" in md


def _assignable_meeting(tmp_path):
    import numpy as np

    from stenograf import assets
    from stenograf.config import MeetingProfile
    from stenograf.output import write_transcript
    from stenograf.transcript import Transcript, TranscriptEntry
    from stenograf.voiceprints import write_meeting_voiceprints

    mdir = tmp_path / "meeting-20260801-120000"
    transcript = Transcript(
        language=None,
        profile=MeetingProfile(),
        entries=[TranscriptEntry(speaker="Remote-1", text="hallo", start=0.0, end=1.0)],
    )
    write_transcript(transcript, mdir, "transcript")
    write_meeting_voiceprints(
        mdir,
        {"Remote-1": np.array([1.0, 0.0, 0.0], np.float32)},
        assets.SPEAKER_EMBEDDING.name,
        date="2026-08-01",
    )
    return mdir


def test_profiles_assign_enrolls_from_the_meeting(tmp_path):
    mdir = _assignable_meeting(tmp_path)
    result = CliRunner().invoke(cli.main, ["profiles", "assign", "Remote-1", "Anna", str(mdir)])
    assert result.exit_code == 0, result.output
    assert "enrolled 'Anna'" in result.output
    assert "renamed 'Remote-1'" in result.output

    listing = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Anna" in listing.output

    again = CliRunner().invoke(cli.main, ["profiles", "assign", "Anna", "Anna", str(mdir)])
    assert again.exit_code == 0, again.output
    assert "reinforced 'Anna' (2 sample(s))" in again.output


def test_profiles_assign_refuses_a_mismatched_voice_until_forced(tmp_path):
    import numpy as np

    from stenograf import assets
    from stenograf.voiceprints import ProfileStore

    mdir = _assignable_meeting(tmp_path)
    store = ProfileStore.load()
    store.enroll("Anna", np.array([0.0, 1.0, 0.0], np.float32), assets.SPEAKER_EMBEDDING.name)
    store.save()
    args = ["profiles", "assign", "Remote-1", "Anna", str(mdir)]
    refused = CliRunner().invoke(cli.main, args)
    assert refused.exit_code != 0
    assert "--force" in refused.output
    forced = CliRunner().invoke(cli.main, [*args, "--force"])
    assert forced.exit_code == 0, forced.output
    assert "reinforced 'Anna' (2 sample(s))" in forced.output


def test_profiles_assign_needs_a_meeting_or_last(tmp_path):
    result = CliRunner().invoke(cli.main, ["profiles", "assign", "Remote-1", "Anna"])
    assert result.exit_code != 0
    assert "--last" in result.output


def test_profiles_assign_unknown_label_lists_speakers(tmp_path):
    mdir = _assignable_meeting(tmp_path)
    result = CliRunner().invoke(cli.main, ["profiles", "assign", "Remote-9", "Anna", str(mdir)])
    assert result.exit_code != 0
    assert "Remote-1" in result.output


def test_profiles_commands_use_the_configured_store_and_threshold(tmp_path):
    import numpy as np

    from stenograf import assets
    from stenograf.settings import settings_path
    from stenograf.voiceprints import ProfileStore

    custom = tmp_path / "custom.json"
    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(
        f'[speakers]\nprofile_store = "{custom.as_posix()}"\nreid_threshold = 0.9\n',
        encoding="utf-8",
    )
    mdir = _assignable_meeting(tmp_path)
    store = ProfileStore(custom)
    # cosine 0.8 against Remote-1's [1, 0, 0]: kept at 0.62, refused at 0.9
    store.enroll("Anna", np.array([0.8, 0.6, 0.0], np.float32), assets.SPEAKER_EMBEDDING.name)
    store.save()
    listing = CliRunner().invoke(cli.main, ["profiles", "list"])
    assert "Anna" in listing.output and str(custom) in listing.output
    refused = CliRunner().invoke(cli.main, ["profiles", "assign", "Remote-1", "Anna", str(mdir)])
    assert refused.exit_code != 0 and "--force" in refused.output
    assert not (custom.parent / "steno-data" / "profiles.json").exists()


def test_profiles_commands_report_a_broken_settings_file_cleanly():
    from stenograf.settings import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text("[speakers]\nreid_threshhold = 0.7\n", encoding="utf-8")
    for args in (["list"], ["rename", "A", "B"], ["remove", "A", "--yes"]):
        result = CliRunner().invoke(cli.main, ["profiles", *args])
        assert result.exit_code == 1, result.output
        assert "reid_threshhold" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)
