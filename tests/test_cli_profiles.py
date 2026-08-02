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
