"""Where meetings land: the output home, --out, overwrite guards, and
recorded audio — shared behavior of `start` and `transcribe`."""


from click.testing import CliRunner
from conftest import (
    fake_load_backends,
    write_settings,
    write_wav,
)

from stenograf import cli, loaders, output


def test_cleanup_checkpoints_removes_every_checkpoint_format(tmp_path):
    for fmt in ("md", "json", "txt"):
        (tmp_path / f"transcript.partial.{fmt}").write_text("x", encoding="utf-8")
    output.cleanup_checkpoints(tmp_path, "transcript")
    assert not list(tmp_path.glob("transcript.partial.*"))


# ---- output home (Stage C1/C2) ---------------------------------------------


def _start_batch(tmp_path, monkeypatch, *extra):
    """Run a minimal, deterministic ``steno start`` (batch replay) and return the result."""
    monkeypatch.setattr(loaders, "load_backends", fake_load_backends)
    mic = tmp_path / "mic.wav"
    write_wav(mic)
    return CliRunner().invoke(
        cli.main,
        ["start", "--local", "1", "--remote", "0", "--replay", str(mic), "--no-live", *extra],
    )


def test_start_writes_a_dated_folder_into_the_output_home(tmp_path, monkeypatch):
    # No --out: the meeting gets its own meeting-YYYYMMDD-HHMMSS/ folder in the
    # visible output home, holding plainly named transcript files.
    from stenograf.transcript import Transcript

    result = _start_batch(tmp_path, monkeypatch, "--title", "Weekly sync")
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert meeting_dir.name.startswith("meeting-")
    assert (meeting_dir / "transcript.json").exists()
    assert str(meeting_dir) in result.output  # the CLI says where the files landed
    transcript = Transcript.from_json((meeting_dir / "transcript.json").read_text())
    assert transcript.profile.title == "Weekly sync"


def test_start_out_is_the_meetings_own_folder(tmp_path, monkeypatch):
    out = tmp_path / "custom"
    result = _start_batch(tmp_path, monkeypatch, "--out", str(out))
    assert result.exit_code == 0, result.output
    assert (out / "transcript.json").exists()  # files land directly in --out
    assert not (tmp_path / "meetings-home").exists()  # the home is untouched


def test_out_refuses_an_existing_transcript_unless_forced(tmp_path, monkeypatch):
    # Fixed file names mean a reused --out would silently replace the previous
    # meeting — refuse, and let --force say overwriting is the point.
    out = tmp_path / "custom"
    assert _start_batch(tmp_path, monkeypatch, "--out", str(out)).exit_code == 0
    first = (out / "transcript.md").read_text(encoding="utf-8")

    refused = _start_batch(tmp_path, monkeypatch, "--out", str(out), "--title", "Second")
    assert refused.exit_code != 0
    assert "--force" in refused.output
    assert (out / "transcript.md").read_text(encoding="utf-8") == first  # untouched

    forced = _start_batch(tmp_path, monkeypatch, "--out", str(out), "--force")
    assert forced.exit_code == 0, forced.output


def test_out_overwrite_guard_ignores_partial_checkpoints(tmp_path, monkeypatch):
    # A crashed run leaves only .partial files; recovering into the same folder
    # must not demand --force.
    out = tmp_path / "custom"
    out.mkdir()
    (out / "transcript.partial.md").write_text("crashed", encoding="utf-8")
    result = _start_batch(tmp_path, monkeypatch, "--out", str(out))
    assert result.exit_code == 0, result.output


def test_transcribe_out_refusal_happens_before_any_transcription(tmp_path, monkeypatch):
    def explode(*, need_diarizer, asr_backend=None, asr_provider=None, announce=None, **_):
        raise AssertionError("backends must not load when --out is refused")

    monkeypatch.setattr(loaders, "load_backends", explode)
    out = tmp_path / "custom"
    out.mkdir()
    (out / "transcript.md").write_text("previous meeting", encoding="utf-8")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out)])
    assert result.exit_code != 0
    assert "--force" in result.output


def test_transcribe_writes_into_the_output_home_by_default(tmp_path, stub_backends):
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert (meeting_dir / "transcript.md").exists()
    assert str(meeting_dir) in result.output


def test_no_index_is_ever_written(tmp_path, monkeypatch):
    # Stage C2: the filesystem is the index. A run leaves exactly the meeting
    # folder — no index.json in the home or the data dir, ever.
    result = _start_batch(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert not list((tmp_path / "meetings-home").rglob("index.json"))
    assert not list((tmp_path / "steno-data").rglob("index.json"))


def test_record_audio_lands_in_the_meeting_folder(tmp_path, monkeypatch):
    result = _start_batch(tmp_path, monkeypatch, "--record-audio")
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert (meeting_dir / "audio.wav").exists()


def test_output_record_audio_setting_keeps_audio_without_a_flag(tmp_path, monkeypatch):
    write_settings("[output]\nrecord_audio = true\n")
    result = _start_batch(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert (meeting_dir / "audio.wav").exists()


def test_no_record_audio_opts_out_of_the_standing_default(tmp_path, monkeypatch):
    write_settings("[output]\nrecord_audio = true\n")
    result = _start_batch(tmp_path, monkeypatch, "--no-record-audio")
    assert result.exit_code == 0, result.output

    (meeting_dir,) = (tmp_path / "meetings-home").iterdir()
    assert not (meeting_dir / "audio.wav").exists()


def test_record_audio_and_no_record_audio_conflict(tmp_path, monkeypatch):
    result = _start_batch(tmp_path, monkeypatch, "--record-audio", "--no-record-audio")
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
