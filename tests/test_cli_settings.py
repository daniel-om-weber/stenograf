"""settings.toml wiring through the CLI, plus `steno settings show/edit`."""

import json
from pathlib import Path

from click.testing import CliRunner
from conftest import (
    fake_load_backends,
    write_settings,
    write_wav,
)

from stenograf import cli, loaders

# ---------------------------------------------------------------------------
# settings.toml wiring — one test per *mechanism* (the resolution helpers are
# shared, so file-beats-default / flag-beats-file / tri-state / merge each need
# proving once, not per field).


def test_settings_formats_are_the_default_but_format_flag_wins(tmp_path, stub_backends):
    write_settings('[transcript]\nformats = ["srt"]\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    out1 = tmp_path / "one"
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out1)])
    assert result.exit_code == 0, result.output
    assert (out1 / "transcript.srt").exists()
    assert not (out1 / "transcript.md").exists()

    out2 = tmp_path / "two"
    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(out2), "--format", "md"]
    )
    assert result.exit_code == 0, result.output
    assert (out2 / "transcript.md").exists()
    assert not (out2 / "transcript.srt").exists()


def test_settings_output_dir_replaces_the_home_and_out_flag_wins(tmp_path, stub_backends):
    home = tmp_path / "configured-home"
    # as_posix(): a raw Windows path in a TOML basic string is invalid (\U…).
    write_settings(f'[output]\ndir = "{home.as_posix()}"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    # File-beats-default: no flags → the meeting folder is created in [output] dir.
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output
    (meeting_dir,) = home.iterdir()
    assert (meeting_dir / "transcript.md").exists()

    # Flag-beats-file: --out bypasses the configured home for this run.
    out = tmp_path / "explicit"
    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "transcript.md").exists()
    assert len(list(home.iterdir())) == 1  # nothing new in the home


def test_settings_vocab_merges_with_flags(tmp_path, stub_backends):
    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text("Idee\n", encoding="utf-8")
    write_settings(f'[vocab]\nglossary_file = "{glossary_file.as_posix()}"\nattendees = ["Ada"]\n'
    )
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--glossary", "Wirklich"]
    )

    assert result.exit_code == 0, result.output
    # Configured file + inline flag merge (2 terms), attendees ride along (1 name).
    assert "glossary: 2 term(s), 1 name(s)" in result.output
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "gute Idee für" in md  # the settings-file term corrected the transcript


def test_settings_missing_glossary_file_is_a_clean_error(tmp_path, stub_backends):
    write_settings('[vocab]\nglossary_file = "/nonexistent/glossary.txt"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "cannot read glossary file" in result.output
    assert "[vocab] glossary_file" in result.output  # says where the bad path came from


def test_broken_settings_fail_fast_with_a_clean_error(tmp_path, stub_backends):
    write_settings('[vocab]\nglossry_file = "x"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "invalid settings" in result.output
    assert "glossry_file" in result.output
    assert "Traceback" not in result.output


def test_settings_asr_backend_reaches_the_loader(tmp_path, monkeypatch):
    calls = {}

    def recording(*, need_diarizer, asr_backend=None, asr_provider=None, announce=None, **_):
        calls["asr_backend"] = asr_backend
        calls["asr_provider"] = asr_provider
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording)
    write_settings('[asr]\nbackend = "parakeet"\nprovider = "dml"\n')
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls["asr_backend"] == "parakeet"
    assert calls["asr_provider"] == "dml"


def test_glossary_reaches_the_loader_for_decode_time_biasing(tmp_path, monkeypatch):
    # The glossary has two jobs, and this is the one that is invisible when it
    # breaks: the terms must reach the ASR *loader*, which compiles them into the
    # boosting tree that steers decoding. Post-correction would still run and the
    # transcript would still look plausible, so nothing else here would notice.
    calls = {}

    def recording(*, need_diarizer, glossary=(), attendee_names=(), boost=None, **_):
        calls.update(glossary=tuple(glossary), attendee_names=tuple(attendee_names), boost=boost)
        return fake_load_backends(need_diarizer=need_diarizer)

    monkeypatch.setattr(loaders, "load_backends", recording)
    write_settings("[asr]\nboost = 2.0\n")
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main,
        # fmt: off
        [
            "transcribe",
            str(audio),
            "--out",
            str(tmp_path),
            "--glossary",
            "Grafana",
            "--attendee",
            "Ada Lovelace",
        ],
        # fmt: on
    )

    assert result.exit_code == 0, result.output
    assert calls["glossary"] == ("Grafana",)
    assert calls["attendee_names"] == ("Ada Lovelace",)
    assert calls["boost"] == 2.0


def test_settings_profile_store_stays_off_the_transcript(tmp_path, stub_backends):
    # The configured store must feed re-ID loading only: MeetingProfile serializes
    # into every transcript, and keeping machine-local paths out of shared files
    # is the settings file's founding rule.
    write_settings(f'[speakers]\nprofile_store = "{tmp_path.as_posix()}/profiles.json"\n'
    )
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    profile = json.loads((tmp_path / "transcript.json").read_text())["profile"]
    assert profile.get("speaker_profile_store") is None


def test_settings_show_reports_values_and_sources(tmp_path, monkeypatch):
    write_settings('[transcript]\nformats = ["srt"]\n')

    result = CliRunner().invoke(cli.main, ["settings", "show"])

    assert result.exit_code == 0, result.output
    assert 'formats = ["srt"]  (settings.toml)' in result.output
    assert "glossary_threshold = 0.95  (default)" in result.output
    assert "[notes.export]" in result.output

    # An env override wins over the file and is attributed to the variable.
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert "backend  = parakeet  ($STENOGRAF_ASR_BACKEND)" in result.output
    assert "provider = cpu  (default)" in result.output
    assert "boost    = 1  (default)" in result.output


def test_settings_show_reports_record_audio(tmp_path):
    # The one key that decides whether raw audio reaches disk: a run announces
    # it, but until then the report that promises the effective configuration
    # left it out entirely.
    write_settings("[output]\nrecord_audio = true\n")

    result = CliRunner().invoke(cli.main, ["settings", "show"])

    assert result.exit_code == 0, result.output
    assert "record_audio = true  (settings.toml)" in result.output


def test_settings_show_covers_every_settings_key():
    """No key may ship without a row — twice now (``[asr] boost``, ``[output]
    record_audio``) a setting loaded and steered the run while the one screen
    that claims to print the effective configuration stayed silent about it."""
    import dataclasses

    from stenograf.cli.settings_cmd import _settings_rows
    from stenograf.settings import Settings

    settings = Settings()
    shown = {(table, key) for table, rows in _settings_rows(settings) for key, _, _ in rows}
    # `meetings` is user-named preset sections, listed by `steno presets`, not
    # a fixed key set; `notes.export_dir` is flattened onto its own table.
    renamed = {("notes", "export_dir"): ("notes.export", "dir")}

    missing = []
    for table in dataclasses.fields(settings):
        if table.name == "meetings":
            continue
        for key in dataclasses.fields(getattr(settings, table.name)):
            row = renamed.get((table.name, key.name), (table.name, key.name))
            if row not in shown:
                missing.append(f"[{row[0]}] {row[1]}")
    assert not missing, f"settings show has no row for: {', '.join(missing)}"


def test_settings_show_names_a_missing_file(tmp_path):
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert result.exit_code == 0, result.output
    assert "not present — all defaults" in result.output


def test_settings_show_broken_file_points_at_edit(tmp_path):
    write_settings("[vocab]\nbad_key = 1\n")
    result = CliRunner().invoke(cli.main, ["settings", "show"])
    assert result.exit_code != 0
    assert "bad_key" in result.output
    assert "steno settings edit" in result.output


def test_settings_edit_creates_the_template_and_validates(tmp_path, monkeypatch):
    opened = {}
    monkeypatch.setattr(cli.click, "edit", lambda filename=None: opened.update(path=filename))

    result = CliRunner().invoke(cli.main, ["settings", "edit"])

    assert result.exit_code == 0, result.output
    path = tmp_path / "steno-data" / "settings.toml"
    assert opened["path"] == str(path)
    assert "created" in result.output
    assert "OK" in result.output
    assert path.read_text(encoding="utf-8").startswith("# stenograf settings")


def test_settings_edit_keeps_and_reports_a_bad_save(tmp_path, monkeypatch):
    def fake_edit(filename=None):
        Path(filename).write_text('[vocab]\nglossry_file = "x"\n', encoding="utf-8")

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = CliRunner().invoke(cli.main, ["settings", "edit"])

    assert result.exit_code != 0
    assert "glossry_file" in result.output
    assert "your edits are saved" in result.output
    # The bad content was not reverted — the user's work survives the failure.
    path = tmp_path / "steno-data" / "settings.toml"
    assert "glossry_file" in path.read_text(encoding="utf-8")
