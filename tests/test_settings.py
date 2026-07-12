import re
from pathlib import Path

import pytest

from stenograf import settings as settings_mod
from stenograf.settings import NotesSettings, Settings, SettingsError, load_settings


def test_missing_file_is_all_defaults(tmp_path):
    loaded = load_settings(tmp_path / "settings.toml")
    assert loaded == Settings()
    assert loaded.notes == NotesSettings()


def test_full_notes_table_parses(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        """
[notes]
backend = "command"
model = "claude-opus-4-8"
command = ["claude", "-p", "Summarize."]
timeout_s = 300
instructions = "~/style.md"
ollama_url = "http://gpu-box:11434"
thinking = false

[notes.export]
dir = "~/Vault/Meetings"
""",
        encoding="utf-8",
    )
    notes = load_settings(path).notes
    assert notes.backend == "command"
    assert notes.model == "claude-opus-4-8"
    assert notes.command == ("claude", "-p", "Summarize.")
    assert notes.timeout_s == 300.0
    assert notes.instructions == Path("~/style.md").expanduser()
    assert notes.ollama_url == "http://gpu-box:11434"
    assert notes.export_dir == Path("~/Vault/Meetings").expanduser()
    assert notes.thinking is False


def test_settings_path_honors_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    assert settings_mod.settings_path() == tmp_path / "settings.toml"


def test_malformed_toml_names_the_file(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[notes\nbackend = ", encoding="utf-8")
    with pytest.raises(SettingsError, match=re.escape(str(path))):
        load_settings(path)


def test_unknown_backend_names_the_file_and_choices(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[notes]\nbackend = "gpt4"\n', encoding="utf-8")
    with pytest.raises(SettingsError) as excinfo:
        load_settings(path)
    assert str(path) in str(excinfo.value)
    assert "ollama" in str(excinfo.value)


def test_command_as_string_is_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[notes]\ncommand = "claude -p"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="array of strings"):
        load_settings(path)


def test_wrong_typed_timeout_is_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[notes]\ntimeout_s = "fast"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="timeout_s"):
        load_settings(path)


def test_thinking_defaults_to_none_and_rejects_junk(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[notes]\n", encoding="utf-8")
    assert load_settings(path).notes.thinking is None  # backend decides
    path.write_text('[notes]\nthinking = "yes"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="thinking"):
        load_settings(path)


def test_max_input_chars_parses_and_rejects_junk(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[notes]\nmax_input_chars = 200000\n", encoding="utf-8")
    assert load_settings(path).notes.max_input_chars == 200_000

    for bad in ('"many"', "true", "0", "-5"):
        path.write_text(f"[notes]\nmax_input_chars = {bad}\n", encoding="utf-8")
        with pytest.raises(SettingsError, match="max_input_chars"):
            load_settings(path)


def test_full_new_tables_parse(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        """
[transcript]
formats = ["md", "srt"]

[vocab]
glossary_file = "~/steno/glossary.txt"
attendees = ["Ada Lovelace", "Grace Hopper"]
glossary_threshold = 0.9

[output]
dir = "~/Documents/Meetings"

[speakers]
diarization = false
reid_threshold = 0.6
profile_store = "~/steno/profiles.json"

[asr]
backend = "parakeet"
provider = "dml"
""",
        encoding="utf-8",
    )
    s = load_settings(path)
    assert s.transcript.formats == ("md", "srt")
    assert s.vocab.glossary_file == Path("~/steno/glossary.txt").expanduser()
    assert s.vocab.attendees == ("Ada Lovelace", "Grace Hopper")
    assert s.vocab.glossary_threshold == 0.9
    assert s.output.dir == Path("~/Documents/Meetings").expanduser()
    assert s.speakers.diarization is False
    assert s.speakers.reid_threshold == 0.6
    assert s.speakers.profile_store == Path("~/steno/profiles.json").expanduser()
    assert s.asr.backend == "parakeet"
    assert s.asr.provider == "dml"


def test_unknown_transcript_format_is_rejected_with_choices(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[transcript]\nformats = ["docx"]\n', encoding="utf-8")
    with pytest.raises(SettingsError) as excinfo:
        load_settings(path)
    assert "docx" in str(excinfo.value)
    assert "vtt" in str(excinfo.value)


def test_out_of_range_thresholds_are_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    for table, key in (("vocab", "glossary_threshold"), ("speakers", "reid_threshold")):
        path.write_text(f"[{table}]\n{key} = 1.5\n", encoding="utf-8")
        with pytest.raises(SettingsError, match="between 0 and 1"):
            load_settings(path)


def test_unknown_key_in_a_table_is_rejected(tmp_path):
    # The typo guard: a misspelled key must fail loudly, not silently do nothing.
    path = tmp_path / "settings.toml"
    path.write_text('[vocab]\nglossry_file = "x"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"unknown setting\(s\) in \[vocab\]: glossry_file"):
        load_settings(path)


def test_unknown_toplevel_table_is_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[vocap]\nglossary_file = "x"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"unknown setting\(s\): vocap"):
        load_settings(path)


def test_unknown_notes_and_export_keys_are_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[notes]\nmodle = "x"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"in \[notes\]: modle"):
        load_settings(path)
    path.write_text('[notes.export]\nfolder = "x"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match=r"in \[notes.export\]: folder"):
        load_settings(path)


def test_unknown_asr_backend_is_rejected_with_choices(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[asr]\nbackend = "whisper"\n', encoding="utf-8")
    with pytest.raises(SettingsError) as excinfo:
        load_settings(path)
    assert "whisper" in str(excinfo.value)
    assert "parakeet" in str(excinfo.value)


def test_unknown_asr_provider_is_rejected_with_choices(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[asr]\nprovider = "metal"\n', encoding="utf-8")
    with pytest.raises(SettingsError) as excinfo:
        load_settings(path)
    assert "metal" in str(excinfo.value)
    assert "dml" in str(excinfo.value)


def test_wrong_typed_bool_is_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[notes]\nthinking = 1\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="must be true or false"):
        load_settings(path)


def test_diarization_defaults_to_none_and_rejects_junk(tmp_path):
    # Tri-state: unset must stay None (= on), not become False.
    path = tmp_path / "settings.toml"
    path.write_text("[speakers]\n", encoding="utf-8")
    assert load_settings(path).speakers.diarization is None
    path.write_text('[speakers]\ndiarization = "off"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="speakers.diarization must be true or false"):
        load_settings(path)


def test_stale_archive_table_names_the_rename(tmp_path):
    # Pre-Stage-C files configured [archive]; the error must say what replaced
    # it, not just "unknown setting".
    path = tmp_path / "settings.toml"
    path.write_text("[archive]\nenabled = false\n", encoding="utf-8")
    with pytest.raises(SettingsError, match=r"\[archive\] was renamed to \[output\]"):
        load_settings(path)


def test_settings_template_loads_as_all_defaults(tmp_path):
    # Every template key is commented out, so the pristine file `steno settings
    # edit` creates must parse (live table headers included) to exactly Settings().
    from stenograf.settings import SETTINGS_TEMPLATE

    path = tmp_path / "settings.toml"
    path.write_text(SETTINGS_TEMPLATE, encoding="utf-8")
    assert load_settings(path) == Settings()
