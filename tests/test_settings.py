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
    with pytest.raises(SettingsError, match=str(path)):
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
