import io
import json
import sys
import urllib.error
import urllib.request

import pytest

from stenograf.notes import (
    NotesBackendSpec,
    NotesBackendUnavailableError,
    NotesGenerationError,
    available_backends,
    create_backend,
    default_backend_name,
    get_spec,
    register_backend,
)
from stenograf.notes.command import CommandBackend
from stenograf.notes.ollama import DEFAULT_MODEL, ModelNotFoundError, OllamaBackend
from stenograf.settings import NotesSettings

MESSAGES = [
    {"role": "system", "content": "You take notes."},
    {"role": "user", "content": "The transcript."},
]
SCHEMA = {"type": "object", "required": ["title"]}


# ---- registry ---------------------------------------------------------------


def test_builtin_backends_registered():
    assert set(available_backends()) >= {"mlx", "ollama", "command"}


def test_get_spec_unknown_name_lists_choices():
    with pytest.raises(ValueError, match="unknown notes backend.*ollama"):
        get_spec("gpt")


def test_default_backend_precedence(monkeypatch):
    import importlib.util

    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)
    # The built-in default is platform-conditional: in-process MLX where its
    # runtime is installed (Apple Silicon), Ollama everywhere else.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert default_backend_name() == "mlx"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert default_backend_name() == "ollama"
    assert default_backend_name("command") == "command"
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "ollama")
    assert default_backend_name("command") == "ollama"  # env beats settings


def test_create_backend_from_settings(monkeypatch):
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)
    backend = create_backend(None, NotesSettings(backend="command", command=("echo",)))
    assert isinstance(backend, CommandBackend)
    backend = create_backend("ollama", NotesSettings())
    assert isinstance(backend, OllamaBackend)


def test_configured_model_does_not_leak_into_another_backend(monkeypatch):
    from stenograf.notes.mlx import DEFAULT_MODEL as MLX_DEFAULT
    from stenograf.notes.mlx import MlxBackend

    # settings.toml written for the command backend; its model is a claude
    # label, meaningless (and harmful) as an HF repo id for mlx.
    settings = NotesSettings(backend="command", command=("claude", "-p"), model="claude-opus-4-8")
    assert create_backend("mlx", settings).model == MLX_DEFAULT
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "mlx")
    assert create_backend(None, settings).model == MLX_DEFAULT
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND")
    # For the backend the table was written for, the model applies.
    assert create_backend(None, settings).model == "claude-opus-4-8"
    assert isinstance(create_backend(None, settings), CommandBackend)
    # An explicit model for an explicit backend is honored (the CLI path).
    assert MlxBackend.from_settings(NotesSettings(backend="mlx", model="x/y")).model == "x/y"


def test_register_backend_makes_it_creatable():
    class FakeBackend:
        name = "fake"
        model = None

        @classmethod
        def from_settings(cls, settings):
            return cls()

    register_backend(NotesBackendSpec(name="fake", module=__name__, cls="_TestFake", label="fake"))
    sys.modules[__name__]._TestFake = FakeBackend
    try:
        assert isinstance(create_backend("fake", NotesSettings()), FakeBackend)
    finally:
        from stenograf.notes.backend import _REGISTRY

        del _REGISTRY["fake"]
        del sys.modules[__name__]._TestFake


# ---- Ollama backend ----------------------------------------------------------


class FakeOllamaServer:
    """Monkeypatched ``urlopen`` speaking the three endpoints the backend uses."""

    def __init__(self, models=("qwen3:8b",), chat_content='{"title": "T"}'):
        self.models = models
        self.chat_content = chat_content
        self.chat_payloads = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        if url.endswith("/api/version"):
            body = {"version": "0.9.0"}
        elif url.endswith("/api/tags"):
            body = {"models": [{"name": m} for m in self.models]}
        elif url.endswith("/api/chat"):
            self.chat_payloads.append(json.loads(request.data.decode("utf-8")))
            body = {"message": {"role": "assistant", "content": self.chat_content}}
        else:
            raise AssertionError(f"unexpected endpoint {url}")
        return io.BytesIO(json.dumps(body).encode("utf-8"))


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("STENOGRAF_NOTES_MODEL", raising=False)


def test_ollama_complete_sends_schema_and_returns_content(monkeypatch):
    server = FakeOllamaServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    backend = OllamaBackend()
    assert backend.is_available()
    assert backend.complete(MESSAGES, SCHEMA) == '{"title": "T"}'
    payload = server.chat_payloads[0]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["format"] == SCHEMA
    assert payload["stream"] is False
    assert payload["messages"] == MESSAGES


def test_ollama_model_not_pulled(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", FakeOllamaServer(models=("llama3:8b",)))
    backend = OllamaBackend(model="qwen3:8b")
    with pytest.raises(ModelNotFoundError, match="ollama pull qwen3:8b"):
        backend.complete(MESSAGES, SCHEMA)


def test_ollama_untagged_model_matches_tagged_install(monkeypatch):
    server = FakeOllamaServer(models=("qwen3:latest",))
    monkeypatch.setattr(urllib.request, "urlopen", server)
    OllamaBackend(model="qwen3").complete(MESSAGES, SCHEMA)  # must not raise


def test_ollama_down_is_unavailable(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    backend = OllamaBackend()
    assert not backend.is_available()
    with pytest.raises(NotesBackendUnavailableError, match="ollama serve"):
        backend.complete(MESSAGES, SCHEMA)


def test_ollama_host_env_and_normalization(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "somehost:11434")
    assert OllamaBackend().url == "http://somehost:11434"
    assert OllamaBackend(url="http://x:1/").url == "http://x:1"


def test_ollama_from_settings():
    backend = OllamaBackend.from_settings(
        NotesSettings(model="llama3:8b", ollama_url="http://gpu-box:11434")
    )
    assert backend.model == "llama3:8b"
    assert backend.url == "http://gpu-box:11434"


def test_input_budget_is_backend_dependent_and_overridable():
    from stenograf.notes import command as command_mod
    from stenograf.notes import ollama as ollama_mod

    # A hosted frontier model takes far more in one pass than a local 8B.
    assert OllamaBackend().max_input_chars == ollama_mod.DEFAULT_MAX_INPUT_CHARS
    big = CommandBackend(("claude", "-p")).max_input_chars
    assert big == command_mod.DEFAULT_MAX_INPUT_CHARS
    assert big > OllamaBackend().max_input_chars

    override = NotesSettings(command=("claude", "-p"), max_input_chars=9000)
    assert CommandBackend.from_settings(override).max_input_chars == 9000
    assert OllamaBackend.from_settings(override).max_input_chars == 9000


# ---- mlx backend ---------------------------------------------------------------


class FakeMlxLm:
    """Stands in for the ``mlx_lm`` module: canned load()/generate()."""

    def __init__(self, response='{"title": "T"}'):
        self.response = response
        self.loaded_repos = []
        self.generate_calls = []
        self.tokenizer = FakeTokenizer()

    def load(self, repo):
        self.loaded_repos.append(repo)
        return ("fake-model", self.tokenizer)

    def generate(self, model, tokenizer, prompt, max_tokens):
        self.generate_calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, **kwargs})
        return [1, 2, 3]  # token ids


@pytest.fixture
def fake_mlx_lm(monkeypatch):
    from stenograf.notes.mlx import MlxBackend

    fake = FakeMlxLm()
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return fake, MlxBackend()


def test_mlx_complete_renders_template_without_thinking(fake_mlx_lm):
    fake, backend = fake_mlx_lm
    assert backend.complete(MESSAGES, SCHEMA) == '{"title": "T"}'
    assert fake.loaded_repos == [backend.model]
    call = fake.tokenizer.template_calls[0]
    assert call["add_generation_prompt"] is True
    assert call["enable_thinking"] is False
    # The schema instruction rides in the last message (no decode-time grammar)
    # and the caller's message list is not mutated.
    assert '"required": ["title"]' in call["messages"][-1]["content"]
    assert MESSAGES[-1]["content"] == "The transcript."
    assert fake.generate_calls[0]["prompt"] == [1, 2, 3]


def test_mlx_strips_a_stray_think_block(fake_mlx_lm):
    fake, backend = fake_mlx_lm
    fake.response = '<think>\nhmm {not json}\n</think>\n{"title": "T"}'
    assert backend.complete(MESSAGES, SCHEMA) == '\n{"title": "T"}'


def test_mlx_model_stays_loaded_across_completions(fake_mlx_lm):
    fake, backend = fake_mlx_lm
    backend.complete(MESSAGES, SCHEMA)
    backend.complete(MESSAGES, SCHEMA)
    assert fake.loaded_repos == [backend.model]  # one load, two generates
    assert len(fake.generate_calls) == 2


def test_mlx_completions_are_bound_to_one_thread(fake_mlx_lm):
    import threading

    fake, backend = fake_mlx_lm
    backend.complete(MESSAGES, SCHEMA)
    caught = []

    def other_thread():
        try:
            backend.complete(MESSAGES, SCHEMA)
        except NotesGenerationError as exc:
            caught.append(str(exc))

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert caught and "thread" in caught[0]


def test_mlx_load_failure_is_unavailable_not_a_crash(monkeypatch):
    from stenograf.notes.mlx import MlxBackend

    fake = FakeMlxLm()
    fake.load = lambda repo: (_ for _ in ()).throw(OSError("no space left on device"))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    with pytest.raises(NotesBackendUnavailableError, match="no space left"):
        MlxBackend().complete(MESSAGES, SCHEMA)


def test_mlx_from_settings_and_env(monkeypatch):
    from stenograf.notes import mlx as mlx_mod
    from stenograf.notes.mlx import MlxBackend

    backend = MlxBackend.from_settings(
        NotesSettings(model="mlx-community/Qwen3-4B-4bit", max_input_chars=9000)
    )
    assert backend.model == "mlx-community/Qwen3-4B-4bit"
    assert backend.max_input_chars == 9000
    assert MlxBackend().model == mlx_mod.DEFAULT_MODEL
    monkeypatch.setenv("STENOGRAF_NOTES_MODEL", "mlx-community/other")
    assert MlxBackend().model == "mlx-community/other"
    # A local 8B takes less in one pass than a hosted frontier model.
    assert MlxBackend().max_input_chars < CommandBackend(("claude",)).max_input_chars


# ---- command backend ---------------------------------------------------------


def python_argv(body: str) -> tuple[str, ...]:
    return (sys.executable, "-c", body)


def test_command_canned_json():
    backend = CommandBackend(python_argv('print(\'{"title": "T"}\')'))
    assert backend.is_available()
    out = backend.complete(MESSAGES, SCHEMA)
    assert json.loads(out) == {"title": "T"}


def test_command_receives_prompt_and_schema_on_stdin(tmp_path):
    # The command echoes its stdin back; the prompt must carry both message
    # contents and the schema instruction (a generic CLI can't be schema-forced).
    backend = CommandBackend(python_argv("import sys; print(sys.stdin.read())"))
    out = backend.complete(MESSAGES, SCHEMA)
    assert "You take notes." in out
    assert "The transcript." in out
    assert '"required": ["title"]' in out


def test_command_nonzero_exit_surfaces_stderr():
    backend = CommandBackend(
        python_argv("import sys; sys.stderr.write('boom: no credits\\n'); sys.exit(3)")
    )
    with pytest.raises(NotesGenerationError, match="boom: no credits"):
        backend.complete(MESSAGES, SCHEMA)


def test_command_empty_output_is_an_error():
    backend = CommandBackend(python_argv("pass"))
    with pytest.raises(NotesGenerationError, match="no output"):
        backend.complete(MESSAGES, SCHEMA)


def test_command_timeout():
    backend = CommandBackend(python_argv("import time; time.sleep(30)"), timeout_s=0.2)
    with pytest.raises(NotesGenerationError, match="timed out"):
        backend.complete(MESSAGES, SCHEMA)


def test_command_missing_binary():
    backend = CommandBackend(("definitely-not-a-real-binary-xyz",))
    assert not backend.is_available()
    with pytest.raises(NotesBackendUnavailableError, match="PATH"):
        backend.complete(MESSAGES, SCHEMA)


def test_command_unconfigured_raises_with_settings_hint():
    with pytest.raises(NotesBackendUnavailableError, match="settings.toml"):
        CommandBackend(())


def test_command_from_settings():
    backend = CommandBackend.from_settings(
        NotesSettings(command=("claude", "-p"), timeout_s=42.0, model="claude-opus-4-8")
    )
    assert backend.argv == ("claude", "-p")
    assert backend.timeout_s == 42.0
    assert backend.model == "claude-opus-4-8"


# ---- real-CLI e2e (opt-in: costs a real model call) ----------------------------


@pytest.mark.skipif(
    "STENOGRAF_NOTES_E2E" not in __import__("os").environ,
    reason="set STENOGRAF_NOTES_E2E=1 to run the real `claude` CLI end-to-end",
)
def test_command_backend_against_real_claude_cli():
    import shutil

    from stenograf.config import Language, MeetingProfile
    from stenograf.notes.generate import generate_notes
    from stenograf.transcript import Transcript, TranscriptEntry

    claude = shutil.which("claude")
    if claude is None:
        pytest.skip("claude CLI not on PATH")
    backend = CommandBackend((claude, "-p", "--output-format", "text"), timeout_s=300.0)
    transcript = Transcript(
        language=Language.ENGLISH,
        profile=MeetingProfile(attendee_names=("Anna", "Ben")),
        entries=[
            TranscriptEntry(
                speaker="Local-1",
                text="Let's ship the exporter on Friday. Ben, can you write the docs?",
                start=0.0,
                end=6.0,
            ),
            TranscriptEntry(
                speaker="Remote-1",
                text="Yes, I'll have the docs done by Thursday.",
                start=6.0,
                end=10.0,
            ),
        ],
    )
    notes = generate_notes(transcript, backend)
    assert notes.title
    assert notes.summary
    assert notes.provenance.backend == "command"
