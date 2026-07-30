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
    {"role": "user", "content": "The transcript.\nFill in this markdown template."},
]
NOTE_MD = "# T\n\nSummary.\n\n## Decisions\n\n- d\n"


# ---- registry ---------------------------------------------------------------


def test_builtin_backends_registered():
    assert set(available_backends()) >= {"mlx", "ollama", "command"}


def test_get_spec_unknown_name_lists_choices():
    with pytest.raises(ValueError, match="unknown notes backend.*ollama"):
        get_spec("gpt")


def test_every_registered_backend_matches_the_protocol():
    # The registry reaches classes via getattr → Any, pyright is basic and
    # scoped to src, and runtime_checkable checks member *presence*, never
    # signatures — so a backend left on an old complete() signature passes
    # ruff AND pyright and dies at runtime. This is the check the type
    # checker cannot make. Classes, not instances: create_backend("command")
    # raises on an empty argv before the signature could be inspected.
    import importlib
    import inspect

    for name in available_backends():
        spec = get_spec(name)
        cls = getattr(importlib.import_module(spec.module), spec.cls)
        params = list(inspect.signature(cls.complete).parameters)
        assert params == ["self", "messages"], f"{name}.complete has drifted: {params}"


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

    def __init__(self, models=("qwen3:8b",), chat_content=NOTE_MD, done_reason="stop"):
        self.models = models
        self.chat_content = chat_content
        self.done_reason = done_reason
        self.chat_payloads = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        if url.endswith("/api/version"):
            body = {"version": "0.9.0"}
        elif url.endswith("/api/tags"):
            body = {"models": [{"name": m} for m in self.models]}
        elif url.endswith("/api/chat"):
            self.chat_payloads.append(json.loads(request.data.decode("utf-8")))
            body = {
                "message": {"role": "assistant", "content": self.chat_content},
                "done_reason": self.done_reason,
            }
        else:
            raise AssertionError(f"unexpected endpoint {url}")
        return io.BytesIO(json.dumps(body).encode("utf-8"))


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("STENOGRAF_NOTES_MODEL", raising=False)


def test_ollama_complete_sends_plain_messages_and_returns_content(monkeypatch):
    server = FakeOllamaServer()
    monkeypatch.setattr(urllib.request, "urlopen", server)
    backend = OllamaBackend()
    assert backend.is_available()
    assert backend.complete(MESSAGES) == NOTE_MD
    payload = server.chat_payloads[0]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["stream"] is False
    assert payload["messages"] == MESSAGES
    # No decode-time grammar: the template rides inside the messages (last, per
    # stenograf.notes.prompt), so the payload must not carry a format= key.
    assert "format" not in payload


def test_ollama_truncated_response_is_an_error_not_a_note(monkeypatch):
    # done_reason == "length" is the server saying the response was cut at a
    # token limit. The old schema path failed on the missing closing brace;
    # markdown has no such tell, so the server's signal is the check.
    monkeypatch.setattr(urllib.request, "urlopen", FakeOllamaServer(done_reason="length"))
    with pytest.raises(NotesGenerationError, match="truncated"):
        OllamaBackend().complete(MESSAGES)


def test_ollama_model_not_pulled(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", FakeOllamaServer(models=("llama3:8b",)))
    backend = OllamaBackend(model="qwen3:8b")
    with pytest.raises(ModelNotFoundError, match="ollama pull qwen3:8b"):
        backend.complete(MESSAGES)


def test_ollama_untagged_model_matches_tagged_install(monkeypatch):
    server = FakeOllamaServer(models=("qwen3:latest",))
    monkeypatch.setattr(urllib.request, "urlopen", server)
    OllamaBackend(model="qwen3").complete(MESSAGES)  # must not raise


def test_ollama_down_is_unavailable(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    backend = OllamaBackend()
    assert not backend.is_available()
    with pytest.raises(NotesBackendUnavailableError, match="ollama serve"):
        backend.complete(MESSAGES)


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


class FakeStreamResponse:
    def __init__(self, text, finish_reason):
        self.text = text
        self.finish_reason = finish_reason


class FakeMlxLm:
    """Stands in for the ``mlx_lm`` module: canned load()/stream_generate()."""

    def __init__(self, response=NOTE_MD, finish_reason="stop"):
        self.response = response
        self.finish_reason = finish_reason
        self.loaded_repos = []
        self.generate_calls = []
        self.tokenizer = FakeTokenizer()

    def load(self, repo):
        self.loaded_repos.append(repo)
        return ("fake-model", self.tokenizer)

    def stream_generate(self, model, tokenizer, prompt, max_tokens, sampler=None):
        self.generate_calls.append({"prompt": prompt, "max_tokens": max_tokens, "sampler": sampler})
        # Two chunks, finish_reason only on the last — the real stream's shape.
        yield FakeStreamResponse(self.response[:3], None)
        yield FakeStreamResponse(self.response[3:], self.finish_reason)


class FakeSampleUtils:
    """Stands in for ``mlx_lm.sample_utils``."""

    @staticmethod
    def make_sampler(**kwargs):
        return ("sampler", kwargs)


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
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", FakeSampleUtils())
    return fake, MlxBackend()


def test_mlx_complete_thinks_by_default_with_qwen_sampling(fake_mlx_lm):
    fake, backend = fake_mlx_lm
    assert backend.complete(MESSAGES) == NOTE_MD
    assert fake.loaded_repos == [backend.model]
    call = fake.tokenizer.template_calls[0]
    assert call["add_generation_prompt"] is True
    assert call["enable_thinking"] is True
    # The messages reach the template untouched — the format instruction is
    # already their tail (stenograf.notes.prompt owns it, for every backend).
    assert call["messages"] == MESSAGES
    generated = fake.generate_calls[0]
    assert generated["prompt"] == [1, 2, 3]
    # Qwen3's card: greedy decoding in thinking mode loops — a sampler is
    # mandatory, and the think block needs output-budget headroom.
    assert generated["sampler"] == ("sampler", {"temp": 0.6, "top_p": 0.95})
    assert generated["max_tokens"] > 4096


def test_mlx_thinking_off_is_greedy_and_lean(fake_mlx_lm, monkeypatch):
    from stenograf.notes.mlx import MlxBackend

    fake, _ = fake_mlx_lm
    backend = MlxBackend(thinking=False)
    backend.complete(MESSAGES)
    assert fake.tokenizer.template_calls[0]["enable_thinking"] is False
    assert fake.generate_calls[0]["sampler"] is None
    assert fake.generate_calls[0]["max_tokens"] == 4096
    assert MlxBackend.from_settings(NotesSettings(thinking=False)).thinking is False
    assert MlxBackend.from_settings(NotesSettings()).thinking is True


def test_mlx_hitting_the_output_cap_is_an_error_not_a_note(fake_mlx_lm):
    # finish_reason == "length" is the runtime saying the output was cut at
    # the token cap. The old schema path failed on the missing closing brace;
    # markdown has no such tell, so the runtime's signal is the check.
    fake, backend = fake_mlx_lm
    fake.finish_reason = "length"
    with pytest.raises(NotesGenerationError, match="truncated"):
        backend.complete(MESSAGES)


def test_mlx_model_stays_loaded_across_completions(fake_mlx_lm):
    fake, backend = fake_mlx_lm
    backend.complete(MESSAGES)
    backend.complete(MESSAGES)
    assert fake.loaded_repos == [backend.model]  # one load, two generates
    assert len(fake.generate_calls) == 2


def test_mlx_completions_are_bound_to_one_thread(fake_mlx_lm):
    import threading

    fake, backend = fake_mlx_lm
    backend.complete(MESSAGES)
    caught = []

    def other_thread():
        try:
            backend.complete(MESSAGES)
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
        MlxBackend().complete(MESSAGES)


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


def test_command_canned_markdown():
    backend = CommandBackend(python_argv("print('# T')"))
    assert backend.is_available()
    assert backend.complete(MESSAGES).strip() == "# T"


def test_command_receives_the_prompt_on_stdin_in_message_order():
    # The command echoes its stdin back; the flattened prompt must carry every
    # message's content in order — which keeps the format instruction (the
    # tail of the last message, per stenograf.notes.prompt) last.
    backend = CommandBackend(python_argv("import sys; print(sys.stdin.read())"))
    out = backend.complete(MESSAGES)
    assert "You take notes." in out
    assert out.index("You take notes.") < out.index("The transcript.")
    assert out.rstrip().endswith("Fill in this markdown template.")


def test_command_nonzero_exit_surfaces_stderr():
    backend = CommandBackend(
        python_argv("import sys; sys.stderr.write('boom: no credits\\n'); sys.exit(3)")
    )
    with pytest.raises(NotesGenerationError, match="boom: no credits"):
        backend.complete(MESSAGES)


def test_command_empty_output_is_an_error():
    backend = CommandBackend(python_argv("pass"))
    with pytest.raises(NotesGenerationError, match="no output"):
        backend.complete(MESSAGES)


def test_command_timeout():
    backend = CommandBackend(python_argv("import time; time.sleep(30)"), timeout_s=0.2)
    with pytest.raises(NotesGenerationError, match="timed out"):
        backend.complete(MESSAGES)


def test_command_missing_binary():
    backend = CommandBackend(("definitely-not-a-real-binary-xyz",))
    assert not backend.is_available()
    with pytest.raises(NotesBackendUnavailableError, match="PATH"):
        backend.complete(MESSAGES)


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
    assert notes.body
    assert "## Action items" in notes.body
    assert notes.provenance.backend == "command"
