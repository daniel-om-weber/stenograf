"""Backend-level guards for the parakeet ASR backend.

The load-thread materialization is a regression guard for a bug that only
reproduces on real hardware (MLX GPU + a worker thread), so it is asserted here
at the mechanism level rather than by running the real model.
"""

import pytest

# The MLX stack only exists on macOS-arm64; keep the suite collecting elsewhere.
mx = pytest.importorskip("mlx.core")
parakeet_mlx = pytest.importorskip("parakeet_mlx")

from stenograf.asr.parakeet import ParakeetMLXBackend, _merge_tokens  # noqa: E402


class _FakeModel:
    def __init__(self) -> None:
        self._params = {"weight": object()}

    def parameters(self):
        return self._params


def test_load_materializes_weights_for_cross_thread_use(monkeypatch):
    # The live LiveWorker decodes on a *different* thread than load() ran on.
    # MLX is lazy and its GPU streams are thread-local, so weights left lazy stay
    # bound to the load thread's Stream(gpu, 0) and the worker's first decode dies
    # with "There is no Stream(gpu, 0) in current thread". load() must force the
    # weights concrete on the load thread; if this eval is dropped, the live pass
    # breaks on real hardware while every mocked test stays green — hence this.
    fake = _FakeModel()
    monkeypatch.setattr(parakeet_mlx, "from_pretrained", lambda model_id: fake)
    evaled: list[tuple] = []
    monkeypatch.setattr(mx, "eval", lambda *args: evaled.append(args))

    backend = ParakeetMLXBackend()
    backend.load()

    assert backend._model is fake
    assert evaled == [(fake.parameters(),)]  # weights materialized on the load thread


def test_load_prefers_the_complete_local_snapshot(monkeypatch, tmp_path):
    # A complete cache must load fully offline: hf_hub_download revision-checks
    # the Hub on every call otherwise (unauthenticated-request warning each run,
    # 10 s/file stall on a hanging network). load() resolves the snapshot with
    # local_files_only=True and hands from_pretrained the directory instead of
    # the repo id.
    import huggingface_hub

    resolved: list[tuple] = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        resolved.append((repo_id, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    loaded: list[str] = []

    def fake_from_pretrained(model_id):
        loaded.append(model_id)
        return _FakeModel()

    monkeypatch.setattr(parakeet_mlx, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(mx, "eval", lambda *args: None)

    ParakeetMLXBackend().load()

    assert loaded == [str(tmp_path)]
    assert [f for _, f, _ in resolved] == ["config.json", "model.safetensors"]
    assert all(kw == {"local_files_only": True} for _, _, kw in resolved)


def test_load_falls_back_online_when_cache_is_incomplete(monkeypatch, tmp_path):
    # First run or an interrupted download: local resolution raises, and load()
    # must pass the repo id through so from_pretrained downloads normally.
    import huggingface_hub

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        if kwargs.get("local_files_only"):
            raise huggingface_hub.errors.LocalEntryNotFoundError("not cached")
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    loaded: list[str] = []

    def fake_from_pretrained(model_id):
        loaded.append(model_id)
        return _FakeModel()

    monkeypatch.setattr(parakeet_mlx, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(mx, "eval", lambda *args: None)

    backend = ParakeetMLXBackend()
    backend.load()

    assert loaded == [backend.model_id]


class _Tok:
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end


def _texts(tokens):
    return [w.text for w in _merge_tokens(tokens)]


def test_merge_tokens_leading_space_starts_a_word():
    toks = [_Tok(" hal", 0.0, 0.2), _Tok("lo", 0.2, 0.3), _Tok(" welt", 0.4, 0.7)]
    assert _texts(toks) == ["hallo", "welt"]


def test_merge_tokens_bare_space_token_is_a_word_boundary():
    # Numbers arrive as a bare " " boundary token followed by digit pieces
    # (real decode of de-2: " und", " ", "1", "5", ".", "7", "."). The empty
    # token has no visible text but must still break the word — this used to
    # render "und15.7.".
    toks = [
        _Tok(" und", 0.0, 0.2),
        _Tok(" ", 0.2, 0.25),
        _Tok("1", 0.25, 0.3),
        _Tok("5", 0.3, 0.35),
        _Tok(".", 0.35, 0.4),
        _Tok("7", 0.4, 0.45),
        _Tok(".", 0.45, 0.5),
    ]
    merged = _merge_tokens(toks)
    assert [w.text for w in merged] == ["und", "15.7."]
    assert merged[1].start == 0.25  # the number's time span, not the space's
    assert merged[1].end == 0.5


def test_merge_tokens_empty_text_token_is_not_a_boundary():
    toks = [_Tok(" ge", 0.0, 0.1), _Tok("", 0.1, 0.1), _Tok("sagt", 0.1, 0.3)]
    assert _texts(toks) == ["gesagt"]
