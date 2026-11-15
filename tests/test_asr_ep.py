"""ONNX Runtime provider selection (Windows acceleration, opt-in everywhere).

The contract under test: CPU is the default unless a provider is configured;
``auto`` collapses against what the installed onnxruntime flavor offers; and an
explicitly requested accelerator that fails to initialize *or* to decode (the
CoreML lesson: session creation succeeding proves nothing) raises with the
reason — the user asked for acceleration, so a silent CPU run at a fraction of
the speed is worse than a clean error naming the setting to change.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stenograf.asr import parakeet_onnx
from stenograf.asr.base import BackendUnavailableError
from stenograf.asr.ep import (
    EP_CHOICES,
    available_accelerators,
    default_ep,
    ort_providers,
    resolve,
    validate_ep,
)
from stenograf.asr.parakeet_onnx import ParakeetOnnxBackend


def _stub_onnxruntime(monkeypatch, providers: list[str]) -> None:
    module = types.SimpleNamespace(get_available_providers=lambda: providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


def test_default_ep_precedence(monkeypatch):
    # env override > configured ([asr] provider in settings.toml) > cpu
    monkeypatch.delenv("STENOGRAF_ASR_PROVIDER", raising=False)
    assert default_ep() == "cpu"
    assert default_ep("dml") == "dml"
    monkeypatch.setenv("STENOGRAF_ASR_PROVIDER", "cuda")
    assert default_ep("dml") == "cuda"


def test_unknown_provider_raises_with_choices():
    with pytest.raises(ValueError, match="unknown ASR provider") as excinfo:
        validate_ep("metal")
    for choice in EP_CHOICES:
        assert choice in str(excinfo.value)


def test_resolve_auto_prefers_dml_then_cuda_then_cpu(monkeypatch):
    _stub_onnxruntime(
        monkeypatch, ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert resolve("auto") == "dml"
    assert available_accelerators() == ("dml", "cuda")
    _stub_onnxruntime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert resolve("auto") == "cuda"
    _stub_onnxruntime(monkeypatch, ["AzureExecutionProvider", "CPUExecutionProvider"])
    assert resolve("auto") == "cpu"
    assert available_accelerators() == ()


def test_resolve_concrete_names_pass_through_without_onnxruntime(monkeypatch):
    # "cpu"/"dml" must not import onnxruntime — settings validation and the
    # explicit path stay independent of what's installed.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import would raise
    assert resolve("cpu") == "cpu"
    assert resolve("dml") == "dml"
    assert ort_providers("dml") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert ort_providers("cpu") == ["CPUExecutionProvider"]


class _FakeModel:
    """Stands in for onnx_asr's loaded model: recognize() can be rigged to fail."""

    def __init__(self, fail_recognize: bool = False) -> None:
        self.fail_recognize = fail_recognize
        self.recognized = 0

    def with_timestamps(self):
        return self

    def recognize(self, samples: np.ndarray, sample_rate: int):
        if self.fail_recognize:
            raise RuntimeError("no DirectML device\nlong ORT traceback continues")
        self.recognized += 1
        return types.SimpleNamespace(tokens=[], timestamps=[])


def _stub_onnx_asr(monkeypatch, models_by_first_provider: dict) -> list:
    """onnx_asr.load_model returning (or raising) per requested first provider.

    The model folder is stubbed too, so no test downloads; returns the list of
    paths load_model was handed."""
    paths: list = []

    def load_model(model_id, path=None, *, quantization=None, providers=None):
        paths.append(path)
        result = models_by_first_provider[providers[0]]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(parakeet_onnx, "model_dir", lambda quantization=None: Path("/models"))
    return paths


def test_backend_defaults_to_cpu_without_canary(monkeypatch):
    cpu = _FakeModel()
    paths = _stub_onnx_asr(monkeypatch, {"CPUExecutionProvider": cpu})
    backend = ParakeetOnnxBackend()
    backend.load()
    assert cpu.recognized == 0  # no canary cost on the CPU path
    assert paths == [Path("/models")]  # loads from its own folder, not the hub cache


class _FakeHub:
    """huggingface_hub.snapshot_download writing the files it was asked for."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def snapshot_download(self, repo_id, *, revision, local_dir, allow_patterns):
        self.calls.append({"repo": repo_id, "revision": revision, "patterns": allow_patterns})
        if self.fail:
            raise OSError("connection reset")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (Path(local_dir) / name).write_bytes(b"x")
        return str(local_dir)


def test_model_dir_downloads_plain_files_once(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    first = parakeet_onnx.model_dir()
    assert parakeet_onnx.model_dir() == first
    assert len(hub.calls) == 1  # the marker short-circuits the second call
    assert hub.calls[0]["revision"] == parakeet_onnx._REVISION
    assert sorted(hub.calls[0]["patterns"]) == [
        "config.json",
        "decoder_joint-model.onnx",
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "vocab.txt",
    ]
    assert first.parent == tmp_path
    assert not any(p.is_symlink() for p in first.iterdir())


def test_model_dir_quantized_variant_has_its_own_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    parakeet_onnx.model_dir()
    parakeet_onnx.model_dir("int8")
    assert len(hub.calls) == 2
    assert "encoder-model.int8.onnx" in hub.calls[1]["patterns"]


def test_interrupted_download_is_not_marked_complete(monkeypatch, tmp_path):
    # onnx-asr treats an existing folder as complete; only the marker may say
    # so, and only after every file is in — the next call resumes the download.
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub(fail=True))
    with pytest.raises(OSError):
        parakeet_onnx.model_dir()
    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    parakeet_onnx.model_dir()
    assert len(hub.calls) == 1


def test_backend_uses_accelerated_provider_after_canary(monkeypatch):
    dml = _FakeModel()
    _stub_onnxruntime(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    _stub_onnx_asr(monkeypatch, {"DmlExecutionProvider": dml})
    backend = ParakeetOnnxBackend(ep="dml")
    backend.load()
    assert dml.recognized == 1  # the canary decode


def test_backend_raises_when_session_creation_fails(monkeypatch):
    _stub_onnxruntime(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    _stub_onnx_asr(
        monkeypatch,
        {"DmlExecutionProvider": RuntimeError("D3D12 device unavailable")},
    )
    backend = ParakeetOnnxBackend(ep="dml")
    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.load()
    assert "D3D12 device unavailable" in str(excinfo.value)
    assert '[asr] provider = "cpu"' in str(excinfo.value)  # names the way out


def test_backend_raises_when_the_canary_fails(monkeypatch):
    # The CoreML lesson: the session may build and still not run the model.
    _stub_onnxruntime(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    _stub_onnx_asr(monkeypatch, {"DmlExecutionProvider": _FakeModel(fail_recognize=True)})
    backend = ParakeetOnnxBackend(ep="dml")
    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.load()
    # Only the first line of a multi-line ORT error is kept for the message.
    assert "no DirectML device" in str(excinfo.value)
    assert "long ORT traceback" not in str(excinfo.value)


def test_backend_raises_when_the_build_lacks_the_provider(monkeypatch):
    # ORT does not raise on an unlisted provider — it warns and silently runs
    # on what remains, so the backend must pre-check the build or it would
    # claim acceleration while running on CPU (observed with cuda requested
    # against the DirectML flavor). The stub dict is empty: reaching
    # load_model at all would KeyError, proving no session is attempted.
    _stub_onnxruntime(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    _stub_onnx_asr(monkeypatch, {})
    backend = ParakeetOnnxBackend(ep="cuda")
    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.load()
    assert "not in this onnxruntime build" in str(excinfo.value)
    assert "DmlExecutionProvider" in str(excinfo.value)  # names what IS available
