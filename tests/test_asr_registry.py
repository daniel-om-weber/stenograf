"""ASR backend registry/factory (Phase 3→4 readiness: the backend-selection seam).

The factory is the single point a Linux ONNX/CTranslate2 backend registers into,
so the finalize path is a drop-in rather than a rewrite. These assert the seam
without needing MLX installed — ``create_backend`` instantiates the backend class
(cheap; MLX is imported only inside ``load``/``transcribe``).
"""

from __future__ import annotations

import pytest

from stenograf.asr import (
    ASRBackend,
    available_backends,
    backend_model_id,
    create_backend,
    default_backend_name,
    get_spec,
)


def test_parakeet_is_registered_and_default():
    assert "parakeet" in available_backends()
    assert default_backend_name() == "parakeet"
    assert get_spec().name == "parakeet"


def test_create_backend_returns_the_default_instance():
    backend = create_backend()
    assert isinstance(backend, ASRBackend)
    assert backend.name == "parakeet"  # no MLX import needed to construct it


def test_env_override_selects_the_default(monkeypatch):
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    assert default_backend_name() == "parakeet"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown ASR backend"):
        get_spec("does-not-exist")
    with pytest.raises(ValueError, match="unknown ASR backend"):
        create_backend("does-not-exist")


def test_backend_model_id_is_importable_without_the_runtime():
    # The module import is MLX-free (heavy runtime lives in the methods), so doctor
    # can show the model id without pulling MLX.
    assert backend_model_id(get_spec("parakeet")) == "mlx-community/parakeet-tdt-0.6b-v3"


def test_default_backend_name_precedence(monkeypatch):
    # env override > configured ([asr] backend in settings.toml) > built-in default
    monkeypatch.delenv("STENOGRAF_ASR_BACKEND", raising=False)
    assert default_backend_name() == "parakeet"
    assert default_backend_name("configured") == "configured"
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "env-backend")
    assert default_backend_name("configured") == "env-backend"
