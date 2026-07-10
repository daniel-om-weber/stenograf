"""ASR backend registry + factory.

One selection seam so a second backend — the Linux ONNX/CTranslate2 path the plan
calls for, or a Whisper/Voxtral backend — is a drop-in *registration* rather than a
rewrite of the CLI's backend loading (PLAN.md §5, Phase 3→4 readiness audit). Imports
stay lazy: choosing one backend never imports another backend's (possibly
platform-specific, e.g. MLX-only) dependencies. Today only ``parakeet`` ships; a new
backend registers a :class:`BackendSpec` and becomes selectable everywhere at once.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from stenograf.asr.base import ASRBackend

_ENV_OVERRIDE = "STENOGRAF_ASR_BACKEND"
_DEFAULT = "parakeet"


@dataclass(frozen=True)
class BackendSpec:
    """How to locate and describe one ASR backend without importing it.

    ``requires`` lists the modules that must be importable for the backend to run
    (surfaced by ``steno doctor``); ``label`` is its human name.
    """

    name: str
    module: str
    cls: str
    requires: tuple[str, ...]
    label: str


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    """Add (or replace) a backend in the registry."""
    _REGISTRY[spec.name] = spec


register_backend(
    BackendSpec(
        name="parakeet",
        module="stenograf.asr.parakeet",
        cls="ParakeetMLXBackend",
        requires=("parakeet_mlx", "mlx"),
        label="parakeet-mlx",
    )
)


def available_backends() -> list[str]:
    """Names of every registered backend."""
    return list(_REGISTRY)


def default_backend_name(configured: str | None = None) -> str:
    """The backend used when none is named: the ``STENOGRAF_ASR_BACKEND`` override,
    else ``configured`` (the ``[asr] backend`` setting), else the built-in default.
    Only ``parakeet`` ships today; a Linux backend that registers here can become
    the platform default."""
    return os.environ.get(_ENV_OVERRIDE) or configured or _DEFAULT


def get_spec(name: str | None = None) -> BackendSpec:
    """The :class:`BackendSpec` for ``name`` (or the default), raising on unknown."""
    name = name or default_backend_name()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown ASR backend {name!r}; choose from {', '.join(_REGISTRY) or 'none'}"
        ) from None


def create_backend(name: str | None = None, **kwargs) -> ASRBackend:
    """Instantiate an ASR backend by name, importing only that backend's module."""
    spec = get_spec(name)
    module = importlib.import_module(spec.module)
    backend_cls = getattr(module, spec.cls)
    return backend_cls(**kwargs)


def backend_model_id(spec: BackendSpec | None = None) -> str | None:
    """The backend module's ``MODEL_ID`` display hint, or ``None``.

    Importing a backend *module* is dependency-free (heavy runtimes like MLX are
    imported inside the methods, not at module top), so ``steno doctor`` can show
    the model id without pulling the backend's runtime.
    """
    spec = spec or get_spec()
    try:
        module = importlib.import_module(spec.module)
    except Exception:
        return None
    return getattr(module, "MODEL_ID", None)
