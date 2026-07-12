"""Notes backend protocol + registry.

The same selection seam :mod:`stenograf.asr.registry` gives ASR: the LLM that
writes meeting notes is a registered *backend*, not a hard dependency. Two ship
today — ``ollama`` (fully local HTTP) and ``command`` (any CLI driven over
stdin/stdout, e.g. ``claude -p``) — and a new provider is a registration, not a
rewrite. Imports stay lazy: choosing one backend never imports another's module.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stenograf.settings import NotesSettings


class NotesBackendError(Exception):
    """Base for everything a notes backend can raise."""


class NotesBackendUnavailableError(NotesBackendError):
    """The backend cannot run here (service down, command not on PATH)."""


class NotesGenerationError(NotesBackendError):
    """The backend ran but produced no usable notes (bad JSON, non-zero exit)."""


@runtime_checkable
class NotesBackend(Protocol):
    """One LLM provider. ``complete`` returns the model's raw text response;
    schema-shaped JSON extraction/validation happens in :mod:`.generate`, shared
    by all backends. ``model`` is a display/provenance hint (may be ``None``)."""

    name: str
    model: str | None
    max_input_chars: int
    """Rendered-transcript budget for ONE completion; longer meetings are
    map-reduced. A backend property because it tracks the model behind it:
    a local 8B model degrades on long input far sooner than a hosted frontier
    model hard-limits. ``[notes] max_input_chars`` in settings.toml overrides."""

    def is_available(self) -> bool: ...

    def complete(self, messages: list[dict[str, str]], schema: dict) -> str: ...

    @classmethod
    def settings_defaults(cls) -> dict[str, object]:
        """Built-in defaults for the ``[notes]`` keys this backend defines,
        for display by ``steno settings show`` (see :func:`settings_defaults`).
        Keys the backend has no say over (e.g. ``thinking`` off mlx) are
        absent — the CLI renders a which-backend placeholder for those."""
        ...


@dataclass(frozen=True)
class NotesBackendSpec:
    """How to locate one backend without importing it (mirrors ``BackendSpec``)."""

    name: str
    module: str
    cls: str
    label: str


_ENV_OVERRIDE = "STENOGRAF_NOTES_BACKEND"

_REGISTRY: dict[str, NotesBackendSpec] = {}


def register_backend(spec: NotesBackendSpec) -> None:
    """Add (or replace) a backend in the registry."""
    _REGISTRY[spec.name] = spec


register_backend(
    NotesBackendSpec(
        name="mlx",
        module="stenograf.notes.mlx",
        cls="MlxBackend",
        label="MLX (local, in-process)",
    )
)
register_backend(
    NotesBackendSpec(
        name="ollama",
        module="stenograf.notes.ollama",
        cls="OllamaBackend",
        label="Ollama (local)",
    )
)
register_backend(
    NotesBackendSpec(
        name="command",
        module="stenograf.notes.command",
        cls="CommandBackend",
        label="external command",
    )
)


def available_backends() -> list[str]:
    """Names of every registered backend."""
    return list(_REGISTRY)


def default_backend_name(configured: str | None = None) -> str:
    """The backend used when none is named on the CLI: the
    ``STENOGRAF_NOTES_BACKEND`` override, else the settings.toml choice, else
    the built-in local default — in-process MLX where its runtime is installed
    (Apple Silicon), Ollama everywhere else."""
    return os.environ.get(_ENV_OVERRIDE) or configured or _builtin_default()


def _builtin_default() -> str:
    try:
        mlx_installed = importlib.util.find_spec("mlx_lm") is not None
    except (ImportError, ValueError):
        mlx_installed = False
    return "mlx" if mlx_installed else "ollama"


def get_spec(name: str) -> NotesBackendSpec:
    """The :class:`NotesBackendSpec` for ``name``, raising on unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown notes backend {name!r}; choose from {', '.join(_REGISTRY) or 'none'}"
        ) from None


def create_backend(name: str | None, settings: NotesSettings) -> NotesBackend:
    """Instantiate a notes backend, importing only that backend's module.

    Every backend class exposes ``from_settings(settings)`` so machine-specific
    configuration (the command argv, the Ollama URL, the model) flows from one
    place — ``settings.toml``'s ``[notes]`` table — regardless of provider.

    ``[notes] model`` names a model *for the configured backend* (an HF repo
    id, an Ollama tag, a provenance label — different vocabularies). When a
    CLI flag or the env var selects a different backend, that model must not
    ride along: mlx would try to fetch an Ollama tag from HuggingFace.
    """
    spec = get_spec(default_backend_name(settings.backend) if name is None else name)
    if settings.backend is not None and spec.name != settings.backend:
        settings = dataclasses.replace(settings, model=None)
    module = importlib.import_module(spec.module)
    backend_cls = getattr(module, spec.cls)
    return backend_cls.from_settings(settings)


def settings_defaults(name: str) -> dict[str, object]:
    """Backend ``name``'s built-in ``[notes]`` defaults, importing only its module.

    The display seam behind ``steno settings show``: per-backend defaults come
    from the backend class itself instead of an ``if backend == …`` chain in
    the CLI."""
    spec = get_spec(name)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.cls).settings_defaults()
