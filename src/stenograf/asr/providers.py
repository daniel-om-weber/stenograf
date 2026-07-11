"""ONNX Runtime execution-provider selection for the ORT-backed ASR backends.

The names users configure are deliberately short (``dml``, not
``DmlExecutionProvider``): they are the ``[asr] provider`` vocabulary in
settings.toml and the ``STENOGRAF_ASR_PROVIDER`` env value. CPU is the default
everywhere — acceleration is opt-in. Measured 2026-07-11 (parakeet-v3 fp32,
RTX 4080 SUPER): DirectML transcribes byte-identically to CPU at ~6.6× the
finalize speed, but CPU already runs 16–30× realtime, so a meeting never
*needs* a GPU; the win is long-file ``transcribe`` and ``--full-finalize``.

Which providers *exist* is a property of the installed onnxruntime flavor —
exactly one flavor may be installed (they all own the ``onnxruntime/`` package
directory and silently clobber each other), so the wheel depends on
``onnxruntime-directml`` on Windows and plain ``onnxruntime`` elsewhere.
``auto`` picks the best accelerated provider that flavor actually offers,
else CPU. This module stays import-light: settings validation reads
``PROVIDER_CHOICES`` without touching onnxruntime.
"""

from __future__ import annotations

import os

ENV_OVERRIDE = "STENOGRAF_ASR_PROVIDER"

_ORT_PROVIDERS: dict[str, tuple[str, ...]] = {
    "cpu": ("CPUExecutionProvider",),
    "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
}

PROVIDER_CHOICES: tuple[str, ...] = ("auto", *_ORT_PROVIDERS)

PROVIDER_LABELS = {"cpu": "CPU", "dml": "DirectML", "cuda": "CUDA"}
"""Human names for messages (`asr: accelerated (DirectML)`, doctor output)."""

_ACCELERATED = ("dml", "cuda")
"""``auto`` preference order; CoreML is deliberately absent — ORT's CoreML
provider fails on the parakeet model (verified 2026-07-11), and macOS runs
the MLX backend anyway."""


def default_provider_name(configured: str | None = None) -> str:
    """The provider used when none is named: the ``STENOGRAF_ASR_PROVIDER``
    override, else ``configured`` (the ``[asr] provider`` setting), else CPU."""
    return os.environ.get(ENV_OVERRIDE) or configured or "cpu"


def validate_provider_name(name: str) -> str:
    """``name`` if it is a known provider choice, raising :class:`ValueError`
    naming the choices otherwise (mirrors the backend-name validation)."""
    if name not in PROVIDER_CHOICES:
        raise ValueError(
            f"unknown ASR provider {name!r} (choose from {', '.join(PROVIDER_CHOICES)})"
        )
    return name


def resolve(name: str) -> str:
    """Collapse ``auto`` to a concrete provider against the installed ORT flavor."""
    validate_provider_name(name)
    if name != "auto":
        return name
    accelerated = available_accelerators()
    return accelerated[0] if accelerated else "cpu"


def available_accelerators() -> tuple[str, ...]:
    """The accelerated providers the installed onnxruntime build offers, in
    ``auto`` preference order. Availability is a claim about the build, not the
    hardware — a DX12-less box still lists DML — which is why the backend
    canary-decodes before committing to a provider."""
    import onnxruntime

    available = set(onnxruntime.get_available_providers())
    return tuple(name for name in _ACCELERATED if _ORT_PROVIDERS[name][0] in available)


def ort_providers(name: str) -> list[str]:
    """The onnxruntime provider list for a concrete (post-:func:`resolve`) name."""
    return list(_ORT_PROVIDERS[name])


def unavailable_reason(name: str) -> str | None:
    """Why the installed onnxruntime build cannot run provider ``name``, or
    ``None`` if it lists it. Must be checked *before* creating a session: ORT
    does not fail on an unlisted provider — it warns and silently builds the
    session from whatever remains, so the canary would pass on CPU and the
    run would claim acceleration it isn't getting (observed 2026-07-11 with
    cuda requested against the DirectML build)."""
    import onnxruntime

    ep = _ORT_PROVIDERS[name][0]
    available = onnxruntime.get_available_providers()
    if ep in available:
        return None
    return f"{ep} is not in this onnxruntime build (available: {', '.join(available)})"
