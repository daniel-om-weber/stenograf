"""Fully local, in-process notes backend: mlx-lm on Apple Silicon.

The zero-install default on macOS: mlx-lm ships with
stenograf under the same platform marker as the parakeet ASR backend, and the
model downloads into the Hugging Face cache on first use — no server, no
daemon, no ``ollama pull``.

Thread constraint (verified empirically, 2026-07-10): on the mlx-lm 0.29 line
the GPU generation stream is created when ``mlx_lm`` is imported and is only
valid on that thread — ``generate()`` anywhere else dies with "There is no
Stream(gpu, 0)", even when load and generate share a worker thread. Fixed
upstream only in mlx-lm >= 0.31, which we cannot ship (see the pyproject
comment: its transformers>=5 floor is broken against current transformers and
collides with the eval group's 4.x pin). So this backend imports mlx_lm
lazily — the first ``complete()`` call's thread becomes the generation
thread — and raises a clear error if a later call comes from another thread.
The CLI always satisfies this (notes run synchronously on the main thread).
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

from stenograf.notes.backend import NotesBackendUnavailableError, NotesGenerationError

if TYPE_CHECKING:
    from stenograf.settings import NotesSettings

DEFAULT_MODEL = "Qwen/Qwen3-8B-MLX-4bit"
"""Official Qwen MLX quant: 4.35 GB on disk, Apache-2.0, 32k context —
comfortable next to desktop apps on a 16 GB machine and the best-verified
quality-per-GB in the 3-9B range as of July 2026."""

DEFAULT_MAX_INPUT_CHARS = 100_000
"""~25k tokens — inside Qwen3's 32k window with room for the chat template
and the JSON response. Longer meetings map-reduce (see :mod:`.generate`)."""

DEFAULT_THINKING = True
"""Reasoning mode on by default: notes are a batch job where minutes don't
matter but a misattributed decision does. ``[notes] thinking = false`` in
settings.toml trades that headroom for speed."""

_MAX_OUTPUT_TOKENS = 4096
"""Hard stop for one completion. Notes are 1-2k tokens of markdown; a model
that runs past this is looping, and an unbounded generate would spin forever."""

_MAX_OUTPUT_TOKENS_THINKING = 12_288
"""With reasoning on, the think block spends output tokens before the note
starts — give it room, still bounded against loops."""

_THINKING_TEMP, _THINKING_TOP_P = 0.6, 0.95
"""Qwen3's model card is explicit: greedy decoding in thinking mode causes
endless repetition. Non-thinking mode keeps mlx-lm's greedy default."""


class MlxBackend:
    """Runs the notes model in-process via mlx-lm, weights from the HF cache.

    The model stays loaded across ``complete()`` calls so a map-reduced
    meeting pays the load once; the CLI process exits right after notes, so
    nothing lingers."""

    name = "mlx"

    def __init__(
        self,
        model: str | None = None,
        max_input_chars: int | None = None,
        thinking: bool | None = None,
    ) -> None:
        self.model = model or os.environ.get("STENOGRAF_NOTES_MODEL") or DEFAULT_MODEL
        self.max_input_chars = max_input_chars or DEFAULT_MAX_INPUT_CHARS
        self.thinking = DEFAULT_THINKING if thinking is None else thinking
        # (model, tokenizer) — Any because mlx_lm's types are only importable
        # lazily, on the generation thread.
        self._loaded: tuple[Any, Any] | None = None
        self._generation_thread: int | None = None

    @classmethod
    def from_settings(cls, settings: NotesSettings) -> MlxBackend:
        return cls(
            model=settings.model,
            max_input_chars=settings.max_input_chars,
            thinking=settings.thinking,
        )

    @classmethod
    def settings_defaults(cls) -> dict[str, object]:
        return {
            "model": DEFAULT_MODEL,
            "max_input_chars": DEFAULT_MAX_INPUT_CHARS,
            "thinking": DEFAULT_THINKING,
        }

    def is_available(self) -> bool:
        from stenograf.doctor import installed

        return installed("mlx_lm")

    def health(self) -> tuple[bool, str]:
        if not self.is_available():
            return (
                False,
                "mlx-lm is not installed here — reinstall stenograf, or configure "
                "another backend under [notes] in settings.toml",
            )
        hint = "cached" if self.weights_cached() else "downloads on first notes run"
        return True, f"MLX in-process, model {self.model} ({hint})"

    def weights_cached(self) -> bool:
        """Whether the model is already in the local HF cache (doctor's hint
        that the first ``--notes`` run will download several GB)."""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.model, local_files_only=True)
        except Exception:
            return False
        return True

    def warm(self) -> None:
        """Load the weights and generate one token NOW, on this thread.

        The meeting-end wait then starts at the prompt instead of at a
        4.35 GB cold load. Binds generation to the calling thread exactly
        like a first ``complete()`` would (module docstring) — the caller
        owns running the real completions on this same thread afterwards
        (:class:`stenograf.notes.warm.NotesWarmer`). The one-token pass is
        what turns the load into a *warm* model: weight upload and kernel
        compilation happen on first generate, not on ``load``."""
        if self._generation_thread is None:
            self._generation_thread = threading.get_ident()
        from mlx_lm import stream_generate

        model, tokenizer = self._load()
        prompt = self._render(tokenizer, [{"role": "user", "content": "ok"}])
        for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1):
            break

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self._generation_thread is None:
            self._generation_thread = threading.get_ident()
        elif threading.get_ident() != self._generation_thread:
            # See the module docstring: fail with a message instead of
            # mlx-lm 0.29's opaque "There is no Stream(gpu, 0)" RuntimeError.
            raise NotesGenerationError(
                "the mlx notes backend is bound to the thread of its first "
                "completion (mlx-lm 0.29 generation streams are per-thread); "
                "run all completions for one backend instance on one thread"
            )
        from mlx_lm import stream_generate

        model, tokenizer = self._load()
        prompt = self._render(tokenizer, messages)
        kwargs: dict[str, Any] = {"max_tokens": _MAX_OUTPUT_TOKENS}
        if self.thinking:
            from mlx_lm.sample_utils import make_sampler

            kwargs = {
                "max_tokens": _MAX_OUTPUT_TOKENS_THINKING,
                "sampler": make_sampler(temp=_THINKING_TEMP, top_p=_THINKING_TOP_P),
            }
        # stream_generate rather than generate for its finish_reason: hitting
        # the output-token cap used to surface only as JSON with no closing
        # brace — markdown has no such tell, so the truncation check must be
        # the runtime's own completion signal, not a text heuristic.
        parts: list[str] = []
        finish_reason = None
        for response in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
            parts.append(response.text)
            finish_reason = response.finish_reason
        if finish_reason == "length":
            raise NotesGenerationError(
                f"the notes model hit its {kwargs['max_tokens']}-token output cap — "
                "the response is truncated, not a note (a model that runs this "
                "long is usually looping)"
            )
        return "".join(parts)

    def _load(self) -> tuple[Any, Any]:
        loaded = self._loaded
        if loaded is None:
            # No try/except: every caller gates on is_available() first
            # (notes.generate, doctor), so the import cannot fail here.
            from mlx_lm import load
            try:
                model, tokenizer = load(self.model)[:2]
            except Exception as exc:
                # One typed error for the whole fetch+load chain (HF download,
                # missing repo, corrupt weights) — the CLI catches it and the
                # transcript stands.
                raise NotesBackendUnavailableError(
                    f"could not load notes model {self.model!r} via mlx-lm: {exc}"
                ) from exc
            loaded = (model, tokenizer)
            self._loaded = loaded
        return loaded

    def _render(self, tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
        """Token ids for the chat. The format instruction (the template) is
        already the tail of the last message — :mod:`.prompt` owns that, once,
        for every backend. ``enable_thinking`` toggles Qwen3's reasoning mode;
        templates without that variable simply ignore it."""
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=self.thinking
            )
        except (TypeError, ValueError) as exc:
            raise NotesGenerationError(
                f"model {self.model!r} has no usable chat template: {exc}"
            ) from exc
