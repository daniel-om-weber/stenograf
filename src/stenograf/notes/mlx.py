"""Fully local, in-process notes backend: mlx-lm on Apple Silicon.

The zero-install default on macOS (PLAN.md §5 Stage D): mlx-lm ships with
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

import importlib.util
import os
import re
import threading
from typing import TYPE_CHECKING, Any

from stenograf.notes.backend import NotesBackendUnavailableError, NotesGenerationError
from stenograf.notes.prompt import schema_instruction

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
"""Hard stop for one completion. Notes are 1-2k tokens of JSON; a model that
runs past this is looping, and an unbounded generate would spin forever."""

_MAX_OUTPUT_TOKENS_THINKING = 12_288
"""With reasoning on, the think block spends output tokens before the JSON
starts — give it room, still bounded against loops."""

_THINKING_TEMP, _THINKING_TOP_P = 0.6, 0.95
"""Qwen3's model card is explicit: greedy decoding in thinking mode causes
endless repetition. Non-thinking mode keeps mlx-lm's greedy default."""

_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


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
        try:
            return importlib.util.find_spec("mlx_lm") is not None
        except (ImportError, ValueError):
            return False

    def weights_cached(self) -> bool:
        """Whether the model is already in the local HF cache (doctor's hint
        that the first ``--notes`` run will download several GB)."""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.model, local_files_only=True)
        except Exception:
            return False
        return True

    def complete(self, messages: list[dict[str, str]], schema: dict) -> str:
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
        from mlx_lm import generate

        model, tokenizer = self._load()
        prompt = self._render(tokenizer, messages, schema)
        if self.thinking:
            from mlx_lm.sample_utils import make_sampler

            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=_MAX_OUTPUT_TOKENS_THINKING,
                sampler=make_sampler(temp=_THINKING_TEMP, top_p=_THINKING_TOP_P),
            )
        else:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=_MAX_OUTPUT_TOKENS)
        # The think block precedes the JSON (and with thinking off a stray one
        # still can); its prose may contain braces, so it must not reach the
        # JSON extraction.
        return _THINK_BLOCK.sub("", text)

    def _load(self) -> tuple[Any, Any]:
        loaded = self._loaded
        if loaded is None:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise NotesBackendUnavailableError(
                    "mlx-lm is not installed here — reinstall stenograf, or configure "
                    "another backend under [notes] in settings.toml"
                ) from exc
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

    def _render(self, tokenizer, messages: list[dict[str, str]], schema: dict) -> list[int]:
        """Token ids for the chat, schema instruction in the last message.

        mlx-lm has no decode-time grammar (Ollama's ``format=``), so like the
        command backend the schema rides along as an instruction and the
        tolerant JSON extraction in :mod:`.generate` does the rest.
        ``enable_thinking`` toggles Qwen3's reasoning mode; templates without
        that variable simply ignore it."""
        messages = [*messages[:-1], dict(messages[-1])]
        messages[-1]["content"] += "\n\n" + schema_instruction(schema)
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=self.thinking
            )
        except (TypeError, ValueError) as exc:
            raise NotesGenerationError(
                f"model {self.model!r} has no usable chat template: {exc}"
            ) from exc
