"""Notes backend that drives any external CLI over stdin/stdout.

The provider-agnostic escape hatch: configure an argv in
``settings.toml`` — ``["claude", "-p", …]``, ``["llm", "-m", "gpt-…"]``, a
shell wrapper — and stenograf feeds it the rendered prompt (the template
instruction is already its tail, per :mod:`.prompt`) on stdin and expects the
completed markdown on stdout. Hosted models with a login-managed CLI thus
need no API key handling here.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from stenograf.notes.backend import NotesBackendUnavailableError, NotesGenerationError

if TYPE_CHECKING:
    from stenograf.settings import NotesSettings

DEFAULT_TIMEOUT_S = 600.0
"""Generous by default: a long meeting through a large hosted model takes
minutes, and a hang is cut off rather than waited on forever."""

DEFAULT_MAX_INPUT_CHARS = 400_000
"""~100k tokens — hosted frontier models take a multi-hour meeting in one pass
(fewer calls, better coherence than map-reduce). Driving a small model through
the command backend? Lower ``[notes] max_input_chars`` in settings.toml."""


class CommandBackend:
    name = "command"

    def __init__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        model: str | None = None,
        max_input_chars: int | None = None,
    ) -> None:
        if not argv:
            raise NotesBackendUnavailableError(
                "the command notes backend has no command configured — set "
                '`command = ["claude", "-p", …]` under [notes] in settings.toml'
            )
        self.argv = tuple(argv)
        self.timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        # Provenance label only — the model is whatever the command runs.
        self.model = model or self.argv[0]
        self.max_input_chars = max_input_chars or DEFAULT_MAX_INPUT_CHARS

    @classmethod
    def from_settings(cls, settings: NotesSettings) -> CommandBackend:
        return cls(
            settings.command,
            timeout_s=settings.timeout_s,
            model=settings.model,
            max_input_chars=settings.max_input_chars,
        )

    @classmethod
    def settings_defaults(cls) -> dict[str, object]:
        # No "model": it is a provenance label with no default here — the CLI
        # renders its explanatory placeholder instead.
        return {
            "timeout_s": DEFAULT_TIMEOUT_S,
            "max_input_chars": DEFAULT_MAX_INPUT_CHARS,
        }

    def is_available(self) -> bool:
        return shutil.which(self.argv[0]) is not None

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Run the command once; its stdout is the model's response.

        No completion signal exists here (a CLI's exit code says nothing about
        a token cap), so unlike mlx/Ollama this backend cannot detect
        truncation itself — the heading validation in :mod:`.generate` is its
        only net. The markdown unwrap there handles a chatty model."""
        prompt = _render(messages)
        try:
            proc = subprocess.run(
                self.argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise NotesBackendUnavailableError(
                f"notes command not found: {self.argv[0]!r} — is it on PATH?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise NotesGenerationError(
                f"notes command timed out after {self.timeout_s:g}s: {' '.join(self.argv)}"
            ) from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip().splitlines()
            why = stderr[-1] if stderr else f"exit code {proc.returncode}"
            raise NotesGenerationError(f"notes command failed: {why}")
        if not proc.stdout.strip():
            raise NotesGenerationError("notes command produced no output")
        return proc.stdout


def _render(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into one prompt text.

    The format instruction (the template) is already the last message's tail —
    :mod:`.prompt` owns that seat for every backend — so flattening preserves
    its deliberate end-of-prompt position."""
    return "\n\n".join(m["content"] for m in messages)
