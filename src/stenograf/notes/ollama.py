"""Fully local notes backend: Ollama over plain HTTP (stdlib only).

No ``ollama`` pip dependency — the three endpoints we need (`/api/version`,
`/api/tags`, `/api/chat`) are a handful of ``urllib`` calls, and staying
stdlib keeps the reserved ``stenograf[ollama]`` extra empty.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from stenograf.notes.backend import (
    NotesBackendError,
    NotesBackendUnavailableError,
    NotesGenerationError,
)

if TYPE_CHECKING:
    from stenograf.settings import NotesSettings

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"  # ~5 GB — fits a 48 GB Mac without swapping
DEFAULT_MAX_INPUT_CHARS = 90_000
"""Sized against the default model's context rather than guessed: with the
reserve below, a prompt this long asks for 40 960 tokens after rounding —
qwen3:8b's ceiling — so a whole meeting is read instead of its ending. Measured
2026-08-12, that is ~4 h of speech (a 37-minute meeting builds a 14 245-char
prompt) and ~29 600 tokens of German prose. Raising it past 92 163 makes the
default model refuse rather than quietly summarize the last minutes; a
smaller-context model needs it lowered."""
_CONNECT_TIMEOUT = 5.0
_CHAT_TIMEOUT = 3600.0
"""A whole meeting now really is read, and reading it is the slow part.
Measured 2026-08-12 on a 24-core CPU box: a 30-minute German meeting took 511 s
end to end, and prefill runs 24–48 tokens/s, so a prompt near
:data:`DEFAULT_MAX_INPUT_CHARS` is tens of minutes — longer than the 600 s this
used to give up after."""

_CHARS_PER_TOKEN = 2.5
"""Below the densest text we have measured, so the context asked for errs large
— asking too small is the silent failure this whole mechanism exists to stop.
Measured 2026-08-12 on qwen3:8b: German transcript prose 3.04, a real German
notes prompt 3.19, English 4.04 chars/token."""

_RESPONSE_TOKENS = 4096
"""Context reserved for the answer, and the cap enforced on it.

Measured 2026-08-12 (qwen3:8b, 30-minute German meeting): 2473 tokens, most of
it a thinking block. It is sent as ``num_predict`` rather than merely reserved
because an answer that runs past the context is **not** refused — the server
shifts the window and evicts the start of the meeting mid-generation, reporting
``done_reason == "stop"`` (measured: a 512-token context produced 1260 tokens
that way). Only an explicit limit makes the ``"length"`` signal below reachable.
"""

_CONTEXT_BUCKET = 4096
"""Context sizes are rounded up to this, so consecutive calls can share a loaded
runner: Ollama keys the runner on ``num_ctx`` and reloads the model whenever it
changes (measured 2026-08-12: 1.7 s per change, against 0.1 s warm). A
map-reduce run calls once per chunk and no two chunks are the same size."""


def _context_tokens(messages: list[dict[str, str]]) -> int:
    """How much context to ask the server for, sized to this prompt.

    Ollama's default context is 4096 tokens, and a prompt that exceeds it is
    **silently cut to about half the context, keeping the tail** — so without
    this a long meeting is summarized from its last minutes with no error
    anywhere (measured 2026-08-12: a 26 957-token prompt was evaluated as 2050
    tokens and the marker planted at its head was invisible to the model, while
    the same prompt at ``num_ctx=40960`` evaluated whole). Sized per request
    rather than from ``max_input_chars`` so a short meeting keeps a small, cheap
    context: this is what the machine pays for the fix, at ~138 KB of KV cache
    per token — the same model is 6.5 GB resident at 6796 context and 11 GB at
    40960, which on a GPU is the difference between fitting and spilling.
    """
    chars = sum(len(m.get("content", "")) for m in messages)
    wanted = int(chars / _CHARS_PER_TOKEN) + _RESPONSE_TOKENS
    return -(-wanted // _CONTEXT_BUCKET) * _CONTEXT_BUCKET


_UNASKED = -1
"""A context length no server can report, so "not asked yet" stays
distinguishable from a server that answered "I don't know" (``None``)."""


def _reported_context_length(shown: dict) -> int | None:
    """The context ceiling out of ``/api/show``, if it names one.

    The key is architecture-qualified (``qwen3.context_length``), so it is found
    by suffix rather than by knowing every architecture's name.
    """
    info = shown.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


class ModelNotFoundError(NotesBackendUnavailableError):
    """Ollama runs, but the requested model isn't pulled."""


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        max_input_chars: int | None = None,
    ) -> None:
        self.url = _normalize_url(url or os.environ.get("OLLAMA_HOST") or DEFAULT_URL)
        self.model = model or os.environ.get("STENOGRAF_NOTES_MODEL") or DEFAULT_MODEL
        self.max_input_chars = max_input_chars or DEFAULT_MAX_INPUT_CHARS
        self._model_verified = False
        self._context_ceiling: int | None = _UNASKED

    @classmethod
    def from_settings(cls, settings: NotesSettings) -> OllamaBackend:
        return cls(
            url=settings.ollama_url,
            model=settings.model,
            max_input_chars=settings.max_input_chars,
        )

    @classmethod
    def settings_defaults(cls) -> dict[str, object]:
        return {
            "model": DEFAULT_MODEL,
            "max_input_chars": DEFAULT_MAX_INPUT_CHARS,
        }

    def is_available(self) -> bool:
        try:
            self._get("/api/version")
        except NotesBackendUnavailableError:
            return False
        return True

    def health(self) -> tuple[bool, str]:
        if not self.is_available():
            return (
                False,
                f"Ollama not reachable at {self.url} — start `ollama serve`, or "
                "configure another backend under [notes] in settings.toml",
            )
        try:
            pulled = self.has_model()
        except NotesBackendError as exc:
            return False, str(exc)
        if not pulled:
            return (
                False,
                f"Ollama up, but model {self.model!r} is not pulled "
                f"(`ollama pull {self.model}`)",
            )
        return True, f"Ollama at {self.url}, model {self.model}"

    def installed_models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m["name"] for m in data.get("models", ())]

    def has_model(self) -> bool:
        """Whether :attr:`model` is pulled — "qwen3" matches an installed
        "qwen3:latest"; a fully tagged name is exact."""
        installed = self.installed_models()
        return self.model in set(installed) | {m.split(":", 1)[0] for m in installed}

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._model_verified:
            # One tags round-trip per backend instance, not per map-reduce chunk.
            self._verify_model()
            self._model_verified = True
        wanted = _context_tokens(messages)
        ceiling = self._context_limit()
        if ceiling is not None and wanted > ceiling:
            # Ollama does not refuse an oversized num_ctx, it clamps it — and a
            # clamped context truncates the prompt exactly as no num_ctx at all
            # would, which is the failure this method exists to prevent. Refused
            # here, before a token is spent, rather than detected afterwards.
            raise NotesGenerationError(
                f"this meeting needs about {wanted} tokens of context but "
                f"{self.model!r} tops out at {ceiling} — Ollama would quietly clamp "
                "the request and write the notes from the meeting's final minutes "
                "only; lower [notes] max_input_chars, or use a larger-context model"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": wanted, "num_predict": _RESPONSE_TOKENS},
        }
        data = self._post(
            "/api/chat",
            payload,
            timeout=_CHAT_TIMEOUT,
            timeout_hint=(
                f"Ollama did not finish this meeting within {_CHAT_TIMEOUT:.0f}s "
                f"(model {self.model!r}) — the whole transcript is read now, which on "
                "a CPU-only machine is minutes per meeting-hour; lower [notes] "
                "max_input_chars, or use a smaller or GPU-backed model"
            ),
        )
        self._check_whole_prompt_was_read(messages, data)
        # The server reports why generation ended, and "length" is reachable
        # only because the answer carries an explicit cap. Markdown has no
        # missing closing brace to fail on, so this signal is the only tell.
        if data.get("done_reason") == "length":
            raise NotesGenerationError(
                f"Ollama's answer ran past its {_RESPONSE_TOKENS}-token budget (model "
                f"{self.model!r}) — the answer overran, not the meeting, and a long "
                "reasoning block counts against the same budget (it can consume all of "
                "it and leave no notes at all); shorten the [notes] instructions or try "
                "another model"
            )
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise NotesGenerationError(f"unexpected /api/chat response: {data!r:.200}") from exc

    def _context_limit(self) -> int | None:
        """The model's own context ceiling, or ``None`` if it won't say.

        Asked once per backend and cached: the ceiling decides whether a meeting
        can be summarized whole, and guessing it from the model name is how the
        clamp above goes unnoticed. An endpoint that isn't there is not fatal —
        an API-compatible proxy need not serve it — so it costs the check rather
        than the notes; what such a server truncates anyway is caught after the
        call instead.
        """
        if self._context_ceiling == _UNASKED:
            try:
                shown = self._post("/api/show", {"model": self.model}, timeout=_CONNECT_TIMEOUT)
            except NotesBackendError:
                self._context_ceiling = None
            else:
                self._context_ceiling = _reported_context_length(shown)
        return self._context_ceiling

    def _check_whole_prompt_was_read(self, messages: list[dict[str, str]], data: dict) -> None:
        """Refuse a note written from part of the meeting.

        The backstop for a server that ignores ``options.num_ctx`` — an older
        Ollama, or an API-compatible proxy in front of it — where the prompt is
        cut down and nothing above notices. ``prompt_eval_count`` is the
        server's own count of what it read, and a truncated prompt collapses it
        far under any real tokenizer's floor for the text we sent, which is what
        this compares against rather than our own estimate. A clamp close to
        what we asked for stays above that floor and slips through — which is
        why an ask past the model's ceiling is refused before the call rather
        than looked for afterwards; a proxy clamping far lower is still caught.
        Prompt caching does not deflate the count — measured 2026-08-12, a
        repeated prompt and a shared prefix both re-reported their full size.
        """
        read = data.get("prompt_eval_count")
        if not isinstance(read, int):
            return
        chars = sum(len(m.get("content", "")) for m in messages)
        floor = chars // 8
        if read < floor:
            raise NotesGenerationError(
                f"Ollama read only {read} tokens of a {chars}-character prompt "
                f"(model {self.model!r}) — the server truncated the meeting and the "
                "notes would describe only its final minutes; lower [notes] "
                "max_input_chars, or use a model with a larger context"
            )

    def _verify_model(self) -> None:
        if self.has_model():
            return
        installed = self.installed_models()  # failure path only: re-fetch for the message
        raise ModelNotFoundError(
            f"model {self.model!r} is not pulled in Ollama at {self.url} "
            f"(`ollama pull {self.model}`); installed: {', '.join(installed) or 'none'}"
        )

    def _get(self, endpoint: str) -> dict:
        request = urllib.request.Request(self.url + endpoint)
        return self._send(request, timeout=_CONNECT_TIMEOUT)

    def _post(
        self, endpoint: str, payload: dict, *, timeout: float, timeout_hint: str | None = None
    ) -> dict:
        request = urllib.request.Request(
            self.url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, timeout=timeout, timeout_hint=timeout_hint)

    def _send(
        self, request: urllib.request.Request, *, timeout: float, timeout_hint: str | None = None
    ) -> dict:
        """``timeout_hint`` names what a slow call means where waiting is normal
        — on generation a timeout is a prompt too big for the machine, not a
        server that isn't there, and sending the user to `ollama serve` would be
        the wrong instruction."""
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if timeout_hint is not None and _is_timeout(exc):
                raise NotesGenerationError(timeout_hint) from exc
            raise NotesBackendUnavailableError(
                f"Ollama not reachable at {self.url} ({exc}) — is `ollama serve` running?"
            ) from exc


def _is_timeout(exc: Exception) -> bool:
    """Whether a failed request ran out of time rather than failing outright —
    urllib delivers the same timeout either bare or wrapped in a URLError."""
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)


def _normalize_url(url: str) -> str:
    """Accept OLLAMA_HOST's laxer forms (``host:port``, trailing slash)."""
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url
