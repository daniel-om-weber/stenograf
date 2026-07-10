"""Fully local notes backend: Ollama over plain HTTP (stdlib only).

No ``ollama`` pip dependency — the three endpoints we need (`/api/version`,
`/api/tags`, `/api/chat`) are a handful of ``urllib`` calls, and staying
stdlib keeps the reserved ``stenograf[ollama]`` extra empty (PLAN.md §5 D2).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from stenograf.notes.backend import NotesBackendUnavailableError, NotesGenerationError

if TYPE_CHECKING:
    from stenograf.settings import NotesSettings

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"  # ~5 GB — fits a 48 GB Mac without swapping
DEFAULT_MAX_INPUT_CHARS = 128_000
"""~32k tokens — a ~2 h meeting in one pass. Qwen3-class local models handle
this; if a smaller model truncates or rambles, lower ``[notes]
max_input_chars`` in settings.toml rather than this default."""
_CONNECT_TIMEOUT = 5.0
_CHAT_TIMEOUT = 600.0


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

    @classmethod
    def from_settings(cls, settings: NotesSettings) -> OllamaBackend:
        return cls(
            url=settings.ollama_url,
            model=settings.model,
            max_input_chars=settings.max_input_chars,
        )

    def is_available(self) -> bool:
        try:
            self._get("/api/version")
        except NotesBackendUnavailableError:
            return False
        return True

    def installed_models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m["name"] for m in data.get("models", ())]

    def complete(self, messages: list[dict[str, str]], schema: dict) -> str:
        if not self._model_verified:
            # One tags round-trip per backend instance, not per map-reduce chunk.
            self._verify_model()
            self._model_verified = True
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": False,
        }
        data = self._post("/api/chat", payload, timeout=_CHAT_TIMEOUT)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise NotesGenerationError(f"unexpected /api/chat response: {data!r:.200}") from exc

    def _verify_model(self) -> None:
        installed = self.installed_models()
        # "qwen3" matches an installed "qwen3:latest"; a fully tagged name is exact.
        names = set(installed) | {m.split(":", 1)[0] for m in installed}
        if self.model not in names:
            raise ModelNotFoundError(
                f"model {self.model!r} is not pulled in Ollama at {self.url} "
                f"(`ollama pull {self.model}`); installed: {', '.join(installed) or 'none'}"
            )

    def _get(self, endpoint: str) -> dict:
        request = urllib.request.Request(self.url + endpoint)
        return self._send(request, timeout=_CONNECT_TIMEOUT)

    def _post(self, endpoint: str, payload: dict, *, timeout: float) -> dict:
        request = urllib.request.Request(
            self.url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, timeout=timeout)

    def _send(self, request: urllib.request.Request, *, timeout: float) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise NotesBackendUnavailableError(
                f"Ollama not reachable at {self.url} ({exc}) — is `ollama serve` running?"
            ) from exc


def _normalize_url(url: str) -> str:
    """Accept OLLAMA_HOST's laxer forms (``host:port``, trailing slash)."""
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url
