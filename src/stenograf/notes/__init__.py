"""Meeting notes: LLM-generated summaries over finalized transcripts.

Stage D of the Phase 4 product layer. The LLM is a *backend
choice*, not a dependency: prompt building, chunking, the markdown template,
and response validation all live here and are shared by every backend, so a
provider is one line of configuration — the in-process ``mlx`` backend (ships
with stenograf on Apple Silicon, zero setup), the ``ollama`` backend (local
server), or the ``command`` backend that drives any CLI (e.g. ``claude -p``)
via stdin/stdout.
"""

from stenograf.notes.backend import (
    NotesBackend,
    NotesBackendError,
    NotesBackendSpec,
    NotesBackendUnavailableError,
    NotesGenerationError,
    available_backends,
    create_backend,
    default_backend_name,
    get_spec,
    register_backend,
)
from stenograf.notes.model import MeetingNotes, NotesProvenance

__all__ = [
    "MeetingNotes",
    "NotesBackend",
    "NotesBackendError",
    "NotesBackendSpec",
    "NotesBackendUnavailableError",
    "NotesGenerationError",
    "NotesProvenance",
    "available_backends",
    "create_backend",
    "default_backend_name",
    "get_spec",
    "register_backend",
]
