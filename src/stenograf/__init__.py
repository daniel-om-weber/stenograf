"""stenograf — accuracy-first local meeting transcription. Audio never touches disk."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stenograf")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0.dev0"
