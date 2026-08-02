"""The diagnostics channel: ``logging.getLogger("stenograf")``, silent by default.

User-facing output has its own surfaces — ``LiveView`` and click for the CLI,
the status lines and dialogs for the Qt app. This logger carries the
diagnostics *behind* them: fallback decisions, refused OS handles, capture
transport chatter. Those used to go to stderr, which an app-icon launch and
``pythonw`` do not have, so they were silently lost exactly where they were
needed. Set ``STENOGRAF_DEBUG`` (any non-empty value) to see them on stderr;
redirecting to a file is the shell's job.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("stenograf")
logger.addHandler(logging.NullHandler())

if os.environ.get("STENOGRAF_DEBUG"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
