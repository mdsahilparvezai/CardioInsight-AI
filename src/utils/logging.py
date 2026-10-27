"""Lightweight logging helper.

Wraps the stdlib ``logging`` module with a single ``get_logger`` factory.
Keeps the dependency footprint at zero and the behaviour predictable.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("cardioinsight")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``cardioinsight`` namespace.

    The first call configures a single stderr handler.  Subsequent calls
    return loggers that inherit that configuration.
    """
    _configure()
    if not name.startswith("cardioinsight"):
        name = f"cardioinsight.{name}"
    return logging.getLogger(name)