"""Logging for the API and the worker: one plain line per record on stdout, which Railway collects.

Configured before anything else touches logging (the mcp SDK calls logging.basicConfig when an MCP server is
built, which is a no-op once the root logger has a handler). Records never carry bodies, queries or keys.
"""
from __future__ import annotations

import logging
import sys

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_brain", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._brain = True   # mark ours, so a second create_app() does not add another
        root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)     # one INFO line per provider call is noise
