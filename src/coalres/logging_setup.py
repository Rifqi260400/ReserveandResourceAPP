"""Logging terstruktur. Tidak ada `print` telanjang di jalur pustaka."""
from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


class _Formatter(logging.Formatter):
    LEVEL_TAG = {
        logging.DEBUG: "debug", logging.INFO: "info",
        logging.WARNING: "WARN", logging.ERROR: "ERROR", logging.CRITICAL: "STOP",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = self.LEVEL_TAG.get(record.levelno, record.levelname.lower())
        base = f"{tag:>5} | {record.getMessage()}"
        extras: dict[str, Any] = getattr(record, "context", {}) or {}
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup(verbosity: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Formatter())
    root = logging.getLogger("coalres")
    root.setLevel(verbosity)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter:
    setup()
    return logging.LoggerAdapter(logging.getLogger(f"coalres.{name}"), {})
