"""Stdlib logging setup — library code logs, CLI prints."""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"evidence_engine.{name}")
