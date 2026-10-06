"""Where Trilha keeps its state. Everything is local to this machine."""

from __future__ import annotations

import os
from pathlib import Path

ACOUSTIC_VERSION = "3.0"
GEO_VERSION = "3.0"
BACKEND = "onnx"

OLLAMA_URL = os.environ.get("TRILHA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("TRILHA_OLLAMA_MODEL", "gemma3:4b")


def home() -> Path:
    """Root of Trilha's local state, overridable with TRILHA_HOME."""
    override = os.environ.get("TRILHA_HOME")
    root = Path(override) if override else Path.home() / ".local" / "share" / "trilha"
    root.mkdir(parents=True, exist_ok=True)
    return root


def journal_path() -> Path:
    return home() / "journal.jsonl"


def geo_cache_path() -> Path:
    return home() / "geo-cache.json"
