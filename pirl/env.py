"""Minimal ``.env`` loader.

Reads ``KEY=VALUE`` lines and exports them to ``os.environ`` *without* overriding
variables already set in the environment (so an explicit ``export`` wins). Values
are literal: no shell expansion, no ``$VAR`` interpolation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load ``path`` (default ``.env`` in the cwd); return the applied pairs.

    Keys already present in ``os.environ`` are left untouched.
    """
    target = Path(path) if path is not None else Path(".env")
    if not target.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
