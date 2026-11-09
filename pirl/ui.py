"""Minimal interactive UI primitives with non-interactive fallbacks.

Mirrors the shape of pi's ``ctx.ui.confirm`` / ``ctx.ui.select``: callers get
an answer even when stdin is not a terminal, so the same code path works in TTY
and piped/headless mode.
"""

from __future__ import annotations

import sys
from typing import Mapping, Optional


def is_interactive() -> bool:
    """True when stdin is a terminal (a human can answer)."""
    return sys.stdin.isatty()


def confirm(question: str, *, default: bool = False) -> bool:
    """Ask a yes/no question; return ``default`` when not interactive."""
    if not is_interactive():
        return default
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {hint} ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if answer == "":
            return default
        print("answer y or n", file=sys.stderr)


def select(
    prompt: str,
    choices: Mapping[str, str],
    *,
    default: Optional[str] = None,
) -> str:
    """Let the user pick one key; return ``default`` (or the first key) when not interactive.

    ``choices`` maps a short key (what the user types) to a longer label.
    """
    if not choices:
        raise ValueError("select() requires at least one choice")

    if not is_interactive():
        if default is not None and default in choices:
            return default
        return next(iter(choices))

    while True:
        print(prompt, file=sys.stderr)
        for key, label in choices.items():
            print(f"  [{key}] {label}", file=sys.stderr)
        answer = input("> ").strip().lower()
        if answer in choices:
            return answer
        if answer == "" and default is not None and default in choices:
            return default
        print(f"input one of: {', '.join(choices)}", file=sys.stderr)
