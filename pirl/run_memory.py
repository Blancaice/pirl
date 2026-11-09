"""Workspace-scoped memory of the previous run (counteracts run-to-run amnesia).

One plaintext note per workspace (``<cwd>/.pirl/last-run.md``) records the last
run's outcome and answer. The next run injects it as an *appended prompt
section*, never spliced into the persona, so the stable prefix stays byte-stable.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR_NAME = ".pirl"
MEMORY_FILE_NAME = "last-run.md"


def memory_path(workspace: str | Path) -> Path:
    """Path of the workspace's last-run note."""
    return Path(workspace) / CONFIG_DIR_NAME / MEMORY_FILE_NAME


def read_memory(workspace: str | Path) -> str:
    """Return the previous-run note, or ``""`` when none exists."""
    try:
        return memory_path(workspace).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_memory(workspace: str | Path, outcome: str, summary: str) -> None:
    """Persist the run's outcome and final summary as the next run's memory."""
    path = memory_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"outcome: {outcome}\n\n{summary}".strip() + "\n"
    path.write_text(content, encoding="utf-8")
