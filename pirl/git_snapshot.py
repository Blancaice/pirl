"""Git snapshot/rollback for workspace safety (coding-agent-harness: 回退要可靠).

Before a run, the pre-run ``HEAD`` (and dirty file list) is recorded; a
``restore`` command can discard working-tree changes made during the run. Only
``git reset --hard`` is used — untracked files (such as ``.pirl/``) are kept, so
run history and memory survive a rollback.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class GitSnapshot:
    branch: str
    head: str
    dirty: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"branch": self.branch, "head": self.head, "dirty": list(self.dirty)}

    @classmethod
    def from_dict(cls, d: dict) -> "GitSnapshot":
        return cls(
            branch=str(d.get("branch", "")),
            head=str(d.get("head", "")),
            dirty=[str(x) for x in (d.get("dirty") or [])],
        )


def _git(workspace: str | Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(workspace), capture_output=True, text=True
    )
    return proc.stdout.strip()


def is_git(workspace: str | Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(workspace),
            capture_output=True,
        )
        return proc.returncode == 0
    except OSError:
        return False


def snapshot(workspace: str | Path) -> Optional[GitSnapshot]:
    """Record the pre-run git state, or ``None`` when not a git repo."""
    if not is_git(workspace):
        return None
    try:
        return GitSnapshot(
            branch=_git(workspace, "branch", "--show-current"),
            head=_git(workspace, "rev-parse", "HEAD"),
            dirty=_git(workspace, "status", "--porcelain").splitlines(),
        )
    except OSError:
        return None


def restore(workspace: str | Path, snap: GitSnapshot) -> None:
    """Discard working-tree changes since the snapshot (``git reset --hard``)."""
    current = _git(workspace, "rev-parse", "HEAD")
    if current != snap.head:
        raise RuntimeError("HEAD moved since the snapshot; refusing to reset")
    _git(workspace, "reset", "--hard", snap.head)
