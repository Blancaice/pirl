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
    dirty_patch: str = ""

    def to_dict(self) -> dict:
        return {
            "branch": self.branch,
            "head": self.head,
            "dirty": list(self.dirty),
            "dirty_patch": self.dirty_patch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GitSnapshot":
        return cls(
            branch=str(d.get("branch", "")),
            head=str(d.get("head", "")),
            dirty=[str(x) for x in (d.get("dirty") or [])],
            dirty_patch=str(d.get("dirty_patch", "")),
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
        dirty = _git(workspace, "status", "--porcelain").splitlines()
        # Capture tracked pre-run edits so restore can put the workspace back
        # instead of silently destroying them. Untracked files are handled by
        # an explicit refusal in restore because their contents need separate
        # archival (and should not be guessed).
        dirty_patch = _git(workspace, "diff", "HEAD", "--binary")
        return GitSnapshot(
            branch=_git(workspace, "branch", "--show-current"),
            head=_git(workspace, "rev-parse", "HEAD"),
            dirty=dirty,
            dirty_patch=dirty_patch,
        )
    except OSError:
        return None


def restore(workspace: str | Path, snap: GitSnapshot) -> None:
    """Discard working-tree changes since the snapshot (``git reset --hard``)."""
    current = _git(workspace, "rev-parse", "HEAD")
    if current != snap.head:
        raise RuntimeError("HEAD moved since the snapshot; refusing to reset")
    if any(line.startswith("??") for line in snap.dirty):
        raise RuntimeError(
            "snapshot contains pre-existing untracked files; refusing to reset "
            "without an untracked-file archive"
        )
    _git(workspace, "reset", "--hard", snap.head)
    if snap.dirty_patch:
        proc = subprocess.run(
            ["git", "apply", "--binary", "-"],
            cwd=str(workspace),
            input=snap.dirty_patch,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"failed to restore pre-run changes: {proc.stderr.strip()}")
