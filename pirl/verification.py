"""Deterministic completion verification, independent from AgentRunner."""
from __future__ import annotations
import subprocess
from pathlib import Path

class CommandVerifier:
    def __init__(self, command: str, workspace: str | Path = ".", timeout: int = 120):
        self.command, self.workspace, self.timeout = command, Path(workspace), timeout
    def __call__(self, _command: str | None = None) -> tuple[bool, str]:
        try:
            proc = subprocess.run(self.command, shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {self.timeout}s"
        except OSError as exc:
            return False, str(exc)
        if proc.returncode == 0:
            return True, ""
        output = proc.stdout + (("\n" if proc.stdout else "") + proc.stderr if proc.stderr else "")
        return False, output[:2000] or f"exit {proc.returncode}"
