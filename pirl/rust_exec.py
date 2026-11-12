"""Optional bridge to the Rust pirl-exec JSONL helper.

The bridge is opt-in via ``PIRL_EXECUTOR`` or an explicit binary path. Python
never silently falls back to an unsafe shell when the Rust helper is selected.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .execution import ExecRequest, ExecResult, ExecutionInfo


class RustExecutor:
    # The current helper constrains argv/resources and requests a private
    # network namespace, but it does not provide a filesystem sandbox.
    info = ExecutionInfo(
        "rust",
        "process limits",
        "host user",
        "private namespace requested",
    )

    def __init__(self, binary: str | Path, workspace: str | Path, *, timeout: float = 120.0):
        self.binary = str(binary)
        self.workspace = str(Path(workspace).expanduser().resolve())
        self.timeout = timeout

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        request = {"op": op, "workspace": self.workspace, **payload}
        proc = subprocess.run(
            [self.binary], input=json.dumps(request) + "\n", capture_output=True,
            text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"rust executor exited {proc.returncode}")
        try:
            result = json.loads(proc.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("rust executor returned invalid JSON") from exc
        if result.get("ok") is False and result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def execute(self, request: ExecRequest) -> ExecResult:
        limits = {
            "timeout_ms": request.timeout_seconds * 1000,
            "max_output_bytes": 20_000,
            "max_memory_bytes": None,
            "max_cpu_seconds": None,
            "network": False,
        }
        result = self.request("exec", argv=list(request.argv), limits=limits)
        return ExecResult(
            result.get("code"),
            result.get("stdout", ""),
            result.get("stderr", ""),
            bool(result.get("timed_out")),
            error="" if result.get("ok") else f"exit {result.get('code')}",
            execution_scope="rust",
        )

    def apply_patch(self, patch: str) -> dict[str, Any]:
        return self.request("patch", patch=patch)

    def append_event(self, journal: str | Path, event: dict[str, Any]) -> dict[str, Any]:
        return self.request("journal", journal=str(journal), event=event)


def executor_from_env(workspace: str | Path) -> RustExecutor | None:
    binary = os.environ.get("PIRL_EXECUTOR", "").strip()
    return RustExecutor(binary, workspace) if binary else None
