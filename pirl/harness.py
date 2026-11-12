"""Task execution and evaluation orchestration around the small Agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .plugins.git_safety import snapshot
from .run_history import RunRecord, RunStore, new_run_id, now_iso
from .plugins.memory import write_memory


@dataclass(frozen=True)
class TaskSpec:
    """A reproducible task boundary understood by a harness."""

    task: str
    workspace: Path
    max_steps: int | None = None
    verify: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_values(cls, task: str, workspace: str | Path, *, max_steps: int | None = None,
                    verify: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        return cls(task, Path(workspace).expanduser().resolve(), max_steps, verify, dict(metadata or {}))


@dataclass
class HarnessResult:
    run_id: str
    result: Any
    record_path: Path
    git_snapshot: Optional[Dict[str, Any]] = None


class TaskHarness:
    """Run one configured Agent and persist its run boundary."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runs = RunStore(self.workspace)

    def run(self, agent: Any, spec: TaskSpec) -> HarnessResult:
        if spec.workspace != self.workspace:
            raise ValueError("task workspace does not match harness workspace")
        run_id = new_run_id()
        started_at = now_iso()
        git = snapshot(self.workspace)
        try:
            result = agent.run(spec.task, run_id=run_id)
        except TypeError as exc:
            if "run_id" not in str(exc):
                raise
            result = agent.run(spec.task)
        write_memory(self.workspace, result.outcome, result.answer or result.message)
        executor = getattr(getattr(agent, "tools", None), "executor", None)
        execution_info = getattr(executor, "info", None)
        record = RunRecord(
            run_id=run_id,
            task=spec.task,
            outcome=result.outcome,
            answer=result.answer or result.message,
            started_at=started_at,
            trace=[{"tool": tc.name, "args": tc.args} for tc in result.trace],
            usage={"prompt_tokens": result.usage.prompt_tokens,
                   "completion_tokens": result.usage.completion_tokens,
                   "total_tokens": result.usage.total_tokens},
            git=git.to_dict() if git else None,
            execution=(
                {
                    "backend": execution_info.name,
                    "isolation": execution_info.isolation,
                    "filesystem_boundary": execution_info.filesystem_boundary,
                    "network_boundary": execution_info.network_boundary,
                    "approval": execution_info.approval,
                }
                if execution_info is not None
                else None
            ),
        )
        path = self.runs.write_run(record)
        return HarnessResult(run_id, result, path, record.git)
