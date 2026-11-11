"""Task execution and evaluation orchestration around the small Agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .git_snapshot import snapshot
from .run_history import RunRecord, RunStore, new_run_id, now_iso
from .run_memory import write_memory


@dataclass(frozen=True)
class TaskSpec:
    """A reproducible task boundary understood by a harness."""

    task: str
    workspace: Path
    max_steps: int = 20
    verify: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_values(cls, task: str, workspace: str | Path, *, max_steps: int = 20,
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
        result = agent.run(spec.task)
        write_memory(self.workspace, result.outcome, result.answer or result.message)
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
        )
        path = self.runs.write_run(record)
        return HarnessResult(run_id, result, path, record.git)
