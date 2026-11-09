"""Durable cross-run history: full trajectories on disk, retrieved on demand.

Each run is written as ``<cwd>/.pirl/runs/run_<id>.json``. By default only the
compact ``last-run.md`` summary is injected into the prompt; the full history is
reached through the ``list_runs`` / ``read_run`` / ``search_history`` tools, so
context stays bounded (the same pattern as on-demand skill loading).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR_NAME = ".pirl"
RUNS_DIR_NAME = "runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


@dataclass
class RunRecord:
    """One completed run's durable record."""

    run_id: str
    task: str
    outcome: str
    answer: str
    started_at: str
    trace: List[Dict[str, str]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    git: Optional[Dict[str, Any]] = None  # pre-run git snapshot (branch/head/dirty)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(d.get("run_id", "")),
            task=str(d.get("task", "")),
            outcome=str(d.get("outcome", "")),
            answer=str(d.get("answer", "")),
            started_at=str(d.get("started_at", "")),
            trace=[
                {str(k): str(v) for k, v in t.items()}
                for t in (d.get("trace") or [])
                if isinstance(t, dict)
            ],
            usage={
                str(k): int(v) for k, v in (d.get("usage") or {}).items()
            },
            git=d.get("git") if isinstance(d.get("git"), dict) else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunStore:
    """Reads/writes run records under ``<cwd>/.pirl/runs/``."""

    def __init__(self, workspace: str | Path) -> None:
        self.runs_dir = Path(workspace) / CONFIG_DIR_NAME / RUNS_DIR_NAME

    def write_run(self, record: RunRecord) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"run_{record.run_id}.json"
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def list_runs(self) -> List[Dict[str, str]]:
        """Summaries (newest first): run_id, task, outcome, started_at."""
        out: List[Dict[str, str]] = []
        if not self.runs_dir.is_dir():
            return out
        for path in sorted(self.runs_dir.glob("run_*.json"), reverse=True):
            record = self._load(path)
            if record is not None:
                out.append(
                    {
                        "run_id": record.run_id,
                        "task": record.task[:200],
                        "outcome": record.outcome,
                        "started_at": record.started_at,
                    }
                )
        return out

    def read_run(self, run_id: str) -> Optional[RunRecord]:
        return self._load(self.runs_dir / f"run_{run_id}.json")

    def search(self, query: str) -> List[Dict[str, str]]:
        """Return runs whose task/answer/trace mention ``query`` (newest first)."""
        hits: List[Dict[str, str]] = []
        if not self.runs_dir.is_dir():
            return hits
        for path in sorted(self.runs_dir.glob("run_*.json"), reverse=True):
            record = self._load(path)
            if record is None:
                continue
            haystack = record.task + "\n" + record.answer + "\n" + json.dumps(record.trace)
            if query.lower() in haystack.lower():
                hits.append(
                    {
                        "run_id": record.run_id,
                        "task": record.task[:200],
                        "outcome": record.outcome,
                    }
                )
        return hits

    def _load(self, path: Path) -> Optional[RunRecord]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RunRecord.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
