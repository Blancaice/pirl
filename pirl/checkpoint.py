"""Durable, atomic checkpoints for resumable agent runs."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict

class CheckpointStore:
    def __init__(self, workspace: str | Path):
        self.root = Path(workspace) / ".pirl" / "checkpoints"
    def save(self, run_id: str, state: Dict[str, Any]) -> Path:
        state = {"schema_version": 1, **state}
        self.root.mkdir(parents=True, exist_ok=True)
        target, tmp = self.root / f"{run_id}.json", self.root / f"{run_id}.tmp"
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return target
    def load(self, run_id: str) -> Dict[str, Any] | None:
        try:
            return json.loads((self.root / f"{run_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
