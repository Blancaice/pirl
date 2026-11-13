"""Durable, atomic checkpoints for resumable agent runs.

The primary journal is now :mod:`pirl.session` (append-only JSONL), which also
acts as the checkpoint, trace, and audit artifact. This module stays for
callers that still write a single JSON snapshot per run, and adds the identity
check that a snapshot needs in order to be replayable: a checkpoint taken
against a different cwd, model, or tool signature must not be silently resumed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

CHECKPOINT_SCHEMA_VERSION = 1
IDENTITY_KEYS = ("cwd", "model", "provider", "tool_signature", "max_steps")


class CheckpointStore:
    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace) / ".pirl" / "checkpoints"

    def save(
        self,
        run_id: str,
        state: Mapping[str, Any],
        identity: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            **dict(state),
        }
        if identity is not None:
            payload["identity"] = dict(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        target, tmp = self.root / f"{run_id}.json", self.root / f"{run_id}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads((self.root / f"{run_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def identity_mismatch(
        state: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> List[str]:
        """Fields that differ between the saved snapshot and the live runtime.

        An empty list means the snapshot is safe to resume. A non-empty list is
        a hard signal: the recorded conversation was produced under different
        assumptions, so continuing from it would silently change semantics.
        """
        recorded = state.get("identity")
        if not isinstance(recorded, Mapping):
            return ["identity"] if state.get("messages") else []
        mismatched: List[str] = []
        for key in IDENTITY_KEYS:
            if key not in recorded:
                continue
            if key in current and recorded.get(key) != current.get(key):
                mismatched.append(key)
        return mismatched

    @staticmethod
    def schema_mismatch(state: Mapping[str, Any]) -> bool:
        return int(state.get("schema_version", CHECKPOINT_SCHEMA_VERSION)) != CHECKPOINT_SCHEMA_VERSION
