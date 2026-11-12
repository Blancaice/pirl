"""Persistent, verifier-backed failure-pattern learning plugin."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from ..proposal import Proposal

EXCLUDED = {
    "infra": ("connection", "network", "provider returned http", "cannot reach provider", "service unavailable"),
    "auth": ("unauthorized", "forbidden", "authentication", "invalid api key", "401", "403"),
    "timeout": ("timeout", "timed out"),
    "cancel": ("cancelled", "canceled"),
}


def classify_failure(message: str) -> str:
    text = message.lower()
    for category, markers in EXCLUDED.items():
        if any(marker in text for marker in markers):
            return category
    return "task"


def failure_signature(message: str, trace: Sequence[Any]) -> str:
    normalized = re.sub(r"\b\d+\b", "#", message.lower())[:1000]
    names = [getattr(call, "name", str(call)) for call in trace]
    payload = json.dumps({"message": normalized, "tools": names}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class FailureCandidateStore:
    """Atomic JSON ledger of repeated signatures, scoped to a workspace."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def record(self, signature: str, message: str, run_id: str = "") -> int:
        data = self._read()
        row = data.setdefault(signature, {"count": 0, "message": message[:1000], "runs": []})
        if run_id and run_id in row["runs"]:
            return int(row["count"])
        row["count"] += 1
        if run_id:
            row["runs"].append(run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)
        return int(row["count"])


class SkillLearningPlugin:
    """Accumulate verifier-backed task failures; generate only at threshold."""
    def __init__(self, candidates: FailureCandidateStore,
                 generator: Callable[[Sequence, str], Proposal], *, threshold: int = 3,
                 proposal_handler: Callable[[Proposal], Any] | None = None):
        self.candidates = candidates
        self.generator = generator
        self.threshold = max(2, threshold)
        self.proposal_handler = proposal_handler
        self.result: Any = None

    def on_run_end(self, *, trace: Sequence, outcome: str, message: str = "",
                   verification_failed: bool = False, run_id: str = "",
                   applied: set[str] | None = None) -> None:
        self.result = None
        if outcome != "failure" or not verification_failed or classify_failure(message) != "task":
            return
        signature = failure_signature(message, trace)
        if self.candidates.record(signature, message, run_id) < self.threshold:
            return
        try:
            proposal = self.generator(trace, message)
            self.result = self.proposal_handler(proposal) if self.proposal_handler else proposal
        except Exception as exc:
            self.result = exc
