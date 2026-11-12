"""Runtime hooks keeping persistence and skill bookkeeping out of the runner."""
from __future__ import annotations
from typing import Any, Set
from ..lifecycle import SkillStore

class SkillUsageHook:
    def __init__(self, store: SkillStore):
        self.store = store
    def on_retrieve(self, name: str) -> None:
        self.store.on_retrieve(name)
    def on_run_end(self, *, trace: list, outcome: str, applied: Set[str], **_metadata) -> None:
        for name in applied:
            self.store.on_apply(name, outcome == "success")


class SelfGenerationHook:
    """Adapter that keeps self-generation out of the runner core."""
    def __init__(self, loop: Any):
        self.loop, self.result = loop, None
    def on_run_end(self, *, trace: list, outcome: str, message: str = "",
                   verification_failed: bool = False, run_id: str = "",
                   applied: Set[str]) -> None:
        callback = self.loop.on_run_end
        try:
            callback(trace=trace, outcome=outcome, message=message,
                     verification_failed=verification_failed, run_id=run_id)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            callback(trace, outcome)
        self.result = getattr(self.loop, "result", None)
