"""Runtime hooks keeping persistence and skill bookkeeping out of the runner."""
from __future__ import annotations
from typing import Any, Set
from .lifecycle import SkillStore

class SkillUsageHook:
    def __init__(self, store: SkillStore):
        self.store = store
    def on_retrieve(self, name: str) -> None:
        self.store.on_retrieve(name)
    def on_run_end(self, *, trace: list, outcome: str, applied: Set[str]) -> None:
        for name in applied:
            self.store.on_apply(name, outcome == "success")


class SelfGenerationHook:
    """Adapter that keeps self-generation out of the runner core."""
    def __init__(self, loop: Any):
        self.loop, self.result = loop, None
    def on_run_end(self, *, trace: list, outcome: str, applied: Set[str]) -> None:
        self.result = self.loop.on_run_end(trace, outcome)
