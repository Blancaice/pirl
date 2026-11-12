"""Injectable context management for the agent runner."""
from __future__ import annotations
from typing import Any, Callable, Dict, List
from .plugins.compaction import compact, estimate_messages_tokens

class ContextManager:
    def __init__(self, summarize: Callable[[str], str], budget: int | None = 16000, keep_recent: int = 6):
        self.summarize, self.budget, self.keep_recent = summarize, budget, keep_recent
    def maybe_compact(self, messages: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        if self.budget and estimate_messages_tokens(messages) > self.budget:
            return compact(self.summarize, messages, task, keep_recent=self.keep_recent)
        return messages
