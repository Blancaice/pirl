"""Small structural interfaces that keep the Agent kernel decoupled.

Implementations may live in the CLI, a plugin, the Rust executor, or an eval
harness.  The kernel only needs these shapes and does not need to own their
persistence or policy.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence


class ModelProvider(Protocol):
    def complete(self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Any] = ()) -> Any:
        ...


class ToolProvider(Protocol):
    def specs(self) -> Sequence[Any]:
        ...

    def execute(self, name: str, args: Mapping[str, Any]) -> Any:
        ...

    def can_parallel(self, names: Sequence[str]) -> bool:
        ...


class Verifier(Protocol):
    def __call__(self, command: str | None = None) -> tuple[bool, str]:
        ...


class EventSink(Protocol):
    def __call__(self, event: Any) -> None:
        ...

class PromptBuilder(Protocol):
    def __call__(self) -> str:
        ...


class CheckpointWriter(Protocol):
    def __call__(self, state: Mapping[str, Any]) -> None:
        ...


class AgentExtension(Protocol):
    """Optional lifecycle hooks; all methods are optional at runtime."""

    def on_retrieve(self, name: str) -> None:
        ...

    def on_run_end(self, *, trace: list, outcome: str, applied: set[str]) -> None:
        ...
