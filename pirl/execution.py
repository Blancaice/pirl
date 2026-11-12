"""Composable process execution backends and middleware.

The execution API deliberately does not promise isolation.  Backends report
their boundary through :class:`ExecutionInfo`; a direct backend runs with the
host user's permissions, while future plugins may provide containers, remote
workers, or stronger sandboxes without changing the model-facing tool.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, Mapping, Protocol

EXECUTOR_ENTRY_POINT_GROUP = "pirl.executors"


@dataclass(frozen=True)
class ExecRequest:
    """A host-reviewable request to start one process without a shell."""

    argv: tuple[str, ...]
    workspace: Path
    reason: str = ""
    timeout_seconds: int = 60
    env: Mapping[str, str] | None = None
    approved_by_host: bool = False


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    denied: bool = False
    error: str = ""
    execution_scope: str = "host"

    @property
    def ok(self) -> bool:
        return not self.denied and not self.timed_out and not self.error and self.exit_code == 0


@dataclass(frozen=True)
class ExecutionInfo:
    name: str
    isolation: str
    filesystem_boundary: str
    network_boundary: str
    approval: str = "unspecified"


class Executor(Protocol):
    @property
    def info(self) -> ExecutionInfo: ...

    def execute(self, request: ExecRequest) -> ExecResult: ...


class DirectExecutor:
    """Run argv directly on the host.  This backend provides no isolation."""

    info = ExecutionInfo("direct", "none", "host user", "host")

    def execute(self, request: ExecRequest) -> ExecResult:
        try:
            proc = subprocess.run(
                request.argv,
                shell=False,
                cwd=request.workspace,
                env=dict(request.env) if request.env is not None else None,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                None,
                exc.stdout or "",
                exc.stderr or "",
                timed_out=True,
                error=f"process timed out after {request.timeout_seconds}s",
            )
        except OSError as exc:
            return ExecResult(None, error=str(exc))
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)


class DisabledExecutor:
    info = ExecutionInfo("none", "disabled", "no process execution", "disabled")

    def execute(self, request: ExecRequest) -> ExecResult:
        return ExecResult(
            None,
            denied=True,
            error="process execution disabled by executor",
            execution_scope="disabled",
        )


class ExecutorMiddleware:
    def __init__(self, inner: Executor) -> None:
        self.inner = inner

    @property
    def info(self) -> ExecutionInfo:
        return self.inner.info


PolicyCheck = Callable[[ExecRequest], str]
Approver = Callable[[ExecRequest], bool]
AuditSink = Callable[[str, ExecRequest, ExecResult | None], None]


class PolicyExecutor(ExecutorMiddleware):
    def __init__(self, inner: Executor, check: PolicyCheck) -> None:
        super().__init__(inner)
        self.check = check

    def execute(self, request: ExecRequest) -> ExecResult:
        reason = self.check(request)
        if reason:
            return ExecResult(
                None,
                denied=True,
                error=reason,
                execution_scope=self.info.name,
            )
        return self.inner.execute(request)


class ApprovalExecutor(ExecutorMiddleware):
    def __init__(self, inner: Executor, mode: str, approver: Approver | None = None) -> None:
        super().__init__(inner)
        if mode not in {"on-request", "auto", "never"}:
            raise ValueError("approval mode must be on-request, auto, or never")
        self.mode = mode
        self.approver = approver

    @property
    def info(self) -> ExecutionInfo:
        return replace(self.inner.info, approval=self.mode)

    def execute(self, request: ExecRequest) -> ExecResult:
        if request.approved_by_host:
            return self.inner.execute(request)
        if self.inner.info.isolation == "disabled":
            return self.inner.execute(request)
        if self.mode == "never":
            return ExecResult(
                None,
                denied=True,
                error="process execution denied by approval policy",
                execution_scope=self.info.name,
            )
        if self.mode == "on-request" and (
            self.approver is None or not self.approver(request)
        ):
            return ExecResult(
                None,
                denied=True,
                error="process execution was not approved",
                execution_scope=self.info.name,
            )
        return self.inner.execute(request)


class AuditExecutor(ExecutorMiddleware):
    def __init__(self, inner: Executor, sink: AuditSink) -> None:
        super().__init__(inner)
        self.sink = sink

    def execute(self, request: ExecRequest) -> ExecResult:
        self.sink("requested", request, None)
        try:
            result = self.inner.execute(request)
        except Exception:
            self.sink("failed", request, None)
            raise
        self.sink("completed", request, result)
        return result


ExecutorFactory = Callable[[Path], Executor]


class ExecutorRegistry:
    """Small plugin registry mapping backend names to workspace factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, ExecutorFactory] = {}

    def register(self, name: str, factory: ExecutorFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"executor already registered or invalid: {name!r}")
        self._factories[name] = factory

    def create(self, name: str, workspace: str | Path) -> Executor:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "(none)"
            raise ValueError(f"unknown executor {name!r}; available: {available}") from exc
        return factory(Path(workspace).expanduser().resolve())

    def load_entry_point(self, name: str) -> bool:
        """Load one explicitly selected third-party executor plugin."""
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            matches = list(discovered.select(group=EXECUTOR_ENTRY_POINT_GROUP, name=name))
        else:  # pragma: no cover - compatibility with older importlib.metadata
            matches = [
                item
                for item in discovered.get(EXECUTOR_ENTRY_POINT_GROUP, ())
                if item.name == name
            ]
        if not matches:
            return False
        if len(matches) > 1:
            raise ValueError(f"multiple executor plugins registered as {name!r}")
        factory = matches[0].load()
        if not callable(factory):
            raise TypeError(f"executor plugin {name!r} is not callable")
        self.register(name, factory)
        return True

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def builtin_executor_registry() -> ExecutorRegistry:
    registry = ExecutorRegistry()
    registry.register("direct", lambda _workspace: DirectExecutor())
    registry.register("none", lambda _workspace: DisabledExecutor())
    return registry

