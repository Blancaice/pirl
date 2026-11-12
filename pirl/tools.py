"""Built-in workspace tools and the registry the agent dispatches through.

File tools reject paths that resolve outside ``workspace``. ``bash`` runs with
``cwd=workspace`` and passes through an explicit :class:`ExecutionPolicy`; this
is a command-policy gate, not an OS-level sandbox. Hosts requiring isolation
must inject a sandboxed executor. Tool output is truncated to bound context.
"""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import json
import re
import shlex

from .execution import ApprovalExecutor, DirectExecutor, ExecRequest, Executor, PolicyExecutor
from .llm import ToolSpec
from .run_history import RunStore

MAX_OUTPUT = 20_000
_SEARCH_SKIP = frozenset({".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"})


@dataclass(frozen=True)
class ToolResult:
    """The text a tool returns to the model; ``error`` marks a failure."""

    content: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not bool(self.error)

    def to_message(self, *, source: str | None = None, trust: str | None = None) -> str:
        """Stable, machine-readable envelope for the model-facing protocol."""
        payload = {"ok": self.ok, "content": self.content, "error": self.error}
        if source is not None:
            payload["source"] = source
        if trust is not None:
            payload["trust"] = trust
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Policy gate for process execution.

    This is deliberately not advertised as a sandbox: OS-level isolation must
    be supplied by the host. The default blocks common escape/destructive
    commands while keeping normal repository commands usable.
    """

    allow_bash: bool = True
    allow_exec: bool = True
    deny_patterns: tuple[str, ...] = (
        r"\brm\s+(-[^\s]*f[^\s]*\s+)*-?r",
        r"\bgit\s+reset\s+--hard\b",
        r"\b(curl|wget|ssh|scp|nc|telnet)\b",
        r"(?:^|[\s;|&])/(?:etc|home|root|proc|sys|dev)(?:/|\b)",
    )

    def check_bash(self, command: str) -> str:
        if not self.allow_bash:
            return "bash execution disabled by policy"
        for pattern in self.deny_patterns:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return f"bash command rejected by policy: {pattern}"
        return ""

    def check_exec(self, argv: Sequence[str]) -> str:
        if not self.allow_exec:
            return "process execution disabled by policy"
        # Reuse the host's hard-deny rules for structured execution.  This is
        # a policy gate, not filesystem isolation: child processes can still
        # access anything available to the host user.
        rendered = shlex.join(argv)
        for pattern in self.deny_patterns:
            if re.search(pattern, rendered, flags=re.IGNORECASE):
                return f"exec argv rejected by policy: {pattern}"
        return ""


@dataclass(frozen=True)
class Tool:
    """A model-facing tool: JSON-Schema declaration plus an execute function."""

    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[[Dict[str, Any], Path], ToolResult]
    parallel_safe: bool = False

    def to_spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    """Name -> Tool lookup; resolves paths against a fixed workspace."""

    def __init__(
        self,
        workspace: str | Path,
        tools: Sequence[Tool] = (),
        *,
        policy: ExecutionPolicy | None = None,
        executor: Executor | None = None,
        approval_mode: str = "on-request",
        approver: Callable[[ExecRequest], bool] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.policy = policy or ExecutionPolicy()
        if executor is None:
            configured: Executor = DirectExecutor()
            configured = ApprovalExecutor(configured, approval_mode, approver)
            configured = PolicyExecutor(
                configured, lambda request: self.policy.check_exec(request.argv)
            )
            self.executor = configured
        else:
            self.executor = executor
        self._tools: Dict[str, Tool] = {t.name: t for t in tools}

    def specs(self) -> List[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def can_parallel(self, names: Sequence[str]) -> bool:
        """Only explicitly read-only tools may run concurrently."""
        return bool(names) and all(self._tools.get(name) is not None and self._tools[name].parallel_safe for name in names)

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("", f"unknown tool: {name}")
        if name == "bash":
            command = args.get("command", "")
            if not isinstance(command, str):
                return ToolResult("", "command must be a string")
            reason = self.policy.check_bash(command)
            if reason:
                return ToolResult("", reason)
        if name == "exec":
            argv = args.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(item, str) or not item for item in argv)
            ):
                return ToolResult("", "argv must be a non-empty array of non-empty strings")
            timeout = args.get("timeout_seconds", 60)
            if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 600:
                return ToolResult("", "timeout_seconds must be an integer between 1 and 600")
            request = ExecRequest(
                tuple(argv), self.workspace, str(args.get("reason", "")), timeout
            )
            try:
                result = self.executor.execute(request)
            except Exception as exc:
                return ToolResult("", f"executor failed: {exc}")
            content = result.stdout
            if result.stderr:
                content += ("\n[stderr]\n" if content else "[stderr]\n") + result.stderr
            if result.error and not content:
                return ToolResult("", result.error)
            error = "" if result.ok else result.error or f"exit {result.exit_code}"
            display = content or f"(no output, exit {result.exit_code})"
            return ToolResult(_truncate(display), error)
        patch_executor = self.executor
        while hasattr(patch_executor, "inner"):
            patch_executor = patch_executor.inner
        if name == "apply_patch" and hasattr(patch_executor, "apply_patch"):
            try:
                result = patch_executor.apply_patch(str(args.get("patch", "")))
                return ToolResult("patch applied" if result.get("ok", True) else "", str(result.get("error", "")))
            except Exception as exc:
                return ToolResult("", f"rust executor failed: {exc}")
        try:
            return tool.execute(args, self.workspace)
        except Exception as exc:  # noqa: BLE001 - tool impls touch fs/process
            return ToolResult("", f"tool {name} raised: {exc}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


def _resolve(workspace: Path, raw: str) -> Optional[Path]:
    """Resolve a relative-to-workspace or absolute path; ``None`` if it escapes."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    if not p.is_relative_to(workspace):
        return None
    return p


# --------------------------------------------------------------------------- #
# Built-in tools
# --------------------------------------------------------------------------- #


def _list_files(args: Dict[str, Any], workspace: Path) -> ToolResult:
    target = _resolve(workspace, args.get("path", "."))
    if target is None:
        return ToolResult("", "path escapes workspace")
    if not target.is_dir():
        return ToolResult("", f"not a directory: {args.get('path', '.')}")
    lines = []
    for name in sorted(os.listdir(target)):
        marker = "d " if (target / name).is_dir() else "  "
        lines.append(marker + name)
    return ToolResult("\n".join(lines) or "(empty)")


def _read_file(args: Dict[str, Any], workspace: Path) -> ToolResult:
    target = _resolve(workspace, str(args.get("path", "")))
    if target is None:
        return ToolResult("", "path escapes workspace")
    if not target.is_file():
        return ToolResult("", f"not a file: {args.get('path', '')}")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult("", str(exc))

    lines = text.splitlines()
    total = len(lines)
    start = max(1, int(args.get("start_line", 1)))
    end = int(args.get("end_line", total)) if args.get("end_line") is not None else total
    end = max(start, min(end, total))
    header = f"# {target} (lines {start}-{end} of {total})\n"
    return ToolResult(header + _truncate("\n".join(lines[start - 1 : end])))


def _write_file(args: Dict[str, Any], workspace: Path) -> ToolResult:
    content = args.get("content", "")
    if not isinstance(content, str):
        return ToolResult("", "content must be a string")
    target = _resolve(workspace, str(args.get("path", "")))
    if target is None:
        return ToolResult("", "path escapes workspace")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult("", str(exc))
    return ToolResult(f"wrote {len(content)} chars to {target}")


def _search(args: Dict[str, Any], workspace: Path) -> ToolResult:
    pattern = args.get("pattern", "")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult("", "pattern is required")
    root = _resolve(workspace, str(args.get("path", ".")))
    if root is None:
        return ToolResult("", "path escapes workspace")

    # Prefer ripgrep for large repositories; retain the pure-Python fallback
    # for minimal hosts where rg is unavailable.
    rg = shutil.which("rg")
    if rg:
        command = [rg, "--fixed-strings", "--line-number", "--no-heading", "--color", "never"]
        for skipped in _SEARCH_SKIP:
            command.extend(["--glob", f"!{skipped}/**"])
        command.extend([pattern, str(root)])
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None:
            if proc.returncode not in (0, 1):
                return ToolResult("", proc.stderr.strip() or "search failed")
            lines = []
            for line in proc.stdout.splitlines()[:200]:
                try:
                    file_name, lineno, text = line.split(":", 2)
                    display = str(Path(file_name).resolve().relative_to(workspace))
                    lines.append(f"{display}:{lineno}:{text}")
                except ValueError:
                    lines.append(line)
            return ToolResult(_truncate("\n".join(lines)) if lines else "no matches")

    hits: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in _SEARCH_SKIP for part in rel.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern in line:
                hits.append(f"{p.relative_to(workspace)}:{lineno}: {line.rstrip()}")
                if len(hits) >= 200:
                    break
        if len(hits) >= 200:
            break
    return ToolResult(_truncate("\n".join(hits)) if hits else "no matches")


def _bash(args: Dict[str, Any], workspace: Path) -> ToolResult:
    command = args.get("command", "")
    if not isinstance(command, str) or not command.strip():
        return ToolResult("", "command is required")
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return ToolResult("", "command timed out after 60s")
    except OSError as exc:
        return ToolResult("", str(exc))

    out = stdout
    if stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + stderr
    if proc.returncode != 0:
        out += f"\n[exit {proc.returncode}]"
    return ToolResult(_truncate(out) if out else f"(no output, exit {proc.returncode})")


def _apply_patch(args: Dict[str, Any], workspace: Path) -> ToolResult:
    patch = args.get("patch", "")
    if not isinstance(patch, str) or not patch.strip():
        return ToolResult("", "patch is required")
    try:
        checked = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if checked.returncode != 0:
            return ToolResult("", f"patch rejected: {checked.stderr.strip()}")
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return ToolResult("", "patch timed out after 30s")
    except OSError as exc:
        return ToolResult("", str(exc))
    if applied.returncode != 0:
        return ToolResult("", f"patch failed: {applied.stderr.strip()}")
    return ToolResult("patch applied")


def _list_runs(args: Dict[str, Any], workspace: Path) -> ToolResult:
    runs = RunStore(workspace).list_runs()
    if not runs:
        return ToolResult("no past runs")
    lines = [
        f"{r['run_id']}\t{r['outcome']}\t{r['started_at']}\t{r['task']}"
        for r in runs
    ]
    return ToolResult(_truncate("\n".join(lines)))


def _read_run(args: Dict[str, Any], workspace: Path) -> ToolResult:
    record = RunStore(workspace).read_run(str(args.get("run_id", "")))
    if record is None:
        return ToolResult("", f"run not found: {args.get('run_id', '')}")
    return ToolResult(_truncate(json.dumps(record.to_dict(), ensure_ascii=False, indent=2)))


def _search_history(args: Dict[str, Any], workspace: Path) -> ToolResult:
    query = str(args.get("query", ""))
    if not query:
        return ToolResult("", "query is required")
    hits = RunStore(workspace).search(query)
    if not hits:
        return ToolResult("no matches")
    lines = [f"{h['run_id']}\t{h['outcome']}\t{h['task']}" for h in hits]
    return ToolResult(_truncate("\n".join(lines)))


def builtin_tools() -> List[Tool]:
    """Return the default coding-agent tool set."""
    return [
        Tool(
            "list_files",
            "List files and directories under a path.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, default '.'"},
                },
            },
            _list_files,
            True,
        ),
        Tool(
            "read_file",
            "Read a file, optionally a line range.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "1-based first line"},
                    "end_line": {"type": "integer", "description": "1-based last line"},
                },
                "required": ["path"],
            },
            _read_file,
            True,
        ),
        Tool(
            "write_file",
            "Create or overwrite a file inside the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
            _write_file,
        ),
        Tool(
            "search",
            "Search file contents for a literal string.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Literal substring"},
                    "path": {"type": "string", "description": "Directory to search, default '.'"},
                },
                "required": ["pattern"],
            },
            _search,
            True,
        ),
        Tool(
            "exec",
            "Run one process from a structured argv array without a shell. Host approval may be required.",
            {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Executable followed by its arguments; no shell syntax",
                    },
                    "reason": {"type": "string", "description": "Why this process is needed"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 600,
                        "description": "Process timeout, default 60 seconds",
                    },
                },
                "required": ["argv"],
            },
            lambda _args, _workspace: ToolResult("", "exec must be dispatched by ToolRegistry"),
        ),
        Tool(
            "bash",
            "Run a shell command with cwd set to the workspace.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
            _bash,
        ),
        Tool(
            "apply_patch",
            "Apply a unified git patch inside the workspace. Prefer this for targeted edits.",
            {
                "type": "object",
                "properties": {"patch": {"type": "string", "description": "Unified diff"}},
                "required": ["patch"],
            },
            _apply_patch,
        ),
        Tool(
            "list_runs",
            "List past runs in this workspace (id, outcome, task).",
            {"type": "object", "properties": {}},
            _list_runs,
            True,
        ),
        Tool(
            "read_run",
            "Read one past run's full record (trace, answer, usage).",
            {
                "type": "object",
                "properties": {"run_id": {"type": "string", "description": "Run id"}},
                "required": ["run_id"],
            },
            _read_run,
            True,
        ),
        Tool(
            "search_history",
            "Search past runs by a literal query (task/answer/trace).",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Literal substring"}},
                "required": ["query"],
            },
            _search_history,
            True,
        ),
    ]
