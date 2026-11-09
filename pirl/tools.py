"""Built-in workspace tools and the registry the agent dispatches through.

Tools run with the process's OS permissions, confined to ``workspace``: file
tools reject paths that resolve outside the workspace. ``bash`` runs with
``cwd=workspace``. Tool output is truncated to keep model context bounded.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import json

from .llm import ToolSpec
from .run_history import RunStore

MAX_OUTPUT = 20_000
_SEARCH_SKIP = frozenset({".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"})


@dataclass(frozen=True)
class ToolResult:
    """The text a tool returns to the model; ``error`` marks a failure."""

    content: str
    error: str = ""


@dataclass(frozen=True)
class Tool:
    """A model-facing tool: JSON-Schema declaration plus an execute function."""

    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[[Dict[str, Any], Path], ToolResult]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    """Name -> Tool lookup; resolves paths against a fixed workspace."""

    def __init__(self, workspace: str | Path, tools: Sequence[Tool] = ()) -> None:
        self.workspace = Path(workspace).resolve()
        self._tools: Dict[str, Tool] = {t.name: t for t in tools}

    def specs(self) -> List[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("", f"unknown tool: {name}")
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
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return ToolResult("", "command timed out after 60s")
    except OSError as exc:
        return ToolResult("", str(exc))

    out = proc.stdout
    if proc.stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + proc.stderr
    if proc.returncode != 0:
        out += f"\n[exit {proc.returncode}]"
    return ToolResult(_truncate(out) if out else f"(no output, exit {proc.returncode})")


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
            "list_runs",
            "List past runs in this workspace (id, outcome, task).",
            {"type": "object", "properties": {}},
            _list_runs,
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
        ),
    ]
