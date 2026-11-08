"""Minimal agent runtime: turn loop over tools, skills, and self-generation.

The loop ends when the model answers with plain text (treated as success) or
calls the ``finish`` tool (``success`` true/false). Skill usage is tracked: a
``read_file`` whose path lands under the active skills dir counts as a
retrieval, and every retrieved skill is marked adopted with the run's outcome
— this is the "integration contract" that makes utility stats and periodic
retirement actually work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .lifecycle import SkillStore
from .llm import ChatClient, ToolSpec
from .loop import LoopResult, SelfGenLoop
from .prompt import format_skills_for_prompt
from .proposal import ToolCall
from .skills import Skill
from .tools import ToolRegistry, ToolResult

BASE_SYSTEM = (
    "You are a coding agent working in a local workspace. "
    "Use the available tools to complete the user's task. "
    "When you are done, call the finish tool with success=true and a summary. "
    "If the task cannot be completed, call finish with success=false."
)

FINISH_NAME = "finish"

FINISH_SPEC = ToolSpec(
    FINISH_NAME,
    "End the run. Call this when the task is done or cannot be completed.",
    {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the task was completed successfully.",
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of what was done or why it failed.",
            },
        },
        "required": ["success", "summary"],
    },
)

MAX_STEPS = 20


@dataclass
class RunResult:
    """What one agent run produced."""

    answer: str
    outcome: str  # "success" | "failure"
    trace: List[ToolCall] = field(default_factory=list)
    loop_result: Optional[LoopResult] = None
    message: str = ""


def _parse_args(raw: str):
    """Parse tool-call arguments; return an error string on failure."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return f"invalid tool arguments: {error}"
    return value if isinstance(value, dict) else "tool arguments must be a JSON object"


class Agent:
    """Runs one user task to completion (or until ``max_steps``)."""

    def __init__(
        self,
        client: ChatClient,
        *,
        skills: Sequence[Skill] = (),
        tools: ToolRegistry | None = None,
        self_gen: SelfGenLoop | None = None,
        skill_store: SkillStore | None = None,
        system_prompt: str = "",
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.client = client
        self.skills = list(skills)
        self.tools = tools
        self.self_gen = self_gen
        self.skill_store = skill_store
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self._retrieved: Set[str] = set()

    def _system_message(self) -> str:
        parts = [BASE_SYSTEM]
        if self.system_prompt:
            parts.append(self.system_prompt)
        skills_block = format_skills_for_prompt(self.skills, file_read_tool="read")
        if skills_block:
            parts.append(skills_block)
        return "\n".join(parts)

    def _tool_specs(self) -> List[ToolSpec]:
        specs = self.tools.specs() if self.tools else []
        return [*specs, FINISH_SPEC]

    def _skill_from_read(self, name: str, args: Dict[str, Any]) -> Optional[str]:
        """Return the active skill name a ``read_file`` call targets, if any."""
        if name != "read_file" or self.skill_store is None:
            return None
        raw = str(args.get("path", ""))
        if not raw:
            return None
        try:
            resolved = Path(raw).expanduser().resolve()
            rel = resolved.relative_to(self.skill_store.active_dir.resolve())
        except (OSError, ValueError):
            return None
        parts = rel.parts
        return parts[0] if parts else None

    def run(self, user_message: str) -> RunResult:
        self._retrieved.clear()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": user_message},
        ]
        tool_specs = self._tool_specs()
        trace: List[ToolCall] = []

        for _ in range(self.max_steps):
            completion = self.client.complete(messages, tools=tool_specs)
            if not completion.tool_calls:
                return self._close(trace, "success", completion.text)

            call_ids = [
                tc.id or f"call_{len(trace)}_{i}"
                for i, tc in enumerate(completion.tool_calls)
            ]
            assistant: Dict[str, Any] = {"role": "assistant"}
            if completion.text:
                assistant["content"] = completion.text
            assistant["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc, call_id in zip(completion.tool_calls, call_ids)
            ]
            messages.append(assistant)

            for tc, call_id in zip(completion.tool_calls, call_ids):
                if tc.name == FINISH_NAME:
                    args = _parse_args(tc.arguments)
                    if isinstance(args, str):
                        return self._close(
                            trace, "failure", "", f"invalid finish args: {args}"
                        )
                    success = bool(args.get("success", True))
                    summary = str(args.get("summary", ""))
                    return self._close(
                        trace, "success" if success else "failure", summary
                    )

                trace.append(ToolCall(tc.name, tc.arguments))
                args = _parse_args(tc.arguments)
                if isinstance(args, str):
                    result = ToolResult("", args)
                elif self.tools is None:
                    result = ToolResult("", "no tools registered")
                else:
                    result = self.tools.execute(tc.name, args)

                skill_name = self._skill_from_read(
                    tc.name, args if isinstance(args, dict) else {}
                )
                if skill_name and skill_name not in self._retrieved:
                    self._retrieved.add(skill_name)
                    if self.skill_store is not None:
                        self.skill_store.on_retrieve(skill_name)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result.error or result.content,
                    }
                )

        return self._close(trace, "failure", "", "max steps exceeded")

    def _close(
        self, trace: List[ToolCall], outcome: str, answer: str, message: str = ""
    ) -> RunResult:
        if self.skill_store is not None:
            for name in self._retrieved:
                self.skill_store.on_adopt(name, outcome == "success")
        loop_result = None
        if self.self_gen is not None:
            loop_result = self.self_gen.on_run_end(trace, outcome)
        return RunResult(answer, outcome, trace, loop_result, message)
