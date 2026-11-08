"""Minimal agent runtime: turn loop over tools, skills, and self-generation.

This is the slice past the design's Part D: it wires Part A (skills in the
system prompt) and Part B (``SelfGenLoop`` at run end) around a real model
client and a workspace-confined tool registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .llm import ChatClient
from .loop import LoopResult, SelfGenLoop
from .prompt import format_skills_for_prompt
from .proposal import ToolCall
from .skills import Skill
from .tools import ToolRegistry, ToolResult

BASE_SYSTEM = (
    "You are a coding agent working in a local workspace. "
    "Use the available tools to complete the user's task. "
    "When the task is done, stop calling tools and give a concise final answer."
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
        system_prompt: str = "",
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.client = client
        self.skills = list(skills)
        self.tools = tools
        self.self_gen = self_gen
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def _system_message(self) -> str:
        parts = [BASE_SYSTEM]
        if self.system_prompt:
            parts.append(self.system_prompt)
        skills_block = format_skills_for_prompt(self.skills, file_read_tool="read")
        if skills_block:
            parts.append(skills_block)
        return "\n".join(parts)

    def run(self, user_message: str) -> RunResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": user_message},
        ]
        tool_specs = self.tools.specs() if self.tools else []
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
                trace.append(ToolCall(tc.name, tc.arguments))
                args = _parse_args(tc.arguments)
                if isinstance(args, str):
                    result = ToolResult("", args)
                elif self.tools is None:
                    result = ToolResult("", "no tools registered")
                else:
                    result = self.tools.execute(tc.name, args)
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
        loop_result = None
        if self.self_gen is not None:
            loop_result = self.self_gen.on_run_end(trace, outcome)
        return RunResult(answer, outcome, trace, loop_result, message)
