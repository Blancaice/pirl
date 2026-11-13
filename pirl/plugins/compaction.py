"""Context compaction: deterministic cut points, structured summaries.

Mirrors pi's compaction contract:

* **Deterministic** — the trigger, the token accounting, and the cut point are
  pure functions of the message list. Only the summary *text* comes from an
  LLM, so the boundary can always be reproduced and audited.
* **Safe failure** — a summary that cannot be persisted (provider error, a
  ``length`` stop with partial text, a tool call, an empty body) raises
  :class:`SummarizationError` instead of silently becoming a checkpoint.
* **Protocol-safe cut** — never start the kept tail at a tool result; tool
  results must stay with the assistant ``tool_calls`` that produced them.

The summarized span is rendered as plain text (``<role>`` blocks) so the model
treats it as material to summarize rather than a conversation to continue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

COMPACTED_PREFIX = "[COMPACTED]"
DEFAULT_RESERVE_TOKENS = 16384
DEFAULT_KEEP_RECENT_TOKENS = 20000
DEFAULT_KEEP_RECENT = 6
MIN_MIDDLE_TOKENS = 200
TOOL_RESULT_TRUNCATE_CHARS = 2000

SUMMARY_SYSTEM = (
    "You summarize an agent's earlier work into a structured context checkpoint "
    "that another LLM will use to continue the task. Keep decisions made, "
    "constraints discovered, failures and their causes, and concrete facts "
    "(file names, function names, command output, values). Drop redundant noise "
    "and repeated attempts. Never invent information that is not present."
)

_INITIAL_FORMAT = """## Goal
[What is the user trying to accomplish? Multiple items if the session covers several tasks.]

## Constraints & Preferences
- [Constraints, preferences, or requirements mentioned by the user]
- [(none) if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, or references needed to continue]
- [(none) if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_FORMAT = """RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Previously done items AND newly completed items]

### In Progress
- [ ] [Current work]

### Blocked
- [Remove resolved blockers]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


class SummarizationError(Exception):
    """A summary that must not be persisted as a checkpoint."""

    def __init__(self, message: str, *, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars // 4), same heuristic pi uses."""
    return max(1, len(str(text)) // 4)


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    total = 0
    for block in content:
        if isinstance(block, Mapping):
            kind = block.get("type")
            if kind == "text":
                total += len(str(block.get("text", "")))
            elif kind == "image":
                total += 4800  # coarse per-image estimate
            else:
                total += len(json.dumps(block, ensure_ascii=False))
        else:
            total += len(str(block))
    return total


def _message_text(message: Mapping[str, Any]) -> str:
    """All context-bearing fields of a message, for accounting and rendering."""
    parts = [str(message.get("content", ""))]
    if message.get("reasoning"):
        parts.append(str(message["reasoning"]))
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def estimate_message_tokens(message: Mapping[str, Any]) -> int:
    return estimate_tokens(_message_text(message))


def estimate_messages_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _reported_usage(message: Mapping[str, Any]) -> int:
    for key in ("_usage", "usage"):
        value = message.get(key)
        if isinstance(value, Mapping):
            total = int(value.get("total_tokens") or 0)
            if total > 0:
                return total
    return 0


def estimate_context_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Prefer real provider usage over the chars/4 heuristic.

    Walks back to the newest message that carries reported usage and only
    estimates the trailing messages, which is what actually drifted.
    """
    for index in range(len(messages) - 1, -1, -1):
        reported = _reported_usage(messages[index])
        if reported:
            return reported + estimate_messages_tokens(messages[index + 1:])
    return estimate_messages_tokens(messages)


def should_compact(
    context_tokens: int,
    context_window: int,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> bool:
    if context_window <= 0:
        return False
    return context_tokens > context_window - reserve_tokens


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------

CUT_ROLES = frozenset({"user", "assistant", "custom"})


@dataclass(frozen=True)
class CompactionPreparation:
    """A deterministic plan: what to summarize and where the tail starts."""

    first_kept_index: int
    messages_to_summarize: List[Dict[str, Any]]
    turn_prefix_messages: List[Dict[str, Any]]
    is_split_turn: bool
    tokens_before: int


def find_cut_point(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    keep_recent_tokens: Optional[int] = None,
) -> int:
    """Index of the first kept message, or ``0`` when nothing can be cut.

    Walks backwards accumulating size until the recent budget is reached, then
    moves the boundary off any tool result so tool calls keep their results.
    """
    total = len(messages)
    if total <= 1:
        return 0

    if keep_recent_tokens:
        boundary = 1
        accumulated = 0
        for index in range(total - 1, 0, -1):
            accumulated += estimate_message_tokens(messages[index])
            if accumulated >= keep_recent_tokens:
                boundary = index
                break
    else:
        boundary = max(1, total - max(1, keep_recent))

    while boundary > 1 and messages[boundary].get("role") == "tool":
        boundary -= 1

    if boundary <= 1:
        return 0
    if messages[boundary].get("role") not in CUT_ROLES:
        return 0
    return boundary


def prepare_compaction(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    keep_recent_tokens: Optional[int] = None,
) -> Optional[CompactionPreparation]:
    """Build a compaction plan, or ``None`` when there is too little to cut."""
    if len(messages) <= 1:
        return None

    boundary = find_cut_point(
        messages, keep_recent=keep_recent, keep_recent_tokens=keep_recent_tokens
    )
    if boundary <= 1:
        return None

    middle = [dict(message) for message in messages[1:boundary]]
    if estimate_messages_tokens(middle) < MIN_MIDDLE_TOKENS:
        return None

    return CompactionPreparation(
        first_kept_index=boundary,
        messages_to_summarize=middle,
        turn_prefix_messages=[],
        is_split_turn=False,
        tokens_before=estimate_context_tokens(messages),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize a span as plain text so the model does not continue it."""
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "tool" and len(content) > TOOL_RESULT_TRUNCATE_CHARS:
            dropped = len(content) - TOOL_RESULT_TRUNCATE_CHARS
            content = content[:TOOL_RESULT_TRUNCATE_CHARS] + f"\n...[truncated {dropped} chars]"
        lines.append(f"<{role}>\n{content}\n</{role}>")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(
                "<tool_call>"
                f"{function.get('name', '')}"
                f"({function.get('arguments', '')})"
                "</tool_call>"
            )
    return "\n".join(lines)


def build_summary_prompt(
    conversation_text: str,
    previous_summary: str = "",
    custom_instructions: str = "",
) -> str:
    """Assemble the summarization request body (format rules included)."""
    parts = [f"<conversation>\n{conversation_text}\n</conversation>"]
    if previous_summary.strip():
        parts.append(f"<previous-summary>\n{previous_summary.strip()}\n</previous-summary>")
        parts.append(
            "The messages above are NEW conversation messages to incorporate into "
            "the existing summary provided in <previous-summary> tags.\n\n"
            + _UPDATE_FORMAT
        )
    else:
        parts.append(
            "The messages above are a conversation to summarize. Create a structured "
            "context checkpoint summary that another LLM will use to continue the "
            "work.\n\nUse this EXACT format:\n\n" + _INITIAL_FORMAT
        )
    if custom_instructions.strip():
        parts.append(f"Additional focus: {custom_instructions.strip()}")
    return "\n\n".join(parts)


def file_operations(messages: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    """Cumulative read/modified file lists extracted from tool calls."""
    read: List[str] = []
    modified: List[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            arguments = function.get("arguments", "")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except (TypeError, ValueError):
                    arguments = {}
            if not isinstance(arguments, Mapping):
                continue
            path = str(arguments.get("path", "")).strip()
            if not path:
                continue
            if name in ("write_file", "apply_patch", "edit_file"):
                if path not in modified:
                    modified.append(path)
            elif name in ("read_file", "read"):
                if path not in read:
                    read.append(path)
    return read, modified


def append_file_lists(summary: str, read: Sequence[str], modified: Sequence[str]) -> str:
    body = summary.rstrip()
    if read:
        body += "\n\n<read-files>\n" + "\n".join(read) + "\n</read-files>"
    if modified:
        body += "\n\n<modified-files>\n" + "\n".join(modified) + "\n</modified-files>"
    return body


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


def summarization_failure(completion: Any, label: str = "Summarization") -> Optional[str]:
    """Reject summaries that must not become a checkpoint."""
    stop_reason = str(getattr(completion, "stop_reason", "") or "")
    if stop_reason == "error":
        return f"{label} failed: provider reported an error"
    if stop_reason == "length":
        return f"{label} failed: generation hit the token cap and the summary is incomplete"
    if getattr(completion, "tool_calls", None):
        return f"{label} failed: the summarizer attempted to call a tool"
    if not str(getattr(completion, "text", "") or "").strip():
        return f"{label} failed: the summarizer returned an empty summary"
    return None


def default_summarizer(client: Any) -> Callable[[str], str]:
    """Build a summarizer that calls ``client.complete`` once per compaction.

    Summarization requests deliberately drop prompt-cache hints: a one-off
    summary is unlikely to be reused, so writing it into a cache is waste.
    """

    def summarize(text: str) -> str:
        completion = client.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": text},
            ]
        )
        failure = summarization_failure(completion)
        if failure:
            kind = "length" if getattr(completion, "stop_reason", "") == "length" else "error"
            raise SummarizationError(failure, kind=kind)
        return str(completion.text).strip()

    return summarize


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def compact(
    summarize: Callable[[str], str],
    messages: List[Dict[str, Any]],
    task: str,
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    keep_recent_tokens: Optional[int] = None,
    previous_summary: str = "",
) -> List[Dict[str, Any]]:
    """Replace the older middle of ``messages`` with one ``[COMPACTED]`` note.

    ``messages[0]`` is the stable system prefix (kept untouched); the tail from
    ``first_kept_index`` stays intact. Returns ``messages`` unchanged when there
    is too little to compress.
    """
    preparation = prepare_compaction(
        messages, keep_recent=keep_recent, keep_recent_tokens=keep_recent_tokens
    )
    if preparation is None:
        return messages

    prompt = build_summary_prompt(
        render_messages(preparation.messages_to_summarize), previous_summary
    )
    summary = str(summarize(prompt)).strip()
    read_files, modified_files = file_operations(preparation.messages_to_summarize)
    summary = append_file_lists(summary, read_files, modified_files)

    note = {
        "role": "user",
        "content": f"{COMPACTED_PREFIX} earlier progress (kept tail follows):\n{summary}",
    }
    return [messages[0], note, *messages[preparation.first_kept_index:]]
