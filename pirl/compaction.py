"""Context compaction: threshold-triggered summarization of the old trajectory.

Grounded in the context-compression playbook and pi's compaction: compress the
*trajectory* (old tool results), never the stable system-prompt prefix. The
summary must preserve decisions, constraints, failures, and concrete facts; it
is marked ``[COMPACTED]`` so it is never re-compressed.

Trigger policy (pi-style): estimate the running message list's tokens and only
compact once it exceeds a budget, keeping the most recent ``keep_recent``
messages intact — so the prefix stays cache-friendly and compaction is not done
every turn.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

COMPACT_SYSTEM = (
    "You summarize an agent's earlier tool results into compact context for later turns. "
    "Keep: decisions made, constraints discovered, failures and their causes, "
    "and concrete facts (file names, command outputs, values). "
    "Drop redundant noise and repeated attempts. Be concise."
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars // 4), matching pi's estimateTokens heuristic."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(estimate_tokens(_message_text(m)) for m in messages)


def _message_text(message: Dict[str, Any]) -> str:
    """Render all context-bearing fields, including assistant tool calls."""
    parts = [str(message.get("content", ""))]
    if message.get("tool_calls"):
        parts.append(str(message["tool_calls"]))
    return "\n".join(parts)


def render_messages(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = _message_text(m)
        lines.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(lines)


def compact(
    summarize: Callable[[str], str],
    messages: List[Dict[str, Any]],
    task: str,
    *,
    keep_recent: int = 6,
) -> List[Dict[str, Any]]:
    """Replace the old middle of ``messages`` with one ``[COMPACTED]`` summary note.

    ``messages[0]`` is the stable system prefix (kept untouched); the last
    ``keep_recent`` messages stay intact; everything between is summarized.
    Returns ``messages`` unchanged when there is too little to compress.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    boundary = len(messages) - keep_recent
    # Tool results are protocol-coupled to the preceding assistant tool_calls
    # message. Never keep an orphaned tool result after compaction.
    while boundary > 1 and messages[boundary].get("role") == "tool":
        boundary -= 1

    middle = messages[1:boundary]
    tail = messages[boundary:]

    # Skip when the middle is tiny — summarizing noise just wastes tokens.
    if estimate_messages_tokens(middle) < 200:
        return messages

    summary = summarize(render_messages(middle))
    note = {"role": "user", "content": f"[COMPACTED] earlier progress:\n{summary}"}
    return [messages[0], note, *tail]


def default_summarizer(client: Any) -> Callable[[str], str]:
    """Build a summarizer that calls ``client.complete`` once per compaction."""

    def summarize(text: str) -> str:
        completion = client.complete(
            [
                {"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": text},
            ]
        )
        return completion.text.strip()

    return summarize
