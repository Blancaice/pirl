import json

from pirl.agent import Agent
from pirl.plugins.compaction import (
    compact,
    estimate_messages_tokens,
    estimate_tokens,
)
from pirl.llm import Completion, ToolCall
from pirl.tools import ToolRegistry, builtin_tools


def _msg(role, content):
    return {"role": role, "content": content}


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_messages_tokens():
    msgs = [_msg("system", "x" * 8), _msg("user", "y" * 8)]
    assert estimate_messages_tokens(msgs) == 4


def test_compact_replaces_middle_with_note():
    messages = [
        _msg("system", "persona"),
        _msg("user", "task"),
        _msg("assistant", "call 1"),
        _msg("tool", "x" * 500),
        _msg("assistant", "call 2"),
        _msg("tool", "y" * 500),
        _msg("assistant", "recent call"),
        _msg("tool", "recent result"),
    ]
    out = compact(lambda text: "summarized", messages, "task", keep_recent=2)
    assert out[0] == messages[0]  # system prefix untouched
    assert "[COMPACTED]" in out[1]["content"]
    assert out[1]["content"].endswith("summarized")
    assert out[2:] == messages[-2:]  # recent tail intact
    assert len(out) == 4


def test_compact_noop_when_too_short():
    messages = [_msg("system", "s"), _msg("user", "t"), _msg("assistant", "a"), _msg("tool", "r")]
    assert compact(lambda t: "never", messages, "t", keep_recent=2) == messages


def test_compact_noop_when_middle_tiny():
    messages = [
        _msg("system", "s"), _msg("user", "t"),
        _msg("assistant", "a"), _msg("tool", "tiny"),
        _msg("assistant", "a2"), _msg("tool", "recent"),
    ]
    assert compact(lambda t: "never", messages, "t", keep_recent=2) == messages


def test_compact_keeps_assistant_with_recent_tool_results():
    assistant = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}],
    }
    messages = [
        _msg("system", "s"),
        _msg("user", "x" * 1000),
        _msg("assistant", "old"),
        assistant,
        _msg("tool", "result one"),
        _msg("tool", "result two"),
    ]
    out = compact(lambda text: "summary", messages, "task", keep_recent=2)
    assert assistant in out
    first_tool = next(i for i, message in enumerate(out) if message["role"] == "tool")
    assert out[first_tool - 1] is assistant


def test_compaction_summary_includes_tool_calls():
    seen = []
    messages = [
        _msg("system", "s"),
        _msg("user", "x" * 1000),
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "write_file", "arguments": "{}"}}],
        },
        _msg("tool", "old result"),
        _msg("assistant", "recent"),
        _msg("user", "continue"),
    ]
    compact(lambda text: seen.append(text) or "summary", messages, "task", keep_recent=2)
    assert "write_file" in seen[0]


def test_agent_compacts_when_over_budget(tmp_path):
    calls = []

    def fake_summarize(text):
        calls.append(text)
        return "compacted summary"

    big = "python3 -c 'print(\"y\"*4000)'"
    from tests.test_agent import _ScriptedClient

    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": big}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": big}), "c2")]),
        Completion("done", []),
    ])
    agent = Agent(
        client,
        tools=ToolRegistry(tmp_path, builtin_tools()),
        context_budget=100,
        keep_recent=1,
        summarize=fake_summarize,
    )
    result = agent.run("x")
    assert result.outcome == "success"
    assert calls  # summarizer was invoked when the budget was exceeded
