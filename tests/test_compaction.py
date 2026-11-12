import json

from pirl.agent import Agent
from pirl.context import ContextManager
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


def test_agent_bounds_wire_context_when_over_budget(tmp_path):
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
    # Deterministic externalization may bring the request under control before
    # the expensive summarizer is needed; either route must bound the wire view.
    assert calls or any(
        "[COMPRESSED]" in str(message.get("content", ""))
        for request, _ in client.calls[1:]
        for message in request
    )


def _tool_exchange(call_id, name, args, content):
    return [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def test_context_deduplicates_unchanged_file_without_mutating_journal(tmp_path):
    result = json.dumps({"ok": True, "content": "same source", "error": "", "source": "tool:read_file"})
    messages = [_msg("system", "s"), _msg("user", "task")]
    messages += _tool_exchange("c1", "read_file", {"path": "calc.py"}, result)
    messages += _tool_exchange("c2", "read_file", {"path": "calc.py"}, result)
    original = json.loads(json.dumps(messages))

    manager = ContextManager(lambda _: "unused", None, object_dir=tmp_path / "objects")
    wire = manager.prepare(messages, "task")

    assert messages == original
    assert wire[3]["content"].startswith("[COMPRESSED]")
    assert wire[5]["content"] == result
    assert len(list((tmp_path / "objects").glob("*.txt"))) == 1


def test_context_keeps_changed_file_versions_distinct(tmp_path):
    first = json.dumps({"ok": True, "content": "version one", "error": ""})
    second = json.dumps({"ok": True, "content": "version two", "error": ""})
    messages = [_msg("system", "s"), _msg("user", "task")]
    messages += _tool_exchange("c1", "read_file", {"path": "a.py"}, first)
    messages += _tool_exchange("c2", "read_file", {"path": "a.py"}, second)

    wire = ContextManager(lambda _: "unused", None, object_dir=tmp_path).prepare(messages, "task")

    assert wire[3]["content"] == first
    assert wire[5]["content"] == second


def test_context_externalizes_large_result_with_preview(tmp_path):
    body = "begin-" + "x" * 1000 + "-end"
    result = json.dumps({"ok": True, "content": body, "error": ""})
    messages = [_msg("system", "s"), _msg("user", "task")]
    messages += _tool_exchange("c1", "read_file", {"path": "large.py"}, result)

    wire = ContextManager(
        lambda _: "unused", None, object_dir=tmp_path, max_inline_chars=256
    ).prepare(messages, "task")

    assert wire[3]["content"].startswith("[COMPRESSED]")
    assert "begin-" in wire[3]["content"] and "-end" in wire[3]["content"]
    assert any(path.read_text() == body for path in tmp_path.glob("*.txt"))


def test_context_deduplicates_identical_command_output_without_skipping_execution(tmp_path):
    messages = [_msg("system", "s"), _msg("user", "task")]
    messages += _tool_exchange("c1", "exec", {"argv": ["pytest", "-q"]}, "4 passed")
    messages += _tool_exchange("c2", "exec", {"argv": ["pytest", "-q"]}, "4 passed")

    wire = ContextManager(lambda _: "unused", None, object_dir=tmp_path).prepare(messages, "task")

    assert wire[3]["content"].startswith("[COMPRESSED]")
    assert wire[5]["content"] == "4 passed"
    # Both calls stay in the protocol; this is context caching, not unsafe
    # reuse of a potentially stale command result.
    assert sum(len(m.get("tool_calls", [])) for m in wire) == 2


def test_context_archives_old_write_body(tmp_path):
    messages = [_msg("system", "s"), _msg("user", "task")]
    messages += _tool_exchange(
        "c1", "write_file", {"path": "a.py", "content": "valuable body"}, "ok"
    )
    messages += [_msg("assistant", "later"), _msg("user", "continue")]

    wire = ContextManager(
        lambda _: "unused", None, keep_recent=2, object_dir=tmp_path
    ).prepare(messages, "task")
    args = json.loads(wire[2]["tool_calls"][0]["function"]["arguments"])

    assert args["content"].startswith("[COMPRESSED object=sha256:")
    assert any(path.read_text() == "valuable body" for path in tmp_path.glob("*.txt"))


def test_context_summary_failure_uses_bounded_fallback_and_circuit_breaker():
    attempts = []
    messages = [_msg("system", "s"), _msg("user", "x" * 1000)]
    messages += _tool_exchange("c1", "read_file", {"path": "a"}, "y" * 1000)
    messages += [_msg("assistant", "recent"), _msg("user", "continue")]

    def broken(text):
        attempts.append(text)
        raise RuntimeError("offline")

    manager = ContextManager(broken, 100, keep_recent=2, max_summary_failures=2)
    first = manager.prepare(messages, "task")
    second = manager.prepare(messages, "task")
    third = manager.prepare(messages, "task")

    assert len(attempts) == 2
    assert first[1]["content"].startswith("[COMPRESSED]")
    assert second == third


def test_context_uses_llm_summary_only_above_high_water():
    seen = []
    messages = [_msg("system", "stable-prefix"), _msg("user", "task")]
    for index in range(5):
        messages += [_msg("assistant", f"decision {index}: " + "x" * 300), _msg("user", "continue")]
    messages += [_msg("assistant", "recent"), _msg("user", "now")]

    wire = ContextManager(
        lambda text: seen.append(text) or "decisions and failures retained",
        budget=200,
        keep_recent=2,
    ).prepare(messages, "task")

    assert seen
    assert wire[0] == messages[0]
    assert wire[1]["content"].startswith("[COMPACTED]")
    assert wire[-2:] == messages[-2:]


def test_agent_checkpoint_keeps_full_journal_while_wire_is_reduced(tmp_path):
    (tmp_path / "a.txt").write_text("same")
    checkpoints = []
    from tests.test_agent import _ScriptedClient
    client = _ScriptedClient([
        Completion("", [ToolCall("read_file", '{"path":"a.txt"}', "c1")]),
        Completion("", [ToolCall("read_file", '{"path":"a.txt"}', "c2")]),
        Completion("done", []),
    ])
    agent = Agent(
        client,
        tools=ToolRegistry(tmp_path, builtin_tools()),
        context_budget=10000,
        checkpoint=checkpoints.append,
    )

    assert agent.run("read twice").outcome == "success"
    full_results = [m for m in checkpoints[-1]["messages"] if m["role"] == "tool"]
    wire_results = [m for m in client.calls[-1][0] if m["role"] == "tool"]
    assert all(not m["content"].startswith("[COMPRESSED]") for m in full_results)
    assert wire_results[0]["content"].startswith("[COMPRESSED]")
    assert not wire_results[1]["content"].startswith("[COMPRESSED]")
