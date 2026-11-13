"""Session journal (append-only), compaction boundaries, and recovery paths."""

import json

import pytest

from pirl.agent import Agent
from pirl.checkpoint import CheckpointStore
from pirl.context import ContextManager
from pirl.llm import Completion, LLMError, ToolCall, Usage
from pirl.plugins.compaction import (
    DEFAULT_KEEP_RECENT_TOKENS,
    SummarizationError,
    default_summarizer,
    estimate_context_tokens,
    find_cut_point,
    prepare_compaction,
    should_compact,
)
from pirl.session import (
    SessionManager,
    SessionStore,
    default_session_path,
    entry_messages,
    strip_runtime_keys,
)
from pirl.tools import ToolRegistry, builtin_tools


class _ScriptedClient:
    def __init__(self, responses, model="test-model"):
        self.responses = list(responses)
        self.calls = []
        self.model = model
        self.base_url = "https://example.test/v1"

    def complete(self, messages, tools=None):
        self.calls.append((messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


# ---------------------------------------------------------------------------
# Journal structure
# ---------------------------------------------------------------------------


def test_journal_is_append_only_with_parent_chain(tmp_path):
    session = SessionManager.create(tmp_path / "s.jsonl", cwd=str(tmp_path))
    first = session.append_message({"role": "system", "content": "sys"})
    second = session.append_message({"role": "user", "content": "task"})

    assert first["parentId"] is None
    assert second["parentId"] == first["id"]
    assert [e["type"] for e in session.entries] == ["session", "message", "message"]

    reopened = SessionManager.open(tmp_path / "s.jsonl")
    assert reopened.build_context() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]


def test_projection_drops_messages_before_first_kept_entry(tmp_path):
    session = SessionManager(cwd=str(tmp_path))
    session.append_message({"role": "system", "content": "sys"})
    session.append_message({"role": "user", "content": "task"})
    kept = session.append_message({"role": "assistant", "content": "recent"})
    session.append_compaction("summary of old work", kept["id"], tokens_before=900)

    projected = session.build_context()
    assert projected[0] == {"role": "system", "content": "sys"}
    assert projected[1]["content"].startswith("[COMPACTED]")
    assert projected[1]["content"].endswith("summary of old work")
    assert projected[2:] == [{"role": "assistant", "content": "recent"}]
    # Original entries are still on disk / in the journal.
    assert len(session.entries) == 5


def test_repeated_compaction_keeps_earlier_retained_tail(tmp_path):
    session = SessionManager(cwd=str(tmp_path))
    session.append_message({"role": "system", "content": "sys"})
    session.append_message({"role": "user", "content": "task"})
    middle = session.append_message({"role": "assistant", "content": "m"})
    tail = session.append_message({"role": "assistant", "content": "t"})
    session.append_compaction("first summary", middle["id"])

    # Message view: [sys, note1, m, t]
    assert session.message_owners() == [session.entries[1]["id"], session.entries[5]["id"], middle["id"], tail["id"]]

    # A second pass over the retained tail supersedes the first summary.
    session.append_compaction("second summary", tail["id"])
    projected = session.build_context()
    assert projected[0]["content"] == "sys"
    assert projected[1]["content"].endswith("second summary")
    assert projected[2] == {"role": "assistant", "content": "t"}

    # Pointing a boundary at the previous compaction resolves to that
    # compaction's own boundary, so old notes never accumulate.
    session.append_compaction("third summary", session.message_owners()[1])
    projected = session.build_context()
    assert sum("[COMPACTED]" in message["content"] for message in projected) == 1
    assert projected[0]["content"] == "sys"


def test_context_edit_omits_and_replaces_without_mutating_journal(tmp_path):
    session = SessionManager(cwd=str(tmp_path))
    session.append_message({"role": "system", "content": "sys"})
    target = session.append_message({"role": "user", "content": "huge"})
    session.append_message({"role": "assistant", "content": "answer"})
    before = json.loads(json.dumps(session.entries))

    session.append_context_edit(target["id"], replacement=None)
    assert session.build_context() == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "answer"},
    ]

    replacement = session.append_context_edit(target["id"], {"content": "short"})
    assert session.build_context()[1] == {"role": "user", "content": "short"}
    assert replacement["type"] == "context_edit"
    # The targeted entry itself was never rewritten.
    assert session.entries[1] == before[1]


def test_session_file_jsonl_roundtrip_and_stats(tmp_path):
    path = default_session_path(tmp_path, "abc123")
    session = SessionManager.create(path, cwd=str(tmp_path))
    session.append_message({"role": "system", "content": "sys"})
    session.append_message({"role": "user", "content": "task"})
    session.append_usage("cache_warm", {"total_tokens": 10})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["type"] == "session"
    assert SessionManager.open(path).stats()["kinds"]["usage"] == 1


def test_runtime_only_keys_never_reach_the_provider(tmp_path):
    session = SessionManager(cwd=str(tmp_path))
    session.append_message({"role": "system", "content": "sys"})
    session.append_message({"role": "assistant", "content": "a"}, usage={"total_tokens": 42})

    journal = session.build_context()
    assert journal[-1]["_usage"] == {"total_tokens": 42}
    assert "_usage" not in strip_runtime_keys(journal)[-1]


# ---------------------------------------------------------------------------
# Deterministic accounting and cut points
# ---------------------------------------------------------------------------


def test_estimate_context_tokens_prefers_reported_usage():
    journal = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x" * 4000},
        {"role": "assistant", "content": "a", "_usage": {"total_tokens": 900}},
        {"role": "user", "content": "y" * 400},
    ]
    # 900 reported tokens + the trailing message, not the 1000-token heuristic.
    assert estimate_context_tokens(journal) == 900 + 100


def test_should_compact_uses_window_minus_reserve():
    assert not should_compact(100, 1000, reserve_tokens=200)
    assert should_compact(801, 1000, reserve_tokens=200)


def test_cut_point_never_starts_the_tail_at_a_tool_result():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u" * 4000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "tool", "tool_call_id": "c1", "content": "r2"},
    ]
    assert find_cut_point(messages, keep_recent=2) == 2
    preparation = prepare_compaction(messages, keep_recent=2)
    assert preparation is not None
    assert preparation.first_kept_index == 2
    assert messages[preparation.first_kept_index]["role"] == "assistant"


def test_prepare_compaction_refuses_a_tiny_middle():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    assert prepare_compaction(messages, keep_recent=1) is None


# ---------------------------------------------------------------------------
# Summary failures are explicit, never persisted
# ---------------------------------------------------------------------------


class _FailingSummarizerClient:
    def __init__(self, completion):
        self.completion = completion

    def complete(self, messages, tools=None):
        return self.completion


def test_length_stop_summary_is_rejected():
    summarize = default_summarizer(_FailingSummarizerClient(Completion("partial", [], stop_reason="length")))
    with pytest.raises(SummarizationError) as error:
        summarize("text")
    assert error.value.kind == "length"


def test_tool_calling_summarizer_is_rejected():
    summarize = default_summarizer(
        _FailingSummarizerClient(Completion("", [ToolCall("bash", "{}")], stop_reason="tool_calls"))
    )
    with pytest.raises(SummarizationError):
        summarize("text")


def test_compaction_view_records_boundary_and_never_mutates_input():
    calls = []
    messages = [{"role": "system", "content": "stable-prefix"}, {"role": "user", "content": "task"}]
    for index in range(4):
        messages.append({"role": "assistant", "content": f"step {index}: " + "x" * 400})
        messages.append({"role": "user", "content": "continue"})
    original = json.loads(json.dumps(messages))
    manager = ContextManager(
        lambda text: calls.append(text) or "## Goal\nship it",
        budget=200,
        keep_recent=2,
    )

    view = manager.prepare_view(messages, "task")

    assert messages == original
    assert calls
    assert view.messages[0] == messages[0]
    assert view.compaction is not None
    assert view.compaction.first_kept_index == len(messages) - 2
    assert view.compaction.tokens_before > 0
    assert view.messages[2:] == messages[view.compaction.first_kept_index:]
    # A file list block is appended when tool calls referenced paths.
    assert "<read-files>" not in view.messages[1]["content"]


def test_failed_summary_falls_back_to_a_recorded_bounded_tail():
    def broken(_text):
        raise RuntimeError("offline")

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 2000}]
    messages += _tool_exchange("c1", "read_file", {"path": "a.py"}, "y" * 2000)
    messages += [{"role": "assistant", "content": "recent"}, {"role": "user", "content": "now"}]

    view = ContextManager(broken, 200, keep_recent=2).prepare_view(messages, "task")

    assert view.compaction is not None
    assert view.compaction.kind == "fallback"
    assert view.messages[1]["content"].startswith("[COMPRESSED]")
    assert view.messages[-2:] == messages[-2:]


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def _registry(tmp_path):
    return ToolRegistry(tmp_path, builtin_tools())


def test_second_compaction_uses_the_previous_summary_as_context():
    prompts = []

    def summarize(text):
        prompts.append(text)
        return f"summary #{len(prompts)}"

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for index in range(6):
        messages.append({"role": "assistant", "content": f"step {index}: " + "x" * 400})
        messages.append({"role": "user", "content": "continue"})

    manager = ContextManager(summarize, budget=200, keep_recent=2)
    first = manager.prepare_view(messages, "task")
    assert "<previous-summary>" not in prompts[0]

    # Feed the compacted view back in as the new journal: the second pass must
    # merge into the first summary instead of re-summarizing from scratch.
    more = first.messages[:]
    for index in range(4):
        more.append({"role": "assistant", "content": f"more {index}: " + "y" * 400})
        more.append({"role": "user", "content": "again"})
    second = manager.prepare_view(more, "task")

    assert "<previous-summary>" in prompts[1]
    assert "summary #1" in prompts[1]
    assert second.compaction is not None
    assert second.compaction.previous_summary_used is True
    assert second.messages[1]["content"].endswith("summary #2")


def test_agent_persists_compaction_as_a_session_entry(tmp_path):
    (tmp_path / "a.txt").write_text("body")
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    big = "z" * 4000
    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c2")]),
        Completion("done", []),
    ])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=100,
        keep_recent=1,
        summarize=lambda _text: "## Goal\ndone",
    )

    result = agent.run("t" * 2000)

    assert result.outcome == "success"
    compactions = session.compactions()
    assert compactions, "expected the view decision to be journalled"
    assert compactions[0]["kind"] == "summary"
    assert compactions[0]["tokensBefore"] > 0
    assert "readFiles" in compactions[0]["details"]
    keep_id = compactions[0]["firstKeptEntryId"]
    assert any(entry.get("id") == keep_id for entry in session.entries)
    # The wire view never carries the journal-only usage key.
    assert all("_usage" not in message for message in client.calls[-1][0])


def test_unusable_summary_falls_back_and_says_so_in_the_journal(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    big = "z" * 4000
    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c2")]),
        Completion("done", []),
    ])

    def broken(_text):
        raise SummarizationError("generation hit the token cap", kind="length")

    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=100,
        keep_recent=1,
        summarize=broken,
    )

    assert agent.run("t" * 2000).outcome == "success"
    compactions = session.compactions()
    assert compactions and compactions[0]["kind"] == "fallback"
    assert session.build_context()[0]["role"] == "system"


def test_agent_never_sends_pre_compaction_history(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    client = _ScriptedClient([Completion("done", [])])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=10 ** 9,
    )
    agent.run("hello")
    assert client.calls[0][0][0]["role"] == "system"
    assert client.calls[0][0][-1] == {"role": "user", "content": "hello"}


def test_overflow_recovery_compacts_once_then_retries(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    big = "q" * 4000
    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c2")]),
        LLMError("API error 400: context length exceeded, reduce the length"),
        Completion("recovered", []),
    ])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=10 ** 9,
        keep_recent=1,
        summarize=lambda _text: "## Goal\nrecover",
    )

    result = agent.run("t" * 2000)

    assert result.outcome == "success"
    assert result.answer == "recovered"
    assert len(session.compactions()) == 1


def test_overflow_recovery_fails_after_one_attempt(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    big = "q" * 4000
    overflow = LLMError("maximum context length exceeded")
    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c2")]),
        overflow,
        overflow,
    ])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=10 ** 9,
        keep_recent=1,
        summarize=lambda _text: "## Goal\nrecover",
    )

    result = agent.run("t" * 2000)

    assert result.outcome == "failure"
    assert result.message.startswith("context overflow recovery failed")
    assert len(session.compactions()) == 1


def test_length_stop_triggers_recovery_instead_of_a_truncated_answer(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    big = "q" * 4000
    client = _ScriptedClient([
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c1")]),
        Completion("", [ToolCall("bash", json.dumps({"command": f"printf {big}"}), "c2")]),
        Completion("truncated", [], stop_reason="length"),
        Completion("complete answer", []),
    ])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        session=session,
        context_budget=10 ** 9,
        keep_recent=1,
        summarize=lambda _text: "## Goal\nrecover",
    )

    result = agent.run("t" * 2000)

    assert result.outcome == "success"
    assert result.answer == "complete answer"


def test_resume_replays_the_journal_instead_of_trusting_a_message_list(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    client = _ScriptedClient([Completion("first", []), Completion("second", [])])
    agent = Agent(client, tools=_registry(tmp_path), session=session)

    assert agent.run("one").answer == "first"
    assert len(client.calls) == 1
    assert agent.run("two").answer == "second"

    replayed = client.calls[-1][0]
    assert [message["role"] for message in replayed] == ["system", "user", "assistant", "user"]
    assert replayed[-1] == {"role": "user", "content": "two"}
    assert replayed[2] == {"role": "assistant", "content": "first"}


def test_model_change_is_recorded_for_replay(tmp_path):
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    agent = Agent(_ScriptedClient([Completion("ok", [])], model="model-a"), tools=_registry(tmp_path), session=session)
    agent.run("x")

    assert [e["type"] for e in session.entries].count("model_change") == 1
    assert session.last_model() == "model-a"


def test_checkpoint_store_reports_identity_mismatch(tmp_path):
    store = CheckpointStore(tmp_path)
    state = {"messages": [{"role": "system", "content": "s"}]}
    store.save("r1", state, identity={"cwd": "/a", "model": "m1"})

    loaded = store.load("r1")
    assert CheckpointStore.identity_mismatch(loaded, {"cwd": "/a", "model": "m1"}) == []
    assert CheckpointStore.identity_mismatch(loaded, {"cwd": "/b", "model": "m2"}) == ["cwd", "model"]


def test_entry_messages_ignores_context_invisible_entries(tmp_path):
    entry = {"type": "usage", "usage": {"total_tokens": 1}}
    assert entry_messages(entry) == []
    assert entry_messages({"type": "model_change", "modelId": "m"}) == []


def test_session_store_appends_without_rewriting(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append({"type": "session", "version": 3, "id": "x"})
    store.append({"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "hi"}})
    store.append({"type": "usage", "id": "b", "parentId": "a"})

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert len(store.load()) == 3
