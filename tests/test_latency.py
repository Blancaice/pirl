"""Latency work: stable wire prefix, cheaper progress checks, streaming."""

import json

import pytest

from pirl.agent import Agent
from pirl.context import ContextManager
from pirl.llm import ChatClient, Completion, ToolCall, Usage, _consume_stream
from pirl.session import SessionManager
from pirl.tools import ToolRegistry, builtin_tools


class _ScriptedClient:
    def __init__(self, responses, model="m", base_url="u"):
        self.responses = list(responses)
        self.calls = []
        self.model = model
        self.base_url = base_url

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _registry(tmp_path):
    return ToolRegistry(tmp_path, builtin_tools())


def _tc(name, args, call_id):
    return ToolCall(name, json.dumps(args), call_id)


# ---------------------------------------------------------------------------
# P0-1: the workspace fingerprint is not computed on read-only turns
# ---------------------------------------------------------------------------


def test_fingerprint_is_skipped_on_read_only_turns(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("body")
    calls = []
    agent = Agent(
        _ScriptedClient([
            Completion("", [_tc("read_file", {"path": "a.txt"}, "c1")]),
            Completion("", [_tc("list_files", {"path": "."}, "c2")]),
            Completion("done", []),
        ]),
        tools=_registry(tmp_path),
    )
    monkeypatch.setattr(agent, "_workspace_fingerprint", lambda: calls.append(1) or "fp")

    assert agent.run("read").outcome == "success"
    assert calls == []


def test_fingerprint_runs_only_around_a_mutation(tmp_path, monkeypatch):
    calls = []
    agent = Agent(
        _ScriptedClient([
            Completion("", [_tc("write_file", {"path": "new.txt", "content": "x"}, "c1")]),
            Completion("done", []),
        ]),
        tools=_registry(tmp_path),
    )
    monkeypatch.setattr(agent, "_workspace_fingerprint", lambda: calls.append(1) or "fp")

    assert agent.run("write").outcome == "success"
    # once before the tool batch, once afterwards — not once per turn start
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# P0-2: reduction decisions are frozen, so the wire prefix stays byte-stable
# ---------------------------------------------------------------------------


def test_wire_prefix_never_changes_after_a_message_is_sent(tmp_path):
    """Stability by construction: no message is rewritten on a later turn.

    Collapsing a repeated observation would rewrite an *older* message, which
    invalidates the provider's prompt cache from that offset onwards. The default
    path therefore leaves history byte-identical and bounds context with
    externalization (content-determined) plus compaction.
    """
    (tmp_path / "a.txt").write_text("same body")
    session = SessionManager(tmp_path / "s.jsonl", cwd=str(tmp_path))
    client = _ScriptedClient([
        Completion("", [_tc("read_file", {"path": "a.txt"}, "c1")]),
        Completion("", [_tc("list_files", {"path": "."}, "c2")]),
        Completion("", [_tc("read_file", {"path": "a.txt"}, "c3")]),
        Completion("", [_tc("list_files", {"path": "."}, "c4")]),
        Completion("done", []),
    ])
    agent = Agent(client, tools=_registry(tmp_path), session=session, keep_recent=2)

    assert agent.run("read twice").outcome == "success"

    renders: dict[int, list] = {}
    for request in client.calls:
        for index, message in enumerate(request):
            renders.setdefault(index, []).append(message.get("content"))
    for index, values in renders.items():
        changes = sum(1 for left, right in zip(values, values[1:]) if left != right)
        assert changes == 0, f"message {index} was rewritten (prefix not stable)"

    # No silent journal rewrites either: the reduction required none.
    assert not [e for e in session.entries if e["type"] == "context_edit"]


def test_large_observation_rendering_is_stable_across_turns(tmp_path):
    (tmp_path / "big.txt").write_text("chunk " * 1500)
    manager = ContextManager(lambda _t: "S", None, object_dir=tmp_path / "obj")
    journal = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
    ]
    journal.append({
        "role": "assistant",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "big.txt"})}}],
    })
    journal.append({"role": "tool", "tool_call_id": "c1", "content": "X" * 6000})

    first = manager.prepare(journal, "task")
    journal.append({
        "role": "assistant",
        "tool_calls": [{"id": "c2", "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "big.txt"})}}],
    })
    journal.append({"role": "tool", "tool_call_id": "c2", "content": "Y" * 6000})
    second = manager.prepare(journal, "task")

    assert first[3]["content"].startswith("[COMPRESSED]")
    assert second[3] == first[3], "externalized rendering must not drift"


def test_already_compressed_content_is_never_recompressed():
    manager = ContextManager(lambda _t: "S", None)
    frozen = {"role": "tool", "tool_call_id": "c1", "content": '[COMPRESSED] {"compressed": true}'}
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        frozen,
    ]
    wire = manager._reduce_observations(messages)
    assert wire[3]["content"] == frozen["content"]


# ---------------------------------------------------------------------------
# P1-4: true SSE streaming
# ---------------------------------------------------------------------------

SSE = [
    b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n',
    b'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}\n',
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.py\\"}"}}]}}]}\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n',
    b"data: [DONE]\n",
]


def test_consume_stream_assembles_text_reasoning_and_tool_calls():
    seen_text, seen_reasoning = [], []
    completion = _consume_stream(
        SSE,
        on_delta=seen_text.append,
        on_reasoning=seen_reasoning.append,
    )

    assert completion.text == "hello"
    assert completion.reasoning == "thinking..."
    assert completion.stop_reason == "tool_calls"
    assert completion.usage == Usage(10, 4, 14)
    assert completion.tool_calls == [ToolCall("read_file", '{"path":"a.py"}', "c1")]
    assert seen_text == ["hel", "lo"]
    assert seen_reasoning == ["thinking..."]


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_chat_client_stream_requests_sse_and_forwards_deltas(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _FakeResponse(SSE)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ChatClient(base_url="https://example.test", api_key="k", model="m")
    deltas, thoughts = [], []

    completion = client.stream(
        [{"role": "user", "content": "hi"}],
        tools=(),
        on_delta=deltas.append,
        on_reasoning=thoughts.append,
    )

    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert completion.text == "hello"
    assert deltas == ["hel", "lo"]
    assert thoughts == ["thinking..."]


def test_agent_streaming_uses_on_reasoning_and_still_logs_it(tmp_path):
    thoughts = []

    class StreamingClient:
        model = "m"
        base_url = "u"

        def complete(self, messages, tools=None):  # pragma: no cover - unused
            raise AssertionError("complete() must not be used when stream=True")

        def stream(self, messages, *, tools=(), on_delta=None, on_reasoning=None):
            if on_reasoning:
                on_reasoning("step 1 thought")
            return Completion("done", [], reasoning="step 1 thought")

    agent = Agent(
        StreamingClient(),
        tools=_registry(tmp_path),
        stream=True,
        on_reasoning=thoughts.append,
    )

    result = agent.run("x")

    assert thoughts == ["step 1 thought"]
    assert result.reasoning == ["step 1 thought"]


# ---------------------------------------------------------------------------
# P1-5: a separate summarizer client
# ---------------------------------------------------------------------------


def test_summarizer_client_is_used_for_compaction(tmp_path):
    summary_calls = []

    class Summarizer:
        def complete(self, messages, tools=None):
            summary_calls.append(messages)
            return Completion("## Goal\nsummarized", [])

    big = "z" * 3000
    client = _ScriptedClient([
        Completion("", [_tc("write_file", {"path": "a.txt", "content": big}, "c1")]),
        Completion("", [_tc("write_file", {"path": "b.txt", "content": big}, "c2")]),
        Completion("done", []),
    ])
    agent = Agent(
        client,
        tools=_registry(tmp_path),
        context_budget=120,
        keep_recent=1,
        summarizer_client=Summarizer(),
    )

    assert agent.run("t" * 1200).outcome == "success"
    assert summary_calls, "compaction must go through the summarizer client"
    assert summary_calls[0][0]["role"] == "system"
