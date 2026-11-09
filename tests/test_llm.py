import io
import json
import urllib.error

import pytest

from pirl.llm import (
    ChatClient,
    LLMConfigError,
    LLMHTTPError,
    LLMResponseError,
    ToolCall,
    client_from_env,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


def _client():
    return ChatClient(base_url="https://api.example.com/v1", api_key="k", model="m")


def test_complete_parses_text_and_tool_calls(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "hello",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.txt"}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(json.dumps(payload).encode()),
    )
    completion = _client().complete([{"role": "user", "content": "hi"}])
    assert completion.text == "hello"
    assert completion.tool_calls == [ToolCall("read_file", '{"path": "a.txt"}')]


def test_complete_sends_tools(monkeypatch):
    captured = {}

    def fake_urlopen(request, **kw):
        captured["body"] = json.loads(request.data.decode())
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    from pirl.llm import ToolSpec

    _client().complete(
        [{"role": "user", "content": "hi"}],
        tools=[ToolSpec("t", "a tool", {"type": "object"})],
    )
    assert captured["body"]["tools"] == [
        {
            "type": "function",
            "function": {"name": "t", "description": "a tool", "parameters": {"type": "object"}},
        }
    ]


def test_complete_raises_on_http_error(monkeypatch):
    def fake_urlopen(*a, **kw):
        raise urllib.error.HTTPError(
            "https://api.example.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error": "bad key"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(LLMHTTPError) as exc:
        _client().complete([{"role": "user", "content": "hi"}])
    assert exc.value.status == 401
    assert "bad key" in exc.value.body


def test_complete_parses_reasoning(monkeypatch):
    payload = {
        "choices": [
            {"message": {"content": "answer", "reasoning_content": "hmm let me think"}}
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(json.dumps(payload).encode()),
    )
    completion = _client().complete([{"role": "user", "content": "hi"}])
    assert completion.text == "answer"
    assert completion.reasoning == "hmm let me think"


def test_complete_parses_usage(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(json.dumps(payload).encode()),
    )
    completion = _client().complete([{"role": "user", "content": "hi"}])
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.total_tokens == 15


def test_complete_reasoning_defaults_to_empty(monkeypatch):
    payload = {"choices": [{"message": {"content": "answer"}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(json.dumps(payload).encode()),
    )
    completion = _client().complete([{"role": "user", "content": "hi"}])
    assert completion.reasoning == ""


def test_complete_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(b"not json"),
    )
    with pytest.raises(LLMResponseError):
        _client().complete([{"role": "user", "content": "hi"}])


def test_complete_raises_on_missing_choices(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: _FakeResponse(b"{}"),
    )
    with pytest.raises(LLMResponseError):
        _client().complete([{"role": "user", "content": "hi"}])


def test_complete_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "too many", {}, io.BytesIO(b"{}"))
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    completion = _client().complete([{"role": "user", "content": "hi"}])
    assert completion.text == "ok"
    assert calls["n"] == 2


def test_complete_gives_up_after_retries(monkeypatch):
    def fake_urlopen(*a, **kw):
        raise urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(LLMHTTPError) as exc:
        _client().complete([{"role": "user", "content": "hi"}])
    assert exc.value.status == 500


def test_client_from_env_requires_all_vars():
    with pytest.raises(LLMConfigError) as exc:
        client_from_env({})
    assert "PIRL_BASE_URL" in str(exc.value)
    assert "PIRL_API_KEY" in str(exc.value)
    assert "PIRL_MODEL" in str(exc.value)


def test_client_from_env_builds_client():
    client = client_from_env(
        {
            "PIRL_BASE_URL": "https://api.example.com/v1",
            "PIRL_API_KEY": "k",
            "PIRL_MODEL": "m",
        }
    )
    assert client.base_url == "https://api.example.com/v1"
    assert client.model == "m"
