"""Minimal OpenAI-compatible chat-completions client (stdlib only).

pirl stays dependency-light: this client speaks the ubiquitous
``POST /chat/completions`` protocol using ``urllib``, so it works with any
OpenAI-compatible endpoint (DeepSeek, SiliconFlow, Ollama's OpenAI endpoint,
right.codes, ...) without adding an HTTP dependency.

Configuration is read from the environment by :func:`client_from_env`:

* ``PIRL_BASE_URL`` — provider base URL (e.g. ``https://api.deepseek.com/v1``)
* ``PIRL_API_KEY``  — bearer key
* ``PIRL_MODEL``    — model id
* ``PIRL_REQUEST_TIMEOUT`` — optional request timeout in seconds (unset = no timeout)
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class LLMError(Exception):
    """Base class for chat-client failures."""


class LLMConfigError(LLMError):
    """Raised when provider configuration is missing or invalid."""


class LLMHTTPError(LLMError):
    """Raised when the provider returns a non-2xx status."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"provider returned HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class LLMResponseError(LLMError):
    """Raised when the provider response cannot be parsed."""


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the model asked for."""

    name: str
    arguments: str  # JSON string, kept verbatim for the caller to decode
    id: str = ""  # provider-assigned call id, for tool-result routing


@dataclass(frozen=True)
class Usage:
    """Token accounting for one response (disjoint input/output totals)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class Completion:
    """The assistant message returned by one chat-completions call."""

    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: str = ""  # model's thinking/reasoning, empty for non-reasoning models
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""  # provider finish_reason: "stop" | "length" | "tool_calls" | ...


@dataclass(frozen=True)
class ToolSpec:
    """A JSON-Schema tool declaration sent to the model."""

    name: str
    description: str
    parameters: Dict[str, Any]


def _tool_spec_to_payload(tool: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class ChatClient:
    """Talks ``/chat/completions`` to one OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] = (),
        max_retries: int = 3,
    ) -> Completion:
        """Send a chat-completions request and return the assistant message.

        Retries transient failures (429, 5xx, connection/timeout) with
        exponential backoff + jitter, honouring ``Retry-After`` when present.
        """
        payload = self._payload(messages, tools)

        body = json.dumps(payload).encode("utf-8")

        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                return _parse_completion(raw)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                if _retryable_status(error.code) and attempt < max_retries:
                    time.sleep(_retry_delay(error, attempt))
                    continue
                raise LLMHTTPError(error.code, detail) from error
            except urllib.error.URLError as error:
                if attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise LLMError(f"cannot reach provider: {error.reason}") from error
            except TimeoutError as error:
                if attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                if self.timeout is None:
                    raise LLMError("provider request timed out") from error
                raise LLMError(
                    f"provider request timed out after {self.timeout:g} seconds"
                ) from error

        raise LLMError("retries exhausted")  # pragma: no cover - unreachable

    def _payload(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Sequence[ToolSpec] = (),
        *,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = [_tool_spec_to_payload(t) for t in tools]
        if stream:
            payload["stream"] = True
            # Ask for the final usage chunk; providers that ignore it do no harm.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def stream(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] = (),
        on_delta: Any | None = None,
        on_reasoning: Any | None = None,
        max_retries: int = 3,
    ) -> Completion:
        """Stream one chat-completions request and return the assembled message.

        Emits text deltas to ``on_delta`` and thinking deltas to ``on_reasoning``
        as they arrive, so the caller can render progress instead of waiting for
        the whole response. Tool-call arguments are accumulated per index.
        """
        body = json.dumps(self._payload(messages, tools, stream=True)).encode("utf-8")
        emitted = {"value": False}

        def mark_emitted() -> None:
            emitted["value"] = True

        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return _consume_stream(
                        response,
                        on_delta=on_delta,
                        on_reasoning=on_reasoning,
                        mark_emitted=mark_emitted,
                    )
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                if _retryable_status(error.code) and attempt < max_retries and not emitted["value"]:
                    time.sleep(_retry_delay(error, attempt))
                    continue
                raise LLMHTTPError(error.code, detail) from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt < max_retries and not emitted["value"]:
                    time.sleep(_backoff(attempt))
                    continue
                if isinstance(error, TimeoutError):
                    if self.timeout is None:
                        raise LLMError("provider request timed out") from error
                    raise LLMError(
                        f"provider request timed out after {self.timeout:g} seconds"
                    ) from error
                raise LLMError(f"cannot reach provider: {error.reason}") from error

        raise LLMError("retries exhausted")  # pragma: no cover - unreachable

    def __repr__(self) -> str:
        return f"ChatClient(base_url={self.base_url!r}, model={self.model!r})"


def _consume_stream(
    response: Any,
    *,
    on_delta: Any | None = None,
    on_reasoning: Any | None = None,
    mark_emitted: Any | None = None,
) -> Completion:
    """Parse an SSE response into one :class:`Completion`."""
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    stop_reason = ""
    usage = Usage()

    def emit_text(chunk: str) -> None:
        text_parts.append(chunk)
        if mark_emitted is not None:
            mark_emitted()
        if on_delta is not None and chunk:
            on_delta(chunk)

    for raw_line in response:
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if isinstance(chunk.get("usage"), dict):
            usage = Usage(
                prompt_tokens=int(chunk["usage"].get("prompt_tokens") or 0),
                completion_tokens=int(chunk["usage"].get("completion_tokens") or 0),
                total_tokens=int(chunk["usage"].get("total_tokens") or 0),
            )

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if thinking:
                reasoning_parts.append(str(thinking))
                if on_reasoning is not None:
                    on_reasoning(str(thinking))
            content = delta.get("content") or ""
            if content:
                emit_text(str(content))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index") or 0)
                entry = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    entry["id"] = str(call["id"])
                function = call.get("function") or {}
                if function.get("name"):
                    entry["name"] = str(function["name"])
                if function.get("arguments"):
                    entry["arguments"] += str(function["arguments"])
            if choice.get("finish_reason"):
                stop_reason = str(choice["finish_reason"])

    tool_calls = [
        ToolCall(name=calls[index]["name"], arguments=calls[index]["arguments"], id=calls[index]["id"])
        for index in sorted(calls)
        if calls[index]["name"]
    ]
    return Completion(
        text="".join(text_parts),
        tool_calls=tool_calls,
        reasoning="".join(reasoning_parts),
        usage=usage,
        stop_reason=stop_reason,
    )


def _parse_completion(raw: str) -> Completion:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LLMResponseError(f"provider returned invalid JSON: {error}") from error

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMResponseError("provider response missing choices[0].message") from error

    text = message.get("content")
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    tool_calls: List[ToolCall] = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") or {}
        tool_calls.append(
            ToolCall(
                name=function.get("name", ""),
                arguments=function.get("arguments", ""),
                id=raw_call.get("id", ""),
            )
        )

    # DeepSeek (and some others) expose reasoning as ``reasoning_content``;
    # other providers use ``reasoning``. Normalize both, default to empty.
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    usage_raw = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        total_tokens=int(usage_raw.get("total_tokens") or 0),
    )

    stop_reason = str(data.get("choices", [{}])[0].get("finish_reason") or "")

    return Completion(
        text=text,
        tool_calls=tool_calls,
        reasoning=reasoning,
        usage=usage,
        stop_reason=stop_reason,
    )


def _retryable_status(status: int) -> bool:
    return status in (429, 500, 502, 503, 504)


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s, ... (±20%)."""
    base = 2 ** attempt
    return base * (0.8 + 0.4 * random.random())


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    """Delay for a retryable HTTP error, honouring ``Retry-After``."""
    header = error.headers.get("Retry-After", "") if error.headers else ""
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return _backoff(attempt)


def client_from_env(env: Optional[Dict[str, str]] = None) -> ChatClient:
    """Build a :class:`ChatClient` from ``PIRL_*`` environment variables.

    Raises :class:`LLMConfigError` listing every missing variable.
    """
    values = dict(os.environ if env is None else env)
    base_url = values.get("PIRL_BASE_URL", "").strip()
    api_key = values.get("PIRL_API_KEY", "").strip()
    model = values.get("PIRL_MODEL", "").strip()

    missing = [name for name, value in
               (("PIRL_BASE_URL", base_url), ("PIRL_API_KEY", api_key), ("PIRL_MODEL", model))
               if not value]
    if missing:
        raise LLMConfigError("missing provider config: " + ", ".join(missing))

    timeout_value = values.get("PIRL_REQUEST_TIMEOUT", "").strip()
    timeout: Optional[float] = None
    if timeout_value:
        try:
            timeout = float(timeout_value)
        except ValueError as error:
            raise LLMConfigError("PIRL_REQUEST_TIMEOUT must be a positive number") from error
        if timeout <= 0:
            raise LLMConfigError("PIRL_REQUEST_TIMEOUT must be a positive number")

    return ChatClient(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
