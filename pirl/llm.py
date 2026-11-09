"""Minimal OpenAI-compatible chat-completions client (stdlib only).

pirl stays dependency-light: this client speaks the ubiquitous
``POST /chat/completions`` protocol using ``urllib``, so it works with any
OpenAI-compatible endpoint (DeepSeek, SiliconFlow, Ollama's OpenAI endpoint,
right.codes, ...) without adding an HTTP dependency.

Configuration is read from the environment by :func:`client_from_env`:

* ``PIRL_BASE_URL`` — provider base URL (e.g. ``https://api.deepseek.com/v1``)
* ``PIRL_API_KEY``  — bearer key
* ``PIRL_MODEL``    — model id
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
        timeout: float = 120.0,
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
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = [_tool_spec_to_payload(t) for t in tools]

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

        raise LLMError("retries exhausted")  # pragma: no cover - unreachable

    def __repr__(self) -> str:
        return f"ChatClient(base_url={self.base_url!r}, model={self.model!r})"


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

    return Completion(
        text=text, tool_calls=tool_calls, reasoning=reasoning, usage=usage
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

    return ChatClient(base_url=base_url, api_key=api_key, model=model)
