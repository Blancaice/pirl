"""Layered, non-destructive context preparation for model requests.

The canonical message journal is never rewritten. This module builds a smaller
wire view using deterministic observation de-duplication and object
externalization first, and LLM summarization only above a high-water mark.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .plugins.compaction import compact, estimate_messages_tokens


class ContextManager:
    """Prepare bounded model context while retaining the full agent journal."""

    def __init__(
        self,
        summarize: Callable[[str], str],
        budget: int | None = 16000,
        keep_recent: int = 6,
        *,
        object_dir: str | Path | None = None,
        max_inline_chars: int = 4000,
        high_water_ratio: float = 0.8,
        max_summary_failures: int = 2,
    ) -> None:
        self.summarize = summarize
        self.budget = budget
        self.keep_recent = keep_recent
        self.object_dir = Path(object_dir) if object_dir is not None else None
        self.max_inline_chars = max(256, max_inline_chars)
        self.high_water_ratio = min(1.0, max(0.1, high_water_ratio))
        self.max_summary_failures = max(1, max_summary_failures)
        self._summary_failures = 0

    def prepare(self, messages: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        """Return a reduced copy suitable for one request without mutating input."""
        wire = self._reduce_observations(messages)
        if not self.budget:
            return wire
        high_water = max(1, int(self.budget * self.high_water_ratio))
        if estimate_messages_tokens(wire) <= high_water:
            return wire
        if self._summary_failures < self.max_summary_failures:
            try:
                summarized = compact(
                    self.summarize, wire, task, keep_recent=self.keep_recent
                )
                self._summary_failures = 0
                if summarized is not wire:
                    return summarized
            except Exception:
                self._summary_failures += 1
        return self._bounded_fallback(wire)

    def maybe_compact(self, messages: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        """Compatibility alias for older integrations."""
        return self.prepare(messages, task)

    def _reduce_observations(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        wire = copy.deepcopy(list(messages))
        calls: Dict[str, tuple[str, Dict[str, Any], int]] = {}
        observations: List[tuple[int, str, Dict[str, Any], str, str]] = []
        for index, message in enumerate(wire):
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    call_id = str(call.get("id", ""))
                    fn = call.get("function") or {}
                    raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except (TypeError, ValueError):
                        args = {}
                    calls[call_id] = (str(fn.get("name", "")), args, index)
            elif message.get("role") == "tool":
                call_id = str(message.get("tool_call_id", ""))
                name, args, _ = calls.get(call_id, ("", {}, -1))
                content = str(message.get("content", ""))
                observations.append((index, name, args, content, self._digest(content)))

        newest_file_version: Dict[tuple[str, str], int] = {}
        newest_generic: Dict[str, int] = {}
        decoded: Dict[int, str] = {}
        for index, name, args, raw, digest in observations:
            payload = self._payload_content(raw)
            decoded[index] = payload
            if name == "read_file":
                newest_file_version[(str(args.get("path", "")), self._digest(payload))] = index
            else:
                newest_generic[digest] = index

        for index, name, args, raw, digest in observations:
            payload = decoded[index]
            if name == "read_file":
                path = str(args.get("path", ""))
                object_hash = self._store(payload)
                duplicate = newest_file_version[(path, self._digest(payload))] != index
                if duplicate:
                    wire[index]["content"] = self._reference(
                        "read_file", object_hash, len(payload), path=path,
                        note="unchanged duplicate; newest observation retained",
                    )
                elif len(payload) > self.max_inline_chars:
                    wire[index]["content"] = self._preview_reference(
                        "read_file", payload, object_hash, path=path
                    )
            elif newest_generic[digest] != index:
                object_hash = self._store(raw)
                wire[index]["content"] = self._reference(
                    name or "tool", object_hash, len(raw),
                    note="duplicate tool result; newest occurrence retained",
                )
            elif len(raw) > self.max_inline_chars:
                object_hash = self._store(raw)
                wire[index]["content"] = self._preview_reference(name or "tool", raw, object_hash)

        recent_start = max(0, len(wire) - self.keep_recent)
        for call_id, (name, args, assistant_index) in calls.items():
            if name not in {"write_file", "apply_patch", "edit_file"}:
                continue
            call = self._find_call(wire[assistant_index], call_id)
            if call is None:
                continue
            fn = call.get("function") or {}
            body_key = "content" if "content" in args else "patch" if "patch" in args else None
            if body_key is None:
                continue
            body = str(args[body_key])
            if assistant_index >= recent_start and len(body) <= self.max_inline_chars:
                continue
            object_hash = self._store(body)
            compact_args = dict(args)
            compact_args[body_key] = f"[COMPRESSED object=sha256:{object_hash} chars={len(body)}]"
            fn["arguments"] = json.dumps(compact_args, ensure_ascii=False)
        return wire

    def _bounded_fallback(self, wire: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(wire) <= self.keep_recent + 2:
            return wire
        boundary = len(wire) - self.keep_recent
        while boundary > 1 and wire[boundary].get("role") == "tool":
            boundary -= 1
        note = {"role": "user", "content": "[COMPRESSED] Earlier trajectory archived; recent canonical state follows."}
        return [wire[0], note, *wire[boundary:]]

    def _store(self, text: str) -> str:
        digest = self._digest(text)
        if self.object_dir is not None:
            self.object_dir.mkdir(parents=True, exist_ok=True)
            path = self.object_dir / f"{digest}.txt"
            if not path.exists():
                path.write_text(text, encoding="utf-8")
        return digest

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _payload_content(raw: str) -> str:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if isinstance(value, Mapping) and isinstance(value.get("content"), str):
            return value["content"]
        return raw

    @staticmethod
    def _find_call(message: Dict[str, Any], call_id: str) -> Dict[str, Any] | None:
        return next((call for call in message.get("tool_calls") or [] if str(call.get("id", "")) == call_id), None)

    def _preview_reference(self, source: str, text: str, digest: str, *, path: str = "") -> str:
        half = self.max_inline_chars // 2
        preview = text[:half] + "\n...[COMPRESSED]...\n" + text[-half:]
        return self._reference(source, digest, len(text), path=path, preview=preview)

    @staticmethod
    def _reference(source: str, digest: str, chars: int, *, path: str = "", note: str = "large observation externalized", preview: str = "") -> str:
        payload = {"compressed": True, "source": source, "path": path, "object": f"sha256:{digest}", "chars": chars, "note": note}
        if preview:
            payload["preview"] = preview
        return "[COMPRESSED] " + json.dumps(payload, ensure_ascii=False)
