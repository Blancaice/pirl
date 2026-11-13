"""Layered, non-destructive context preparation for model requests.

The canonical journal is never rewritten. Each request gets a derived *view*
built in escalating stages:

1. **Object externalization** — any single observation larger than
   ``max_inline_chars`` is written to ``.pirl/objects/<sha256>.txt`` and replaced
   by a head/tail preview plus a content-addressed reference, so it stays
   recoverable by re-reading the object. The rendering depends only on the
   message's own content, so it never changes on a later turn.
2. **Observation de-duplication (opt-in, off by default)** — collapsing a
   repeated read means rewriting an *older* message, which invalidates the
   provider's prompt cache from that offset onwards. It is available as
   ``dedupe=True`` for callers that prefer tokens over prefill latency.
3. **LLM summarization** — only above the high-water mark, and only if a valid,
   persistable summary is produced. Otherwise the view falls back to a
   deterministic bounded tail after ``max_summary_failures`` attempts.

Every stage returns a :class:`ContextView` that records *what was reduced and
where the kept tail starts*, so the agent can persist the decision as a
``compaction`` entry instead of leaving it as an unauditable in-memory note.

The stability contract: a message already sent in a previous request renders
to the same bytes in every later request, unless a compaction replaces the
whole span at once. That is what makes prompt caching effective.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .plugins.compaction import (
    DEFAULT_KEEP_RECENT,
    COMPACTED_PREFIX,
    CompactionPreparation,
    SummarizationError,
    append_file_lists,
    build_summary_prompt,
    estimate_context_tokens,
    file_operations,
    prepare_compaction,
    render_messages,
)

COMPRESSED_PREFIX = "[COMPRESSED]"
FALLBACK_NOTE = f"{COMPRESSED_PREFIX} Earlier trajectory archived; recent canonical state follows."


@dataclass
class CompactionRecord:
    """The view decision, ready to be appended to the session journal."""

    summary: str
    first_kept_index: int
    tokens_before: int
    details: Dict[str, Any] = field(default_factory=dict)
    kind: str = "summary"  # "summary" | "fallback"
    previous_summary_used: bool = False


@dataclass
class ContextView:
    """One request's bounded message list plus the reason it was bounded."""

    messages: List[Dict[str, Any]]
    compaction: Optional[CompactionRecord] = None
    summary_failures: int = 0

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)


class ContextManager:
    """Prepare bounded model context while retaining the full journal."""

    def __init__(
        self,
        summarize: Callable[[str], str],
        budget: int | None = 16000,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        *,
        keep_recent_tokens: int | None = None,
        object_dir: str | Path | None = None,
        max_inline_chars: int = 4000,
        high_water_ratio: float = 0.8,
        max_summary_failures: int = 2,
        dedupe: bool = False,
    ) -> None:
        self.summarize = summarize
        self.budget = budget
        self.keep_recent = keep_recent
        self.keep_recent_tokens = keep_recent_tokens
        self.object_dir = Path(object_dir) if object_dir is not None else None
        self.max_inline_chars = max(256, max_inline_chars)
        self.high_water_ratio = min(1.0, max(0.1, high_water_ratio))
        self.max_summary_failures = max(1, max_summary_failures)
        # Collapsing a repeated observation rewrites an *older* message, which
        # invalidates the provider's prompt cache from that offset onwards. It is
        # therefore off by default: externalization (content-determined, hence
        # stable) plus compaction keeps context bounded without moving the prefix.
        self.dedupe = bool(dedupe)
        self._summary_failures = 0
        self._previous_summary = ""

    # -- public API ------------------------------------------------------

    def prepare_view(
        self,
        messages: List[Dict[str, Any]],
        task: str,
    ) -> ContextView:
        """Return a reduced copy plus the compaction decision, without mutating input."""
        wire = self._reduce_observations(messages)
        if not self.budget:
            return ContextView(wire, None, self._summary_failures)

        high_water = max(1, int(self.budget * self.high_water_ratio))
        if estimate_context_tokens(wire) <= high_water:
            return ContextView(wire, None, self._summary_failures)

        return self._compact_or_fallback(wire, task)

    def prepare(self, messages: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        """Compatibility wrapper: the message list for one request."""
        return self.prepare_view(messages, task).messages

    def maybe_compact(self, messages: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        """Compatibility alias for older integrations."""
        return self.prepare(messages, task)

    def compact_now(self, messages: List[Dict[str, Any]], task: str) -> ContextView:
        """Force a compaction attempt (overflow recovery), ignoring the high-water mark."""
        wire = self._reduce_observations(messages)
        return self._compact_or_fallback(wire, task)

    # -- compaction ------------------------------------------------------

    def _compact_or_fallback(self, wire: List[Dict[str, Any]], task: str) -> ContextView:
        preparation = prepare_compaction(
            wire,
            keep_recent=self.keep_recent,
            keep_recent_tokens=self.keep_recent_tokens,
        )
        if preparation is None:
            return ContextView(wire, None, self._summary_failures)

        if self._summary_failures >= self.max_summary_failures:
            return self._bounded_fallback(wire, preparation)

        prompt = build_summary_prompt(
            render_messages(preparation.messages_to_summarize),
            self._previous_summary,
        )
        previous_summary_used = bool(self._previous_summary)
        try:
            summary = str(self.summarize(prompt)).strip()
            if not summary:
                raise SummarizationError("summarizer returned an empty summary")
        except Exception:
            # Never persist an unusable summary; fall back deterministically and
            # stop paying for further attempts after max_summary_failures.
            self._summary_failures += 1
            return self._bounded_fallback(wire, preparation)

        read_files, modified_files = file_operations(preparation.messages_to_summarize)
        summary = append_file_lists(summary, read_files, modified_files)
        note = {
            "role": "user",
            "content": (
                f"{COMPACTED_PREFIX} earlier progress "
                f"(kept tail follows):\n{summary}"
            ),
        }
        kept = wire[preparation.first_kept_index:]
        self._previous_summary = summary
        self._summary_failures = 0
        return ContextView(
            [wire[0], note, *kept],
            CompactionRecord(
                summary=summary,
                first_kept_index=preparation.first_kept_index,
                tokens_before=preparation.tokens_before,
                details={"readFiles": read_files, "modifiedFiles": modified_files},
                kind="summary",
                previous_summary_used=previous_summary_used,
            ),
            self._summary_failures,
        )

    def _bounded_fallback(
        self,
        wire: List[Dict[str, Any]],
        preparation: CompactionPreparation | None = None,
    ) -> ContextView:
        """Deterministic bounded tail: keep the newest messages verbatim."""
        if len(wire) <= self.keep_recent + 2:
            return ContextView(wire, None, self._summary_failures)
        if preparation is not None:
            boundary = preparation.first_kept_index
        else:
            boundary = len(wire) - self.keep_recent
        boundary = max(1, min(boundary, len(wire) - 1))
        while boundary > 1 and wire[boundary].get("role") == "tool":
            boundary -= 1
        note = {"role": "user", "content": FALLBACK_NOTE}
        return ContextView(
            [wire[0], note, *wire[boundary:]],
            CompactionRecord(
                summary=FALLBACK_NOTE,
                first_kept_index=boundary,
                tokens_before=estimate_context_tokens(wire),
                details={"readFiles": [], "modifiedFiles": []},
                kind="fallback",
            ),
            self._summary_failures,
        )

    # -- stage 1 + 2: de-duplication and externalization ------------------

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
                content = str(message.get("content", ""))
                if content.startswith(COMPRESSED_PREFIX):
                    # Already reduced (committed or from a resumed journal):
                    # never recompute, or the prefix would shift every turn.
                    continue
                call_id = str(message.get("tool_call_id", ""))
                name, args, _ = calls.get(call_id, ("", {}, -1))
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
                duplicate = (
                    self.dedupe
                    and newest_file_version[(path, self._digest(payload))] != index
                )
                if duplicate:
                    object_hash = self._store(payload)
                    wire[index]["content"] = self._reference(
                        "read_file", object_hash, len(payload), path=path,
                        note="unchanged duplicate; newest observation retained",
                    )
                elif len(payload) > self.max_inline_chars:
                    object_hash = self._store(payload)
                    wire[index]["content"] = self._preview_reference(
                        "read_file", payload, object_hash, path=path
                    )
            elif self.dedupe and newest_generic[digest] != index:
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
            if body.startswith(COMPRESSED_PREFIX) or body.startswith("[COMPRESSED object="):
                continue
            if assistant_index >= recent_start and len(body) <= self.max_inline_chars:
                continue
            object_hash = self._store(body)
            compact_args = dict(args)
            compact_args[body_key] = f"[COMPRESSED object=sha256:{object_hash} chars={len(body)}]"
            fn["arguments"] = json.dumps(compact_args, ensure_ascii=False)
        return wire

    # -- helpers ---------------------------------------------------------

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
        return f"{COMPRESSED_PREFIX} " + json.dumps(payload, ensure_ascii=False)
