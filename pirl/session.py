"""Append-only session journal with a derived model view.

This is the pirl counterpart of pi's session tree: the journal (a JSONL file)
is the single source of truth for messages, compaction boundaries, context
edits, model switches, and usage. Nothing in it is ever rewritten; every model
request is built by *projecting* the active branch.

Why this shape:

* **Journal is append-only.** Compaction does not delete or rewrite messages —
  it appends a ``compaction`` entry that names the first entry to keep
  (``firstKeptEntryId``). The original entries stay on disk and can be
  inspected, replayed, or revealed again by navigating before the edit.
* **The view is a pure function.** ``build_context()`` derives the message list
  from the branch, so a compaction is reversible by construction.
* **State is a file.** The same JSONL doubles as journal, checkpoint, trace, and
  audit artifact — there is no second store that can drift out of sync.

Entry types
-----------
``session``        header (id, cwd, model, version) — first line only
``message``        one chat message
``compaction``     summary + ``firstKeptEntryId`` + ``tokensBefore``
``context_edit``   omit (``replacement: null``) or replace one target entry
``model_change``   model/provider switch, replayed on resume
``usage``          model-attributed usage that is not an assistant message
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SESSION_VERSION = 3
CONFIG_DIR_NAME = ".pirl"
SESSIONS_DIR_NAME = "sessions"
COMPACTED_PREFIX = "[COMPACTED]"

# Keys that only exist for local accounting and must never reach the provider.
RUNTIME_ONLY_PREFIX = "_"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


class SessionError(Exception):
    """Raised when a session file cannot be read or is structurally invalid."""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class SessionStore:
    """Append-only JSONL persistence for one session file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, entry: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SessionError(f"invalid session line: {error}") from error
        return entries

    def exists(self) -> bool:
        return self.path.exists()


def default_session_path(workspace: str | Path, session_id: Optional[str] = None) -> Path:
    """``<cwd>/.pirl/sessions/<timestamp>_<id>.jsonl``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sid = session_id or uuid.uuid4().hex[:8]
    return Path(workspace) / CONFIG_DIR_NAME / SESSIONS_DIR_NAME / f"{stamp}_{sid}.jsonl"


def latest_session_path(workspace: str | Path) -> Optional[Path]:
    root = Path(workspace) / CONFIG_DIR_NAME / SESSIONS_DIR_NAME
    if not root.is_dir():
        return None
    files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedEntry:
    """One journal entry plus the messages it contributes to the model view."""

    entry: Dict[str, Any]
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.entry.get("id", ""))


def _apply_edit(message: Mapping[str, Any], edit: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the replacement message, or ``None`` when the target is omitted."""
    replacement = edit.get("replacement")
    if replacement is None:
        return None
    if not isinstance(replacement, Mapping):
        replacement = {"content": str(replacement)}
    updated = dict(message)
    if "content" in replacement:
        updated["content"] = replacement["content"]
    return updated


def entry_messages(
    entry: Mapping[str, Any],
    edit: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Map one session entry to the messages it contributes to the model view."""
    kind = str(entry.get("type", ""))

    if kind == "message":
        message = entry.get("message")
        if not isinstance(message, Mapping):
            return []
        message = dict(message)
        if edit is not None:
            replaced = _apply_edit(message, edit)
            return [replaced] if replaced is not None else []
        return [message]

    if kind == "compaction":
        if edit is not None and edit.get("replacement") is None:
            return []
        summary = str(entry.get("summary", "")).strip()
        first_kept = str(entry.get("firstKeptEntryId", ""))
        header = f"{COMPACTED_PREFIX} earlier progress"
        if first_kept:
            header += f" (kept from entry {first_kept})"
        return [{"role": "user", "content": f"{header}:\n{summary}"}]

    if kind == "branch_summary":
        summary = str(entry.get("summary", "")).strip()
        return [{"role": "user", "content": f"[BRANCH SUMMARY]\n{summary}"}]

    # context_edit / model_change / usage / custom carry no model context.
    return []


def strip_runtime_keys(messages: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Drop local-only keys (``_usage`` etc.) before sending a request."""
    return [
        {
            key: value
            for key, value in message.items()
            if not str(key).startswith(RUNTIME_ONLY_PREFIX)
        }
        for message in messages
    ]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Owns one append-only journal and derives the model view from it."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        *,
        cwd: Optional[str | Path] = None,
        model: str = "",
        provider: str = "",
        entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.store = (
            store
            if store is None or isinstance(store, SessionStore)
            else SessionStore(store)
        )
        if entries is None:
            self.entries = self.store.load() if self.store is not None else []
        else:
            self.entries = list(entries)
        self._leaf_id: Optional[str] = None

        if self.entries:
            header = self.entries[0]
            if header.get("type") != "session":
                raise SessionError("session file must start with a session header")
            self.cwd = str(header.get("cwd", cwd or ""))
            self.model = str(header.get("model", model))
            self.provider = str(header.get("provider", provider))
            self.session_id = str(header.get("id", ""))
            self._leaf_id = self.entries[-1].get("id")
        else:
            self.cwd = str(cwd or "")
            self.model = model
            self.provider = provider
            self.session_id = uuid.uuid4().hex[:8]
            self._append(
                {
                    "type": "session",
                    "version": SESSION_VERSION,
                    "id": self.session_id,
                    "timestamp": now_iso(),
                    "cwd": self.cwd,
                    "model": model,
                    "provider": provider,
                },
                header=True,
            )

    # -- construction ----------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> "SessionManager":
        store = SessionStore(path)
        return cls(store, entries=store.load(), **kwargs)

    @classmethod
    def create(cls, path: str | Path, **kwargs: Any) -> "SessionManager":
        path = Path(path)
        if path.exists():
            raise SessionError(f"session file already exists: {path}")
        return cls(SessionStore(path), **kwargs)

    # -- append ----------------------------------------------------------

    def _append(self, entry: Dict[str, Any], *, header: bool = False) -> Dict[str, Any]:
        if header:
            # The header is metadata, not a tree node: it has no parent and is
            # never a parent. Session entries form the tree.
            if self.store is not None:
                self.store.append(entry)
            self.entries.append(entry)
            return entry
        entry = {"id": _new_id(), "parentId": self._leaf_id, **entry}
        entry.setdefault("timestamp", now_iso())
        if self.store is not None:
            self.store.append(entry)
        self.entries.append(entry)
        if entry.get("id"):
            self._leaf_id = entry["id"]
        return entry

    def append_message(self, message: Mapping[str, Any], usage: Mapping[str, int] | None = None) -> Dict[str, Any]:
        stored = dict(message)
        if usage is not None:
            stored["_usage"] = dict(usage)
        return self._append({"type": "message", "message": stored})

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int = 0,
        *,
        details: Mapping[str, Any] | None = None,
        usage: Mapping[str, int] | None = None,
        previous_summary_used: bool = False,
        kind: str = "summary",
    ) -> Dict[str, Any]:
        return self._append(
            {
                "type": "compaction",
                "kind": kind,
                "summary": summary,
                "firstKeptEntryId": first_kept_entry_id,
                "tokensBefore": int(tokens_before),
                "details": dict(details or {}),
                "usage": dict(usage or {}),
                "previousSummaryUsed": bool(previous_summary_used),
            }
        )

    def append_context_edit(self, target_id: str, replacement: Any = None) -> Dict[str, Any]:
        return self._append(
            {"type": "context_edit", "targetId": str(target_id), "replacement": replacement}
        )

    def append_model_change(self, provider: str, model_id: str) -> Dict[str, Any]:
        self.model = model_id
        self.provider = provider
        return self._append(
            {"type": "model_change", "provider": provider, "modelId": model_id}
        )

    def append_usage(self, kind: str, usage: Mapping[str, int], *, provider: str = "", model: str = "") -> Dict[str, Any]:
        return self._append(
            {
                "type": "usage",
                "kind": kind,
                "provider": provider or self.provider,
                "model": model or self.model,
                "usage": dict(usage),
            }
        )

    def import_messages(self, messages: Iterable[Mapping[str, Any]]) -> None:
        for message in messages:
            self.append_message(message)

    # -- derived views ---------------------------------------------------

    @property
    def leaf_id(self) -> Optional[str]:
        return self._leaf_id

    def branch(self, leaf_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Entries from the header to ``leaf_id`` (default: newest leaf)."""
        if not self.entries:
            return []
        index = {str(e.get("id", "")): e for e in self.entries if e.get("id")}
        target = leaf_id or self._leaf_id
        current = index.get(str(target)) if target else None
        if current is None:
            current = self.entries[-1]

        path: List[Dict[str, Any]] = []
        seen = set()
        while current is not None:
            entry_id = str(current.get("id", ""))
            if entry_id in seen:  # defensive: corrupt parent cycle
                break
            seen.add(entry_id)
            path.append(current)
            parent_id = current.get("parentId")
            current = index.get(str(parent_id)) if parent_id else None
        path.reverse()
        header = self.entries[0]
        if path and path[0].get("type") != "session":
            path.insert(0, header)
        return path

    def projection(self, leaf_id: Optional[str] = None) -> List[ProjectedEntry]:
        """Apply compaction boundaries and context edits to the branch."""
        path = self.branch(leaf_id)
        index_by_id = {
            str(entry.get("id", "")): position for position, entry in enumerate(path)
        }

        edits: Dict[str, Dict[str, Any]] = {}
        for entry in path:
            if entry.get("type") == "context_edit":
                edits[str(entry.get("targetId", ""))] = entry

        compaction_index = -1
        for position, entry in enumerate(path):
            if entry.get("type") == "compaction":
                compaction_index = position

        selected = path
        if compaction_index >= 0:
            compaction = path[compaction_index]
            kept_index = index_by_id.get(str(compaction.get("firstKeptEntryId", "")), -1)
            # A boundary that points at an older compaction resolves to *that*
            # compaction's own boundary, so retained messages get re-summarized
            # and superseded notes never accumulate.
            while 0 <= kept_index < compaction_index and path[kept_index].get("type") == "compaction":
                inner = index_by_id.get(str(path[kept_index].get("firstKeptEntryId", "")), -1)
                if inner < 0 or inner >= kept_index:
                    kept_index = -1
                    break
                kept_index = inner
            if 0 <= kept_index <= compaction_index:
                retained = path[kept_index:compaction_index] + path[compaction_index + 1:]
            elif kept_index > compaction_index:
                retained = path[kept_index:]
            else:
                retained = path[compaction_index + 1:]
            selected = [compaction] + [
                entry for entry in retained if entry.get("type") != "compaction"
            ]
            # The stable system prefix is never summarized away: keep the first
            # system message of the branch byte-identical across compactions.
            lead = next(
                (
                    entry
                    for entry in path[:compaction_index]
                    if entry.get("type") == "message"
                    and str((entry.get("message") or {}).get("role", "")) == "system"
                ),
                None,
            )
            if lead is not None and all(entry.get("id") != lead.get("id") for entry in selected):
                selected = [lead] + selected

        projected: List[ProjectedEntry] = []
        for entry in selected:
            edit = edits.get(str(entry.get("id", ""))) if entry.get("type") == "message" else None
            projected.append(ProjectedEntry(entry=entry, messages=entry_messages(entry, edit)))
        return projected

    def build_context(self, leaf_id: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for projected in self.projection(leaf_id):
            messages.extend(projected.messages)
        return messages

    def message_owners(self, leaf_id: Optional[str] = None) -> List[str]:
        """Entry id for each message index in :meth:`build_context`."""
        owners: List[str] = []
        for projected in self.projection(leaf_id):
            owners.extend([projected.id] * len(projected.messages))
        return owners

    def message_index_of(self, entry_id: str, leaf_id: Optional[str] = None) -> int:
        owners = self.message_owners(leaf_id)
        try:
            return owners.index(str(entry_id))
        except ValueError:
            return -1

    def compactions(self, leaf_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.branch(leaf_id) if e.get("type") == "compaction"]

    def last_model(self, leaf_id: Optional[str] = None) -> str:
        model = self.model
        for entry in self.branch(leaf_id):
            if entry.get("type") == "model_change":
                model = str(entry.get("modelId", model))
            elif entry.get("type") == "message":
                message = entry.get("message") or {}
                if message.get("role") == "assistant" and message.get("model"):
                    model = str(message["model"])
        return model

    # -- audit -----------------------------------------------------------

    def identity(self) -> Dict[str, Any]:
        """Facts that make a journal replay safe (or not)."""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "model": self.last_model(),
            "entries": len(self.entries),
            "leaf_id": self._leaf_id,
        }

    def identity_mismatch(self, *, cwd: Optional[str] = None, model: Optional[str] = None) -> List[str]:
        """Report identity fields that differ from what the journal recorded."""
        mismatched: List[str] = []
        if cwd is not None and self.cwd and str(cwd) != self.cwd:
            mismatched.append("cwd")
        if model is not None and self.last_model() and model != self.last_model():
            mismatched.append("model")
        return mismatched

    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for entry in self.entries:
            kind = str(entry.get("type", "?"))
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "entries": len(self.entries),
            "kinds": kinds,
            "compactions": kinds.get("compaction", 0),
            "context_edits": kinds.get("context_edit", 0),
            "path": str(self.store.path) if self.store is not None else "",
        }
