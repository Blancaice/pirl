"""Skill lifecycle state machine and sidecar registry (B1/B2).

Four states, carried by *directories* (activation == moving files, simplest and
most auditable):

    draft ──验证通过──▶ candidate ──人工激活──▶ active ──退化/闲置──▶ retired
       └─验证失败→ 丢弃

Directory layout (``~/.pirl/skills/`` by default)::

    skills/
    ├── active/        # only this directory feeds <available_skills>
    ├── candidate/     # awaiting human activation
    ├── retired/       # retired, never loaded
    └── skill-registry.json   # sidecar registry (status + utility stats)

Skill *files* stay immutable; state and statistics live in the sidecar registry.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .skills import CONFIG_DIR_NAME

STATUS_DRAFT = "draft"
STATUS_CANDIDATE = "candidate"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"

ALL_STATUSES = (STATUS_DRAFT, STATUS_CANDIDATE, STATUS_ACTIVE, STATUS_RETIRED)


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (registry timestamps)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SkillRecord:
    """A registry entry: lifecycle state + utility statistics.

    The utility stats follow the MemRL "Q-value" idea: ``retrievals`` counts how
    often the skill was pulled into a prompt, ``adoptions`` how often the skill
    was adopted (used) at all, ``failures`` the subset of those adoptions where
    the task failed. ``success_rate`` is ``(adoptions - failures) / adoptions``.
    """

    name: str
    status: str  # draft | candidate | active | retired
    created_at: str
    version: int = 1  # +1 on every revise
    # --- utility stats ---
    retrievals: int = 0  # times recalled into the prompt
    adoptions: int = 0  # times adopted and the task succeeded
    failures: int = 0  # times adopted and the task failed
    # --- audit ---
    reviewer: str = ""
    activate_reason: str = ""
    # --- extensions (not in the design's dataclass, added for bookkeeping) ---
    reject_reason: str = ""
    activated_at: str = ""
    retired_at: str = ""
    retire_reason: str = ""

    @property
    def success_rate(self) -> float:
        """Adoption success rate (1.0 when there are no adoptions yet)."""
        if self.adoptions == 0:
            return 1.0
        return (self.adoptions - self.failures) / self.adoptions


_FIELD_NAMES = [f.name for f in fields(SkillRecord)]


def _record_from_dict(payload: dict) -> SkillRecord:
    """Build a :class:`SkillRecord` from a registry dict, tolerating missing keys."""
    kwargs: Dict[str, object] = {}
    for f in fields(SkillRecord):
        if f.name in payload:
            kwargs[f.name] = payload[f.name]
        elif f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:  # pragma: no cover - no factories today
            kwargs[f.name] = f.default_factory()
        else:
            kwargs[f.name] = ""  # required fields (name/status/created_at) fallback
    return SkillRecord(**kwargs)


def default_store_root() -> Path:
    """Default skills store root: ``~/.pirl/skills``."""
    return Path.home() / CONFIG_DIR_NAME / "skills"


class SkillStore:
    """Owns the skills directory layout and the sidecar ``skill-registry.json``."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_store_root()
        self.active_dir = self.root / "active"
        self.candidate_dir = self.root / "candidate"
        self.retired_dir = self.root / "retired"
        self.registry_path = self.root / "skill-registry.json"
        self.init()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def init(self) -> None:
        """Create the directory layout and an empty registry if missing."""
        for directory in (self.active_dir, self.candidate_dir, self.retired_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})

    # ------------------------------------------------------------------ #
    # Registry persistence
    # ------------------------------------------------------------------ #

    def _read_registry(self) -> Dict[str, SkillRecord]:
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        skills = data.get("skills", {}) if isinstance(data, dict) else {}
        if not isinstance(skills, dict):
            return {}
        records: Dict[str, SkillRecord] = {}
        for name, payload in skills.items():
            if isinstance(payload, dict):
                records[str(name)] = _record_from_dict(payload)
        return records

    def _write_registry(self, records: Dict[str, SkillRecord]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": {name: asdict(rec) for name, rec in sorted(records.items())}}
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self.registry_path)

    def _upsert(self, record: SkillRecord) -> None:
        records = self._read_registry()
        records[record.name] = record
        self._write_registry(records)

    # ------------------------------------------------------------------ #
    # Read access
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Optional[SkillRecord]:
        return self._read_registry().get(name)

    def records(self, status: Optional[str] = None) -> List[SkillRecord]:
        records = list(self._read_registry().values())
        if status is not None:
            records = [r for r in records if r.status == status]
        return records

    # ------------------------------------------------------------------ #
    # Lifecycle transitions
    # ------------------------------------------------------------------ #

    def promote_to_candidate(self, proposal, src_dir: str | Path) -> SkillRecord:
        """Move a *verified* proposal dir into ``candidate/<name>/``.

        ``src_dir`` is typically a temporary draft dir that has already passed
        the verification gate. A re-proposal of an existing candidate bumps the
        version (revise).
        """
        self.init()
        src = Path(src_dir)
        existing = self.get(proposal.name)

        version = 1
        if existing is not None and existing.status == STATUS_CANDIDATE:
            version = existing.version + 1  # revise

        dest = self.candidate_dir / proposal.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

        record = SkillRecord(
            name=proposal.name,
            status=STATUS_CANDIDATE,
            created_at=existing.created_at if existing else now_iso(),
            version=version,
            reviewer=existing.reviewer if existing else "",
            activate_reason=existing.activate_reason if existing else "",
        )
        self._upsert(record)
        return record

    def activate(self, name: str, reviewer: str = "", reason: str = "") -> SkillRecord:
        """Human-in-the-loop: ``candidate/<name>`` -> ``active/<name>``."""
        self.init()
        src = self.candidate_dir / name
        if not src.exists():
            raise FileNotFoundError(f"candidate skill not found: {name}")

        dest = self.active_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

        existing = self.get(name) or SkillRecord(
            name=name, status=STATUS_CANDIDATE, created_at=now_iso()
        )
        record = SkillRecord(
            name=name,
            status=STATUS_ACTIVE,
            created_at=existing.created_at,
            version=existing.version,
            retrievals=existing.retrievals,
            adoptions=existing.adoptions,
            failures=existing.failures,
            reviewer=reviewer,
            activate_reason=reason,
            activated_at=now_iso(),
        )
        self._upsert(record)
        return record

    def retire(self, name: str, reason: str = "") -> SkillRecord:
        """``active/<name>`` -> ``retired/<name>``."""
        self.init()
        src = self.active_dir / name
        if not src.exists():
            raise FileNotFoundError(f"active skill not found: {name}")

        dest = self.retired_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

        existing = self.get(name) or SkillRecord(
            name=name, status=STATUS_ACTIVE, created_at=now_iso()
        )
        record = SkillRecord(
            name=name,
            status=STATUS_RETIRED,
            created_at=existing.created_at,
            version=existing.version,
            retrievals=existing.retrievals,
            adoptions=existing.adoptions,
            failures=existing.failures,
            reviewer=existing.reviewer,
            activate_reason=existing.activate_reason,
            activated_at=existing.activated_at,
            retired_at=now_iso(),
            retire_reason=reason,
        )
        self._upsert(record)
        return record

    def reject_candidate(self, name: str, reason: str = "") -> SkillRecord:
        """Reject a candidate: ``candidate/<name>`` -> ``retired/<name>``."""
        self.init()
        src = self.candidate_dir / name
        if not src.exists():
            raise FileNotFoundError(f"candidate skill not found: {name}")

        dest = self.retired_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

        existing = self.get(name) or SkillRecord(
            name=name, status=STATUS_CANDIDATE, created_at=now_iso()
        )
        record = SkillRecord(
            name=name,
            status=STATUS_RETIRED,
            created_at=existing.created_at,
            version=existing.version,
            retrievals=existing.retrievals,
            adoptions=existing.adoptions,
            failures=existing.failures,
            reviewer=existing.reviewer,
            activate_reason=existing.activate_reason,
            retired_at=now_iso(),
            retire_reason=f"rejected: {reason}" if reason else "rejected",
        )
        self._upsert(record)
        return record

    def reject(self, name: str, reason: str = "") -> SkillRecord:
        """Record a rejected draft (audit only — nothing lands in ``candidate/``)."""
        self.init()
        record = SkillRecord(
            name=name, status=STATUS_DRAFT, created_at=now_iso(), reject_reason=reason
        )
        self._upsert(record)
        return record

    # ------------------------------------------------------------------ #
    # Utility statistics (B6)
    # ------------------------------------------------------------------ #

    def on_retrieve(self, name: str) -> SkillRecord:
        """Increment the retrieval counter when a skill enters a prompt.

        Self-healing: an active-dir skill loaded from disk may not be registered
        yet, so a missing record is created as ``active`` rather than dropped.
        """
        record = self.get(name)
        if record is None:
            record = SkillRecord(name=name, status=STATUS_ACTIVE, created_at=now_iso())
        record.retrievals += 1
        self._upsert(record)
        return record

    def on_adopt(self, name: str, success: bool) -> SkillRecord:
        """Record one adoption (self-healing, like :meth:`on_retrieve`).

        ``adoptions`` always increments (total adoptions); ``failures`` only
        increments when ``success`` is False.
        """
        record = self.get(name)
        if record is None:
            record = SkillRecord(name=name, status=STATUS_ACTIVE, created_at=now_iso())
        record.adoptions += 1
        if not success:
            record.failures += 1
        self._upsert(record)
        return record

    # ------------------------------------------------------------------ #
    # Periodic retirement (B6)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_dt(value: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _is_stale(self, record: SkillRecord, stale_weeks: int, now: datetime) -> bool:
        ref = record.activated_at or record.created_at
        dt = self._parse_dt(ref)
        if dt is None:
            return False
        return now - dt >= timedelta(weeks=stale_weeks)

    def periodic_retire(
        self,
        *,
        stale_weeks: int = 4,
        min_adoptions: int = 5,
        min_success_rate: float = 0.3,
        now: Optional[datetime] = None,
    ) -> List[str]:
        """Retire active skills that are stale or perform poorly.

        Returns the names of the skills that were retired.
        """
        now = now or datetime.now(timezone.utc)
        retired: List[str] = []
        for record in self.records(STATUS_ACTIVE):
            if record.retrievals == 0 and self._is_stale(record, stale_weeks, now):
                self.retire(
                    record.name, f"stale: never retrieved in {stale_weeks} weeks"
                )
                retired.append(record.name)
            elif (
                record.adoptions >= min_adoptions
                and record.success_rate < min_success_rate
            ):
                self.retire(
                    record.name,
                    f"success rate {record.success_rate:.2f} below {min_success_rate}",
                )
                retired.append(record.name)
        return retired

    # ------------------------------------------------------------------ #
    # Bridge to Part A
    # ------------------------------------------------------------------ #

    def load_active_skills(self):
        """Discover active skills using Part A's loader (for ``<available_skills>``)."""
        from .skills import discover_skills

        return discover_skills(self.active_dir, "path")
