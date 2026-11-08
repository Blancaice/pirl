import json
from datetime import datetime, timedelta, timezone

import pytest

from pirl.lifecycle import (
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_DRAFT,
    STATUS_RETIRED,
    SkillRecord,
    SkillStore,
)
from pirl.proposal import Proposal, CaseSpec


def make_proposal(name="upper", description="Uppercase input."):
    return Proposal(
        name=name,
        description=description,
        body="# Uppercase\n",
        scripts={"run": "#!/usr/bin/env python3\nimport sys\nprint(sys.stdin.read().upper(), end='')\n"},
        tests=[CaseSpec("hello\n", "HELLO\n")],
    )


# --------------------------------------------------------------------------- #
# B1/B2 — directory layout + registry
# --------------------------------------------------------------------------- #


def test_init_creates_layout_and_registry(tmp_path):
    store = SkillStore(tmp_path / "skills")
    assert (tmp_path / "skills" / "active").is_dir()
    assert (tmp_path / "skills" / "candidate").is_dir()
    assert (tmp_path / "skills" / "retired").is_dir()
    assert (tmp_path / "skills" / "skill-registry.json").is_file()


def test_registry_persists_across_instances(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)
    store.promote_to_candidate(make_proposal(), draft)

    # A fresh instance re-reads the registry from disk.
    store2 = SkillStore(root)
    assert store2.get("upper").status == STATUS_CANDIDATE
    assert store2.get("upper").version == 1


def test_registry_is_valid_json(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.reject("nope", "bad tests")
    data = json.loads((tmp_path / "skills" / "skill-registry.json").read_text())
    assert "skills" in data
    assert data["skills"]["nope"]["status"] == "draft"


# --------------------------------------------------------------------------- #
# Lifecycle transitions (B1)
# --------------------------------------------------------------------------- #


def test_promote_to_candidate(tmp_path):
    store = SkillStore(tmp_path / "skills")
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)

    record = store.promote_to_candidate(make_proposal(), draft)

    assert record.status == STATUS_CANDIDATE
    assert (store.candidate_dir / "upper" / "SKILL.md").is_file()
    assert not draft.exists()  # draft was moved


def test_activate_moves_candidate_to_active(tmp_path):
    store = SkillStore(tmp_path / "skills")
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)
    store.promote_to_candidate(make_proposal(), draft)

    record = store.activate("upper", reviewer="blanca", reason="looks good")

    assert record.status == STATUS_ACTIVE
    assert record.reviewer == "blanca"
    assert record.activate_reason == "looks good"
    assert record.activated_at
    assert (store.active_dir / "upper" / "SKILL.md").is_file()
    assert not (store.candidate_dir / "upper").exists()


def test_retire_moves_active_to_retired(tmp_path):
    store = SkillStore(tmp_path / "skills")
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)
    store.promote_to_candidate(make_proposal(), draft)
    store.activate("upper", reviewer="blanca", reason="ok")

    record = store.retire("upper", reason="no longer used")

    assert record.status == STATUS_RETIRED
    assert record.retire_reason == "no longer used"
    assert record.retired_at
    assert (store.retired_dir / "upper" / "SKILL.md").is_file()
    assert not (store.active_dir / "upper").exists()


def test_revise_bumps_version(tmp_path):
    store = SkillStore(tmp_path / "skills")
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)
    first = store.promote_to_candidate(make_proposal(), draft)
    assert first.version == 1

    draft2 = tmp_path / "draft2"
    make_proposal().materialize(draft2)
    second = store.promote_to_candidate(make_proposal(), draft2)
    assert second.version == 2
    assert second.status == STATUS_CANDIDATE


def test_reject_records_draft_audit_without_candidate_dir(tmp_path):
    store = SkillStore(tmp_path / "skills")
    record = store.reject("bad", "held-out tests failed")
    assert record.status == STATUS_DRAFT
    assert record.reject_reason == "held-out tests failed"
    assert not (store.candidate_dir / "bad").exists()


def test_activate_missing_candidate_raises(tmp_path):
    store = SkillStore(tmp_path / "skills")
    with pytest.raises(FileNotFoundError):
        store.activate("missing")


# --------------------------------------------------------------------------- #
# Utility statistics (B6)
# --------------------------------------------------------------------------- #


def test_on_retrieve_increments(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.reject("upper", "x")  # seed a record
    store.on_retrieve("upper")
    store.on_retrieve("upper")
    assert store.get("upper").retrievals == 2


def test_on_adopt_increments_and_tracks_failures(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.reject("upper", "x")
    store.on_adopt("upper", True)
    store.on_adopt("upper", False)
    record = store.get("upper")
    assert record.adoptions == 2
    assert record.failures == 1
    assert record.success_rate == 0.5


def test_success_rate_no_adoptions_is_one(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.reject("upper", "x")
    assert store.get("upper").success_rate == 1.0


# --------------------------------------------------------------------------- #
# Periodic retirement (B6)
# --------------------------------------------------------------------------- #


def _seed_active(store, name, *, retrievals=0, adoptions=0, failures=0, activated_at=""):
    draft = store.root.parent / f"draft-{name}"
    make_proposal(name=name).materialize(draft)
    store.promote_to_candidate(make_proposal(name=name), draft)
    store.activate(name, reviewer="tester", reason="test")
    record = store.get(name)
    record.retrievals = retrievals
    record.adoptions = adoptions
    record.failures = failures
    if activated_at:
        record.activated_at = activated_at
    store._upsert(record)


def test_periodic_retire_never_retrieved_stale_skill(tmp_path):
    store = SkillStore(tmp_path / "skills")
    old = (datetime.now(timezone.utc) - timedelta(weeks=5)).isoformat()
    _seed_active(store, "stale", retrievals=0, activated_at=old)

    retired = store.periodic_retire(now=datetime.now(timezone.utc))

    assert retired == ["stale"]
    assert store.get("stale").status == STATUS_RETIRED
    assert (store.retired_dir / "stale").is_dir()


def test_periodic_retire_low_success_rate(tmp_path):
    store = SkillStore(tmp_path / "skills")
    _seed_active(store, "flaky", retrievals=10, adoptions=10, failures=8)

    retired = store.periodic_retire(now=datetime.now(timezone.utc))

    assert retired == ["flaky"]
    assert store.get("flaky").status == STATUS_RETIRED


def test_periodic_retire_keeps_healthy_and_recent(tmp_path):
    store = SkillStore(tmp_path / "skills")
    _seed_active(store, "healthy", retrievals=10, adoptions=10, failures=0)
    recent = datetime.now(timezone.utc).isoformat()
    _seed_active(store, "recent", retrievals=0, activated_at=recent)

    retired = store.periodic_retire(now=datetime.now(timezone.utc))

    assert retired == []
    assert store.get("healthy").status == STATUS_ACTIVE
    assert store.get("recent").status == STATUS_ACTIVE  # not stale yet


# --------------------------------------------------------------------------- #
# Bridge to Part A
# --------------------------------------------------------------------------- #


def test_load_active_skills_uses_part_a_loader(tmp_path):
    store = SkillStore(tmp_path / "skills")
    draft = tmp_path / "draft"
    make_proposal().materialize(draft)
    store.promote_to_candidate(make_proposal(), draft)
    store.activate("upper", reviewer="blanca", reason="ok")

    result = store.load_active_skills()
    assert [s.name for s in result.skills] == ["upper"]
