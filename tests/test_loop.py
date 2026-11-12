import pytest

from pirl.lifecycle import STATUS_ACTIVE, STATUS_CANDIDATE, STATUS_DRAFT, SkillStore
from pirl.plugins.self_generation import SelfGenLoop
from pirl.proposal import CaseSpec, Proposal, ToolCall


def make_proposal(name="upper", script=None, tests=None):
    return Proposal(
        name=name,
        description="Convert input to uppercase.",
        body="# Uppercase\n",
        scripts={
            "run": script
            if script is not None
            else "#!/usr/bin/env python3\nimport sys\nprint(sys.stdin.read().upper(), end='')\n"
        },
        tests=[CaseSpec("hello\n", "HELLO\n")] if tests is None else tests,
    )


@pytest.fixture
def store(tmp_path):
    return SkillStore(tmp_path / "skills")


# --------------------------------------------------------------------------- #
# Step 5 — self-generation loop
# --------------------------------------------------------------------------- #


def test_clean_success_skips_proposal(store):
    loop = SelfGenLoop(store, generator=lambda t, o: make_proposal())
    result = loop.on_run_end([ToolCall("read")], "success")
    assert result.proposed is False
    assert result.status == "skipped"
    assert store.candidate_dir.exists() and not any(store.candidate_dir.iterdir())


def test_failed_run_produces_candidate(store):
    loop = SelfGenLoop(store, generator=lambda t, o: make_proposal())
    result = loop.on_run_end([], "success", user_requested=True)

    assert result.proposed is True
    assert result.status == "candidate"
    assert result.name == "upper"
    assert result.verification.passed is True
    assert store.get("upper").status == STATUS_CANDIDATE
    assert (store.candidate_dir / "upper" / "SKILL.md").is_file()
    assert (store.candidate_dir / "upper" / "tests" / "case_1.in").is_file()


def test_fake_skill_is_rejected_before_candidate(store):
    fake = make_proposal(
        name="fake",
        script="#!/usr/bin/env python3\nprint('WRONG')\n",
        tests=[CaseSpec("hello\n", "HELLO\n")],
    )
    loop = SelfGenLoop(store, generator=lambda t, o: fake)
    result = loop.on_run_end([], "success", proposal=fake)

    assert result.status == "rejected"
    assert not (store.candidate_dir / "fake").exists()  # 验证失败不进入候选
    record = store.get("fake")
    assert record.status == STATUS_DRAFT
    assert record.reject_reason == "held-out tests failed"


def test_trigger_without_generator_is_error(store):
    loop = SelfGenLoop(store)  # no generator
    result = loop.on_run_end([], "success", user_requested=True)
    assert result.status == "error"


def test_explicit_proposal_bypasses_generator(store):
    loop = SelfGenLoop(store)
    result = loop.on_run_end(
        [], "success", proposal=make_proposal()
    )
    assert result.status == "candidate"
    assert store.get("upper").status == STATUS_CANDIDATE


def test_full_cycle_to_active(store):
    loop = SelfGenLoop(store, generator=lambda t, o: make_proposal())
    result = loop.on_run_end([], "success", user_requested=True)
    assert result.status == "candidate"

    record = loop.activate("upper", reviewer="blanca", reason="approved")
    assert record.status == STATUS_ACTIVE
    assert (store.active_dir / "upper" / "SKILL.md").is_file()
    assert not (store.candidate_dir / "upper").exists()


# --------------------------------------------------------------------------- #
# Step 6 — utility + retirement through the loop
# --------------------------------------------------------------------------- #


def test_retrieve_and_adopt_flow(store):
    loop = SelfGenLoop(store, generator=lambda t, o: make_proposal())
    loop.on_run_end([], "success", user_requested=True)
    loop.activate("upper", reviewer="blanca", reason="ok")

    loop.on_retrieve("upper")
    loop.on_retrieve("upper")
    loop.on_adopt("upper", True)

    record = store.get("upper")
    assert record.retrievals == 2
    assert record.adoptions == 1
    assert record.failures == 0


def test_bookkeeping_self_heals_missing_record(store):
    loop = SelfGenLoop(store)
    record = loop.on_retrieve("nope")
    assert record is not None
    assert record.status == "active"
    assert record.retrievals == 1
    adopted = loop.on_adopt("nope", True)
    assert adopted.adoptions == 1
    assert adopted.failures == 0
