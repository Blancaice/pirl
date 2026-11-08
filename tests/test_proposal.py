import pytest

from pirl.proposal import (
    CaseSpec,
    Proposal,
    ToolCall,
    repeated_tool_pattern,
    should_propose,
    verify,
)


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


# --------------------------------------------------------------------------- #
# B3 — trigger sources
# --------------------------------------------------------------------------- #


def test_should_propose_on_failure():
    assert should_propose([], "failure") is True


def test_should_propose_on_repeated_pattern():
    trace = [ToolCall("bash"), ToolCall("bash"), ToolCall("bash")]
    assert should_propose(trace, "success") is True


def test_should_propose_on_user_request():
    assert should_propose([], "success", user_requested=True) is True


def test_should_not_propose_clean_success():
    assert should_propose([ToolCall("bash"), ToolCall("read")], "success") is False


def test_repeated_tool_pattern_single_tool():
    assert repeated_tool_pattern(["bash", "bash", "bash"], 3) is True
    assert repeated_tool_pattern(["bash", "bash"], 3) is False


def test_repeated_tool_pattern_repeating_pair():
    trace = ["bash", "read", "bash", "read", "bash", "read"]
    assert repeated_tool_pattern(trace, 3) is True


def test_repeated_tool_pattern_accepts_toolcall_objects():
    trace = [ToolCall("a"), ToolCall("a"), ToolCall("a")]
    assert repeated_tool_pattern(trace, 3) is True


def test_repeated_tool_pattern_no_repetition():
    assert repeated_tool_pattern(["a", "b", "c", "d"], 3) is False


# --------------------------------------------------------------------------- #
# B4 — proposal structure
# --------------------------------------------------------------------------- #


def test_render_skill_md_has_frontmatter_and_body():
    proposal = make_proposal()
    md = proposal.render_skill_md()
    assert md.startswith("---\nname: upper\ndescription: Convert input to uppercase.\n---\n")
    assert "# Uppercase" in md


def test_materialize_writes_full_layout(tmp_path):
    proposal = make_proposal()
    target = tmp_path / "upper"
    proposal.materialize(target)

    assert (target / "SKILL.md").is_file()
    assert (target / "scripts" / "run").is_file()
    assert (target / "tests" / "case_1.in").read_text() == "hello\n"
    assert (target / "tests" / "case_1.out").read_text() == "HELLO\n"
    assert (target / "scripts" / "run").stat().st_mode & 0o111  # executable


def test_materialize_roundtrip_via_from_dir(tmp_path):
    proposal = make_proposal()
    target = tmp_path / "upper"
    proposal.materialize(target)

    loaded = Proposal.from_dir(target)
    assert loaded.name == "upper"
    assert loaded.description == "Convert input to uppercase."
    assert loaded.body == "# Uppercase"
    assert loaded.tests == [CaseSpec("hello\n", "HELLO\n")]
    assert "run" in loaded.scripts


# --------------------------------------------------------------------------- #
# B5 — verification gate
# --------------------------------------------------------------------------- #


def test_verify_passes_correct_skill(tmp_path):
    result = verify(make_proposal(), work_dir=tmp_path / "w")
    assert result.passed is True
    assert all(r.passed for r in result.results)
    assert result.reason == ""


def test_verify_rejects_fake_skill(tmp_path):
    # A "fake" skill whose script always outputs the wrong thing must be caught.
    fake = make_proposal(
        name="fake",
        script="#!/usr/bin/env python3\nprint('WRONG')\n",
        tests=[CaseSpec("hello\n", "HELLO\n")],
    )
    result = verify(fake, work_dir=tmp_path / "w")
    assert result.passed is False
    assert result.results[0].passed is False
    assert result.results[0].actual.strip() == "WRONG"
    assert result.reason == "held-out tests failed"


def test_verify_rejects_proposal_without_tests(tmp_path):
    proposal = make_proposal(tests=[])
    result = verify(proposal, work_dir=tmp_path / "w")
    assert result.passed is False
    assert result.reason == "proposal carries no tests"


def test_verify_rejects_proposal_without_entrypoint(tmp_path):
    proposal = Proposal(
        name="no-script",
        description="nothing runnable",
        body="# nothing",
        tests=[CaseSpec("x", "y")],
        scripts={},
    )
    result = verify(proposal, work_dir=tmp_path / "w")
    assert result.passed is False
    assert "no runnable entrypoint" in result.reason


def test_verify_uses_temp_dir_when_work_dir_omitted():
    result = verify(make_proposal())
    assert result.passed is True


def test_verify_multiple_cases_all_must_pass(tmp_path):
    script = "#!/usr/bin/env python3\nimport sys\nd = sys.stdin.read()\nprint('yes' if 'ok' in d else 'no', end='')\n"
    proposal = Proposal(
        name="multi",
        description="checks for ok",
        body="# multi",
        scripts={"run": script},
        tests=[CaseSpec("ok\n", "yes"), CaseSpec("nope\n", "yes")],
    )
    result = verify(proposal, work_dir=tmp_path / "w")
    assert result.passed is False
    assert [r.passed for r in result.results] == [True, False]
