import json

import pytest

from pirl.generate import (
    build_generation_messages,
    generator_from_client,
    parse_proposal_json,
)
from pirl.proposal import CaseSpec, Proposal, ToolCall

PROPOSAL = {
    "name": "upper-text",
    "description": "Uppercase text. Use when converting text to uppercase.",
    "body": "# Upper\nRun scripts/run with text on stdin.",
    "scripts": {"run": "#!/usr/bin/env python3\nimport sys\nprint(sys.stdin.read().upper(), end='')\n"},
    "tests": [
        {"input": "hello\n", "expected": "HELLO\n"},
        {"input": "x\n", "expected": "X\n"},
    ],
}


def test_parse_proposal_json_plain():
    proposal = parse_proposal_json(json.dumps(PROPOSAL))
    assert proposal.name == "upper-text"
    assert proposal.description.startswith("Uppercase")
    assert proposal.tests == [CaseSpec("hello\n", "HELLO\n"), CaseSpec("x\n", "X\n")]
    assert proposal.scripts["run"].startswith("#!/usr/bin/env")


def test_parse_proposal_json_fenced():
    text = "```json\n" + json.dumps(PROPOSAL) + "\n```"
    proposal = parse_proposal_json(text)
    assert proposal.name == "upper-text"
    assert len(proposal.tests) == 2


def test_parse_proposal_json_tolerates_surrounding_prose():
    text = "Here is the skill:\n" + json.dumps(PROPOSAL) + "\nDone."
    proposal = parse_proposal_json(text)
    assert proposal.name == "upper-text"


def test_parse_proposal_json_requires_name():
    with pytest.raises(ValueError, match="name"):
        parse_proposal_json(json.dumps({"description": "d", "body": "b"}))


def test_parse_proposal_json_requires_description():
    with pytest.raises(ValueError, match="description"):
        parse_proposal_json(json.dumps({"name": "n", "body": "b"}))


def test_parse_proposal_json_requires_body():
    with pytest.raises(ValueError, match="body"):
        parse_proposal_json(json.dumps({"name": "n", "description": "d"}))


def test_parse_proposal_json_no_json_object():
    with pytest.raises(ValueError, match="no JSON"):
        parse_proposal_json("just some prose")


def test_parse_proposal_json_empty_tests_ok():
    proposal = parse_proposal_json(
        json.dumps({"name": "n", "description": "d", "body": "b"})
    )
    assert proposal.tests == []
    assert proposal.scripts == {}


def test_build_generation_messages_renders_trace():
    messages = build_generation_messages(
        [ToolCall("bash", ""), ToolCall("read_file", ""), "plain"], "failure"
    )
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "'failure'" in user
    assert "bash" in user
    assert "read_file" in user
    assert "plain" in user


def test_generator_from_client_builds_proposal(monkeypatch):
    class _Stub:
        def complete(self, messages):
            assert messages[0]["content"]  # system prompt present
            return type("C", (), {"text": json.dumps(PROPOSAL)})()

    generate = generator_from_client(_Stub())
    proposal = generate([ToolCall("bash", "")], "failure")
    assert isinstance(proposal, Proposal)
    assert proposal.name == "upper-text"
