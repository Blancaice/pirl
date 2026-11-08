"""Real ``generate_skill``: turn a run trace into a :class:`Proposal` via a model.

Part B's ``SelfGenLoop`` takes a ``SkillGenerator`` callable; this module
supplies the real one. The model is asked to emit a JSON proposal describing a
skill (name, description, body, optional scripts, held-out tests); that JSON is
parsed and validated into a :class:`Proposal`.

The verification gate (:func:`pirl.proposal.verify`) is deliberately *not*
performed here — the caller runs it afterwards on the held-out tests, so the
skill cannot prove itself by replaying the trace it was distilled from.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from .llm import ChatClient
from .loop import SkillGenerator
from .proposal import CaseSpec, Proposal

PROPOSAL_CONTRACT = (
    "You distill agent runs into reusable skills. "
    "Respond with ONLY a JSON object (no markdown fences, no prose) shaped like:\n"
    "{\n"
    '  "name": "lowercase-hyphenated name",\n'
    '  "description": "one line: what it does and when to use it (<=1024 chars)",\n'
    '  "body": "markdown instructions for the skill",\n'
    '  "disable_model_invocation": false,\n'
    '  "scripts": {"run": "#!/usr/bin/env ...\\n..."},\n'
    '  "tests": [{"input": "...", "expected": "..."}]\n'
    "}\n"
    '"scripts" and "disable_model_invocation" are optional. '
    '"tests" is required: give 3-5 held-out (input, expected) cases that a fresh '
    "run of scripts/run can check independently of the original trace."
)


def _render_trace(trace: Sequence) -> str:
    lines: List[str] = []
    for item in trace:
        if hasattr(item, "name"):
            lines.append(f"- {item.name}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else "(empty)"


def build_generation_messages(trace: Sequence, outcome: str) -> List[Dict[str, Any]]:
    """Build the chat messages that ask the model for a skill proposal."""
    user = (
        "The agent just finished a run with outcome "
        f"{outcome!r}.\n\n"
        f"Tool-call trace:\n{_render_trace(trace)}\n\n"
        "Distill this into a reusable, verifiable skill. Write a small "
        "scripts/run entrypoint and 3-5 held-out test cases so the skill can be "
        "verified on fresh inputs, not by replaying this trace."
    )
    return [
        {"role": "system", "content": PROPOSAL_CONTRACT},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> Any:
    """Pull the first JSON object out of (possibly fenced) model output."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def parse_proposal_json(text: str) -> Proposal:
    """Parse model output into a :class:`Proposal`, validating required fields.

    Raises ``ValueError`` with a specific message when the output is unusable.
    """
    try:
        data = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse proposal JSON: {error}") from error

    if not isinstance(data, dict):
        raise ValueError("proposal JSON must be an object")

    name = data.get("name")
    description = data.get("description")
    body = data.get("body")
    if not isinstance(name, str) or not name:
        raise ValueError("proposal missing 'name'")
    if not isinstance(description, str) or not description:
        raise ValueError("proposal missing 'description'")
    if not isinstance(body, str):
        raise ValueError("proposal missing 'body'")

    scripts = data.get("scripts") or {}
    if not isinstance(scripts, dict):
        raise ValueError("proposal 'scripts' must be an object")

    tests_raw = data.get("tests") or []
    if not isinstance(tests_raw, list):
        raise ValueError("proposal 'tests' must be an array")
    tests = []
    for item in tests_raw:
        if not isinstance(item, dict):
            raise ValueError("each test case must be an object")
        tests.append(
            CaseSpec(
                input=str(item.get("input", "")),
                expected=str(item.get("expected", "")),
            )
        )

    return Proposal(
        name=name,
        description=description,
        body=body,
        tests=tests,
        scripts={str(k): str(v) for k, v in scripts.items()},
        disable_model_invocation=data.get("disable_model_invocation") is True,
    )


def generator_from_client(client: ChatClient) -> SkillGenerator:
    """Return a ``SkillGenerator`` backed by a real model client."""

    def generate(trace: Sequence, outcome: str) -> Proposal:
        messages = build_generation_messages(trace, outcome)
        completion = client.complete(messages)
        return parse_proposal_json(completion.text)

    return generate
