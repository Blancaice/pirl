"""Trigger sources, proposal structure, and the verification gate (B3/B4/B5).

B3 — when to propose a skill:

    * the run ended in failure (post-mortem)
    * the same tool sequence repeated ``>= n`` times
    * the user explicitly asked (``/skill-extract``)

B4 — a proposal is a skill *directory* that must carry tests:

    candidate/<name>/
    ├── SKILL.md          # description = routing words, body = instructions
    ├── scripts/          # optional runnable scripts (``scripts/run`` is the entrypoint)
    └── tests/            # 3~5 held-out (input, expected) examples
        └── case_1.{in,out}

B5 — the verification gate runs the *held-out* examples, never the successful
trace, so a skill cannot "prove itself" by replaying what just worked.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# B3 — trigger sources
# --------------------------------------------------------------------------- #

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"

# Canonical runnable entrypoint that verification exercises.
ENTRYPOINT = "scripts/run"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation in a run trace."""

    name: str
    args: str = ""


def _tool_name(tool) -> str:
    return tool.name if isinstance(tool, ToolCall) else str(tool)


def repeated_tool_pattern(trace: Sequence, n: int = 3) -> bool:
    """True when the same contiguous tool sequence repeats ``n`` times in a row.

    Examples (n=3)::

        ["bash", "bash", "bash"]                     -> True  (block of length 1)
        ["bash", "read", "bash", "read", "bash", "read"] -> True  (block of length 2)
        ["bash", "read"]                             -> False
    """
    names = [_tool_name(t) for t in trace]
    max_block = len(names) // n
    for length in range(1, max_block + 1):
        for start in range(0, len(names) - length * n + 1):
            block = names[start : start + length]
            if names[start : start + length * n] == block * n:
                return True
    return False


def should_propose(
    trace: Sequence,
    outcome: str,
    *,
    user_requested: bool = False,
    n: int = 3,
) -> bool:
    """B3: decide whether to propose a new skill for this run."""
    return (
        outcome == OUTCOME_FAILURE
        or repeated_tool_pattern(trace, n)
        or user_requested
    )


# --------------------------------------------------------------------------- #
# B4 — proposal structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseSpec:
    """One held-out (input, expected) example."""

    input: str = ""
    expected: str = ""


@dataclass
class Proposal:
    """A candidate skill generated from a run trace.

    ``scripts`` maps relative file names (under ``scripts/``) to their content.
    ``tests`` are the held-out examples written as ``tests/case_N.in`` /
    ``tests/case_N.out``.
    """

    name: str
    description: str
    body: str
    tests: List[CaseSpec] = field(default_factory=list)
    scripts: Dict[str, str] = field(default_factory=dict)
    disable_model_invocation: bool = False

    def render_skill_md(self) -> str:
        """Render the full ``SKILL.md`` (frontmatter + body)."""
        fm = f"name: {self.name}\ndescription: {self.description}"
        if self.disable_model_invocation:
            fm += "\ndisable-model-invocation: true"
        return f"---\n{fm}\n---\n{self.body}\n"

    def materialize(self, target: str | Path) -> Path:
        """Write this proposal to disk as a skill directory and return it."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(self.render_skill_md(), encoding="utf-8")

        if self.scripts:
            scripts_dir = target / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for filename, content in self.scripts.items():
                path = scripts_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)

        if self.tests:
            tests_dir = target / "tests"
            tests_dir.mkdir(exist_ok=True)
            for index, case in enumerate(self.tests, start=1):
                (tests_dir / f"case_{index}.in").write_text(case.input, encoding="utf-8")
                (tests_dir / f"case_{index}.out").write_text(case.expected, encoding="utf-8")

        return target

    @classmethod
    def from_dir(cls, directory: str | Path) -> "Proposal":
        """Load a proposal from an existing skill directory (for re-verification)."""
        from .frontmatter import parse_frontmatter

        directory = Path(directory)
        skill_md = directory / "SKILL.md"
        if not skill_md.is_file():
            raise FileNotFoundError(f"SKILL.md not found in {directory}")

        parsed = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        frontmatter = parsed["frontmatter"]
        name = frontmatter.get("name") or directory.name
        description = frontmatter.get("description") or ""
        body = parsed["body"]

        scripts: Dict[str, str] = {}
        scripts_dir = directory / "scripts"
        if scripts_dir.is_dir():
            for path in sorted(scripts_dir.rglob("*")):
                if path.is_file():
                    scripts[str(path.relative_to(scripts_dir))] = path.read_text(
                        encoding="utf-8"
                    )

        tests: List[CaseSpec] = []
        tests_dir = directory / "tests"
        if tests_dir.is_dir():
            for in_path in sorted(tests_dir.glob("*.in")):
                out_path = in_path.with_suffix(".out")
                tests.append(
                    CaseSpec(
                        input=in_path.read_text(encoding="utf-8"),
                        expected=(
                            out_path.read_text(encoding="utf-8")
                            if out_path.is_file()
                            else ""
                        ),
                    )
                )

        return cls(
            name=name,
            description=description,
            body=body,
            tests=tests,
            scripts=scripts,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        )


# --------------------------------------------------------------------------- #
# B5 — verification gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseResult:
    """Outcome of running one held-out test case."""

    case_index: int
    passed: bool
    expected: str
    actual: str
    error: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Result of the verification gate."""

    passed: bool
    results: List[CaseResult]
    reason: str = ""


def run_case(entrypoint: Path, case: CaseSpec, *, timeout: int = 10) -> tuple[str, str]:
    """Run the entrypoint script with ``case.input`` on stdin; return (stdout, error)."""
    try:
        proc = subprocess.run(
            [str(entrypoint)],
            input=case.input,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return "", f"entrypoint not executable: {entrypoint}"
    except subprocess.TimeoutExpired:
        return "", f"timed out after {timeout}s"
    except OSError as exc:
        return "", str(exc)

    if proc.returncode != 0:
        return proc.stdout, proc.stderr.strip() or f"exit code {proc.returncode}"
    return proc.stdout, ""


def verify(
    proposal: Proposal,
    *,
    work_dir: str | Path | None = None,
    entrypoint: str = ENTRYPOINT,
    timeout: int = 10,
) -> VerificationResult:
    """Run the proposal's held-out examples independently.

    The proposal is materialized into ``work_dir`` (a temp dir when omitted) and
    every ``tests/case_N.in`` is fed to ``scripts/run``; its stdout must match
    ``tests/case_N.out``. Never replays the originating trace.
    """
    if not proposal.tests:
        return VerificationResult(False, [], "proposal carries no tests")

    owns_dir = work_dir is None
    work_dir = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="pirl-verify-")
    )

    try:
        proposal.materialize(work_dir)

        entry = work_dir / entrypoint
        if not entry.is_file():
            return VerificationResult(
                False, [], f"no runnable entrypoint at {entrypoint}"
            )

        results: List[CaseResult] = []
        for index, case in enumerate(proposal.tests, start=1):
            actual, error = run_case(entry, case, timeout=timeout)
            passed = error == "" and actual.strip() == case.expected.strip()
            results.append(
                CaseResult(
                    case_index=index,
                    passed=passed,
                    expected=case.expected,
                    actual=actual,
                    error=error,
                )
            )

        all_passed = all(r.passed for r in results)
        reason = "" if all_passed else "held-out tests failed"
        return VerificationResult(all_passed, results, reason)
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
