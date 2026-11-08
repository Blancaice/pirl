"""Command-line helper for inspecting skills, discovery, and the lifecycle.

Usage::

    python -m pirl scan DIR [--source path] [--no-ignore]
    python -m pirl prompt DIR [--file-read-tool read]
    python -m pirl expand DIR SKILL_NAME [ARGS...]

    python -m pirl init [ROOT]
    python -m pirl ls [ROOT] [--status active]
    python -m pirl activate NAME [ROOT] --reason "..." [--reviewer "..."]
    python -m pirl retire NAME [ROOT] --reason "..."
    python -m pirl verify DIR
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .commands import expand_skill_command
from .lifecycle import SkillStore
from .prompt import format_skills_for_prompt
from .proposal import Proposal, verify
from .skills import Skill, discover_skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pirl", description="Skill system")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="discover skills in a directory")
    scan.add_argument("directory")
    scan.add_argument("--source", default="path")
    scan.add_argument("--no-ignore", action="store_true", help="do not honour ignore files")

    prompt = sub.add_parser("prompt", help="render the <available_skills> block")
    prompt.add_argument("directory")
    prompt.add_argument("--source", default="path")
    prompt.add_argument(
        "--file-read-tool", choices=["read", "bash", "indirect"], default="read"
    )

    expand = sub.add_parser("expand", help="expand /skill:name args")
    expand.add_argument("directory")
    expand.add_argument("skill_name")
    expand.add_argument("args", nargs="*")

    init = sub.add_parser("init", help="initialise a skills store (active/candidate/retired)")
    init.add_argument("root", nargs="?", default=None)

    ls = sub.add_parser("ls", help="list registry records")
    ls.add_argument("root", nargs="?", default=None)
    ls.add_argument("--status", choices=["draft", "candidate", "active", "retired"])

    activate = sub.add_parser("activate", help="candidate -> active (human-in-the-loop)")
    activate.add_argument("name")
    activate.add_argument("root", nargs="?", default=None)
    activate.add_argument("--reviewer", default="")
    activate.add_argument("--reason", default="")

    retire = sub.add_parser("retire", help="active -> retired")
    retire.add_argument("name")
    retire.add_argument("root", nargs="?", default=None)
    retire.add_argument("--reason", default="")

    verify_cmd = sub.add_parser("verify", help="run a proposal directory's held-out tests")
    verify_cmd.add_argument("directory")

    return parser


def _discover(directory: str, source: str, respect_ignore: bool) -> tuple[List[Skill], list]:
    result = discover_skills(directory, source, respect_ignore=respect_ignore)
    return result.skills, result.diagnostics


def _cmd_scan(args: argparse.Namespace) -> int:
    skills, diagnostics = _discover(args.directory, args.source, not args.no_ignore)
    for skill in skills:
        print(f"{skill.name}\t{skill.source}\t{skill.path}")
    for diagnostic in diagnostics:
        print(f"[{diagnostic.type}] {diagnostic.message} ({diagnostic.path})", file=sys.stderr)
    print(f"\n{len(skills)} skill(s), {len(diagnostics)} diagnostic(s)", file=sys.stderr)
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    skills, diagnostics = _discover(args.directory, args.source, True)
    output = format_skills_for_prompt(skills, file_read_tool=args.file_read_tool)
    if output:
        print(output)
    for diagnostic in diagnostics:
        print(f"[{diagnostic.type}] {diagnostic.message} ({diagnostic.path})", file=sys.stderr)
    return 0


def _cmd_expand(args: argparse.Namespace) -> int:
    skills, _diagnostics = _discover(args.directory, "path", True)
    text = f"/skill:{args.skill_name}"
    if args.args:
        text += " " + " ".join(args.args)
    print(expand_skill_command(text, skills))
    return 0


def _store(args: argparse.Namespace) -> SkillStore:
    return SkillStore(args.root)


def _cmd_init(args: argparse.Namespace) -> int:
    store = SkillStore(args.root)
    print(f"initialised store at {store.root}")
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    store = _store(args)
    records = store.records(status=args.status)
    for record in records:
        rate = record.success_rate
        print(
            f"{record.status:<10} {record.name:<24} v{record.version} "
            f"retrievals={record.retrievals} adoptions={record.adoptions} "
            f"failures={record.failures} success_rate={rate:.2f}"
        )
    print(f"\n{len(records)} record(s)", file=sys.stderr)
    return 0


def _cmd_activate(args: argparse.Namespace) -> int:
    store = _store(args)
    record = store.activate(args.name, reviewer=args.reviewer, reason=args.reason)
    print(f"activated {record.name} -> active (reviewer={record.reviewer or '-'})")
    return 0


def _cmd_retire(args: argparse.Namespace) -> int:
    store = _store(args)
    record = store.retire(args.name, reason=args.reason)
    print(f"retired {record.name} -> retired (reason={record.retire_reason or '-'})")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    proposal = Proposal.from_dir(args.directory)
    result = verify(proposal)
    for case in result.results:
        mark = "PASS" if case.passed else "FAIL"
        print(f"[{mark}] case_{case.case_index}")
        if not case.passed:
            print(f"  expected: {case.expected!r}")
            print(f"  actual:   {case.actual!r}")
            if case.error:
                print(f"  error:    {case.error}")
    print(
        f"\n{'PASSED' if result.passed else 'FAILED'}: {result.reason or 'all held-out tests passed'}",
        file=sys.stderr,
    )
    return 0 if result.passed else 1


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "scan": _cmd_scan,
        "prompt": _cmd_prompt,
        "expand": _cmd_expand,
        "init": _cmd_init,
        "ls": _cmd_ls,
        "activate": _cmd_activate,
        "retire": _cmd_retire,
        "verify": _cmd_verify,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
