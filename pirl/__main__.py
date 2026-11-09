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

    python -m pirl run "TASK" [--cwd DIR] [--max-steps N]
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import ui
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

    reject = sub.add_parser("reject", help="reject a candidate (candidate -> retired)")
    reject.add_argument("name")
    reject.add_argument("root", nargs="?", default=None)
    reject.add_argument("--reason", default="")

    review = sub.add_parser("review", help="interactively review candidate skills")
    review.add_argument("root", nargs="?", default=None)

    verify_cmd = sub.add_parser("verify", help="run a proposal directory's held-out tests")
    verify_cmd.add_argument("directory")

    run = sub.add_parser("run", help="run one task in a workspace")
    run.add_argument("task")
    run.add_argument("--cwd", default=".")
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument(
        "--verify",
        default=None,
        help="command to deterministically verify a finish(success=true)",
    )

    restore_cmd = sub.add_parser(
        "restore", help="rollback workspace to a run's pre-run git snapshot"
    )
    restore_cmd.add_argument("run_id")
    restore_cmd.add_argument("--cwd", default=".")

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


def _cmd_reject(args: argparse.Namespace) -> int:
    store = _store(args)
    record = store.reject_candidate(args.name, reason=args.reason)
    print(f"rejected {record.name} -> retired (reason={record.retire_reason or '-'})")
    return 0


def _candidate_prompt(store: SkillStore, name: str) -> str:
    """Show one candidate and ask the user; returns activate/reject/skip."""
    skill_md = store.candidate_dir / name / "SKILL.md"
    try:
        body = skill_md.read_text(encoding="utf-8")
    except OSError:
        body = "(unreadable)"
    print(f"\n--- candidate skill「{name}」---", file=sys.stderr)
    print(body[:2000], file=sys.stderr)
    print(f"\n(full file: {skill_md})", file=sys.stderr)
    return ui.select(
        f"Review candidate「{name}」:",
        {"a": "activate", "r": "reject", "s": "skip"},
        default="s",
    )


def _cmd_review(args: argparse.Namespace) -> int:
    store = _store(args)
    candidates = store.records(status="candidate")
    if not candidates:
        print("no candidate skills", file=sys.stderr)
        return 0
    for record in candidates:
        action = _candidate_prompt(store, record.name)
        if action == "activate":
            store.activate(record.name, reason="approved via review")
            print(f"activated {record.name}", file=sys.stderr)
        elif action == "reject":
            store.reject_candidate(record.name, reason="rejected via review")
            print(f"rejected {record.name}", file=sys.stderr)
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    from .git_snapshot import GitSnapshot, restore
    from .run_history import RunStore

    record = RunStore(args.cwd).read_run(args.run_id)
    if record is None or record.git is None:
        print(f"no git snapshot for run {args.run_id}", file=sys.stderr)
        return 1
    snap = GitSnapshot.from_dict(record.git)
    if not ui.confirm(
        f"restore workspace to {snap.head[:8]}? This discards uncommitted changes.",
        default=False,
    ):
        print("cancelled", file=sys.stderr)
        return 1
    restore(args.cwd, snap)
    print(f"restored to {snap.head[:8]}", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .agent import Agent, FinishCalled, StepStart, ToolCalled
    from .env import load_dotenv
    from .generate import generator_from_client
    from .git_snapshot import snapshot
    from .lifecycle import SkillStore
    from .llm import LLMConfigError, LLMError, client_from_env
    from .loop import SelfGenLoop
    from .run_memory import read_memory, write_memory
    from .run_history import RunRecord, RunStore, new_run_id, now_iso
    from .skills import discover_skills
    from .tools import ToolRegistry, builtin_tools

    def render_step(event) -> str:
        match event:
            case StepStart(step, max_steps):
                return f"[step {step}/{max_steps}] asking model..."
            case ToolCalled(name, arguments):
                return f"  → {name} {arguments}"
            case FinishCalled(arguments):
                return f"  → finish({arguments})"
            case _:
                return str(event)

    load_dotenv()
    try:
        client = client_from_env()
    except LLMConfigError as error:
        print(f"pirl: {error}", file=sys.stderr)
        print("set PIRL_BASE_URL / PIRL_API_KEY / PIRL_MODEL (see README)", file=sys.stderr)
        return 2

    store = SkillStore()
    skills = list(store.load_active_skills().skills)
    # project-local skills: any SKILL.md under <cwd>/.pirl/skills (deduped by name)
    active_names = {s.name for s in skills}
    for skill in discover_skills(Path(args.cwd) / ".pirl" / "skills", "project").skills:
        if skill.name not in active_names:
            skills.append(skill)
            active_names.add(skill.name)

    registry = ToolRegistry(args.cwd, builtin_tools())
    loop = SelfGenLoop(store, generator=generator_from_client(client))
    agent = Agent(
        client,
        skills=skills,
        tools=registry,
        self_gen=loop,
        skill_store=store,
        memory=read_memory(args.cwd),
        max_steps=args.max_steps,
        verify_command=args.verify,
        on_step=lambda event: print(render_step(event), file=sys.stderr, flush=True),
    )
    snap = snapshot(args.cwd)
    started_at = now_iso()
    try:
        result = agent.run(args.task)
    except LLMError as error:
        print(f"pirl: {error}", file=sys.stderr)
        return 1

    write_memory(args.cwd, result.outcome, result.answer or result.message)
    RunStore(args.cwd).write_run(
        RunRecord(
            run_id=new_run_id(),
            task=args.task,
            outcome=result.outcome,
            answer=result.answer or result.message,
            started_at=started_at,
            trace=[{"tool": tc.name, "args": tc.args} for tc in result.trace],
            usage={
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            git=snap.to_dict() if snap else None,
        )
    )

    for step, block in enumerate(result.reasoning, 1):
        print(f"--- thinking (step {step}) ---", file=sys.stderr)
        print(block, file=sys.stderr)
    print(result.answer or f"[{result.outcome}] {result.message}", flush=True)
    usage = result.usage
    print(
        f"[usage] ↑{usage.prompt_tokens} ↓{usage.completion_tokens} Σ{usage.total_tokens}",
        file=sys.stderr,
    )
    if result.loop_result is not None:
        loop_result = result.loop_result
        print(
            f"[self-gen] {loop_result.status}: {loop_result.message}",
            file=sys.stderr,
        )
        if loop_result.status == "candidate" and loop_result.name:
            action = _candidate_prompt(store, loop_result.name)
            if action == "activate":
                store.activate(loop_result.name, reason="approved interactively")
                print(f"activated {loop_result.name}", file=sys.stderr)
            elif action == "reject":
                store.reject_candidate(loop_result.name, reason="rejected interactively")
                print(f"rejected {loop_result.name}", file=sys.stderr)

    return 0 if result.outcome == "success" else 1


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
        "reject": _cmd_reject,
        "review": _cmd_review,
        "restore": _cmd_restore,
        "verify": _cmd_verify,
        "run": _cmd_run,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
