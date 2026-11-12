"""Skill model, validation, and discovery (Part A2/A3/A4).

Ported from pi's ``core/skills.ts``. Kept faithful to the Agent Skills spec and
to pi's discovery rules:

* a directory containing ``SKILL.md`` is a skill root — do not recurse into it
* ``.gitignore`` / ``.ignore`` / ``.fdignore`` are honoured
* dot-directories are skipped; Python adds ``__pycache__``, ``.venv``, ``.git``
  and ``node_modules`` to the default skip list
* symlinks are followed but deduplicated by ``realpath``
* name collisions: first one wins, later ones emit a warning and are dropped
* validation failures emit warnings (never fatal); a missing ``description``
  causes the skill to be skipped entirely
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..frontmatter import parse_frontmatter
from ..ignore import IgnoreMatcher

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

NAME_RE = re.compile(r"^[a-z0-9-]+$")

IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

# Directories that are always skipped during discovery. ``node_modules`` mirrors
# pi; the rest are Python-specific equivalents of what pi skips for Node.
ALWAYS_SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".git", ".venv", ".pytest_cache"})

CONFIG_DIR_NAME = ".pirl"


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Collision:
    """Details of a name collision between two skills."""

    resource_type: str
    name: str
    winner_path: str
    loser_path: str


@dataclass(frozen=True)
class Diagnostic:
    """A non-fatal discovery/validation message."""

    type: str  # "warning" | "collision"
    message: str
    path: str = ""
    collision: Optional[Collision] = None


# --------------------------------------------------------------------------- #
# Data model (A2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Skill:
    """A first-class skill, as discovered from ``SKILL.md`` on disk."""

    name: str
    description: str
    path: str  # absolute path to SKILL.md
    base_dir: str  # parent directory of SKILL.md (base for relative paths)
    source: str  # "user" | "project" | "path"
    disable_model_invocation: bool = False


@dataclass
class LoadSkillsResult:
    skills: List[Skill] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation (A4)
# --------------------------------------------------------------------------- #


def validate_name(name: str) -> List[str]:
    """Validate a skill name per the Agent Skills spec; return error strings."""
    errors: List[str] = []

    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")

    if not NAME_RE.match(name):
        errors.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )

    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")

    if "--" in name:
        errors.append("name must not contain consecutive hyphens")

    return errors


def validate_description(description: object) -> List[str]:
    """Validate a skill description; return error strings."""
    errors: List[str] = []

    if not isinstance(description, str) or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )

    return errors


# --------------------------------------------------------------------------- #
# Ignore-rule plumbing
# --------------------------------------------------------------------------- #


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _add_ignore_rules(ig: IgnoreMatcher, directory: Path, root_dir: Path) -> None:
    """Load ignore-file patterns from ``directory``, scoped to ``root_dir``."""
    base_dir = _to_posix(directory.relative_to(root_dir)) if directory != root_dir else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path = directory / filename
        try:
            if not ignore_path.is_file():
                continue
            content = ignore_path.read_text(encoding="utf-8")
        except OSError:
            continue
        patterns = [line for line in content.splitlines() if line]
        if patterns:
            ig.add(patterns, base_dir=base_dir)


# --------------------------------------------------------------------------- #
# Loading a single skill file (A2/A4)
# --------------------------------------------------------------------------- #


def load_skill_from_file(
    file_path: str | Path, source: str
) -> tuple[Skill | None, list[Diagnostic]]:
    """Parse and validate one ``SKILL.md`` (or ``.md``) file."""
    file_path = Path(file_path)
    diagnostics: List[Diagnostic] = []
    is_declared_skill = file_path.name == "SKILL.md"

    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(
            Diagnostic(type="warning", message=str(error), path=str(file_path))
        )
        return None, diagnostics

    try:
        parsed = parse_frontmatter(raw_content)
    except Exception as error:  # noqa: BLE001 — mirror pi's catch-all warning
        if is_declared_skill:
            diagnostics.append(
                Diagnostic(type="warning", message=str(error), path=str(file_path))
            )
        return None, diagnostics

    frontmatter: Dict[str, object] = parsed["frontmatter"]
    description = frontmatter.get("description")
    has_description = isinstance(description, str) and description.strip() != ""
    if not is_declared_skill and not has_description:
        return None, diagnostics

    skill_dir = file_path.parent

    # Validate description (warnings only — skip happens below).
    for error in validate_description(description):
        diagnostics.append(Diagnostic(type="warning", message=error, path=str(file_path)))

    # Use frontmatter name, falling back to the parent directory name.
    frontmatter_name = frontmatter.get("name")
    name = frontmatter_name if isinstance(frontmatter_name, str) else skill_dir.name

    for error in validate_name(name):
        diagnostics.append(Diagnostic(type="warning", message=error, path=str(file_path)))

    # Missing/empty description => skip entirely, even if name is bad.
    if not has_description:
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,  # type: ignore[arg-type] # checked above
            path=str(file_path),
            base_dir=str(skill_dir),
            source=source,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


# --------------------------------------------------------------------------- #
# Directory discovery (A3)
# --------------------------------------------------------------------------- #


def _walk_skill_roots(
    directory: Path,
    ig: IgnoreMatcher,
    root_dir: Path,
    realpath_seen: set[str],
    out_dirs: List[Path],
    skip_ignored: bool,
) -> None:
    """Recursively find skill roots (directories containing ``SKILL.md``).

    Ignore rules are accumulated lazily as we descend, mirroring pi: rules from
    the root ignore files are added first, deeper ones later, so deeper rules
    win ties ("last matching rule wins").
    """
    if skip_ignored:
        _add_ignore_rules(ig, directory, root_dir)
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return

    # 1) A directory that contains a *valid, non-ignored* SKILL.md is a skill
    # root — stop recursing. An ignored (or non-file) SKILL.md does not stop
    # recursion: fall through to (2) and keep descending, matching pi.
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        try:
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            is_file = False
        rel = os.path.relpath(entry.path, root_dir).replace(os.sep, "/")
        if is_file and not (skip_ignored and ig.ignores(rel)):
            out_dirs.append(directory)
            return  # skill root: never recurse further

    # 2) Otherwise recurse into non-dot, non-skipped, non-ignored subdirs.
    for entry in entries:
        name = entry.name
        if name.startswith(".") or name in ALWAYS_SKIP_DIRS:
            continue

        try:
            is_dir = entry.is_dir(follow_symlinks=True)
        except OSError:
            continue
        if not is_dir:
            continue

        # Symlink dedup by realpath.
        try:
            real = os.path.realpath(entry.path)
        except OSError:
            continue
        if real in realpath_seen:
            continue
        realpath_seen.add(real)

        rel = os.path.relpath(entry.path, root_dir).replace(os.sep, "/")
        if skip_ignored and ig.ignores(rel, is_dir=True):
            continue

        _walk_skill_roots(
            Path(entry.path), ig, root_dir, realpath_seen, out_dirs, skip_ignored
        )


def discover_skills(
    directory: str | Path,
    source: str,
    *,
    respect_ignore: bool = True,
) -> LoadSkillsResult:
    """Discover all skills under ``directory``.

    Returns skills plus diagnostics (name collisions, validation warnings).
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        return LoadSkillsResult()

    root_dir = directory
    ig = IgnoreMatcher()

    skill_roots: List[Path] = []
    realpath_seen: set[str] = {str(os.path.realpath(root_dir))}
    _walk_skill_roots(
        root_dir, ig, root_dir, realpath_seen, skill_roots, skip_ignored=respect_ignore
    )

    skills: List[Skill] = []
    diagnostics: List[Diagnostic] = []
    seen: Dict[str, Skill] = {}

    for skill_root in skill_roots:
        skill_file = skill_root / "SKILL.md"
        skill, ds = load_skill_from_file(skill_file, source)
        diagnostics.extend(ds)
        if skill is None:
            continue

        existing = seen.get(skill.name)
        if existing is not None:
            diagnostics.append(
                Diagnostic(
                    type="collision",
                    message=f'name "{skill.name}" collision',
                    path=skill.path,
                    collision=Collision(
                        resource_type="skill",
                        name=skill.name,
                        winner_path=existing.path,
                        loser_path=skill.path,
                    ),
                )
            )
            continue

        seen[skill.name] = skill
        skills.append(skill)

    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Top-level loading (defaults + explicit paths)
# --------------------------------------------------------------------------- #


def default_agent_dir() -> Path:
    """Return the agent directory that holds global (user) skills."""
    return Path.home() / CONFIG_DIR_NAME


def _realpath_key(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def load_skills(
    *,
    cwd: str | Path | None = None,
    agent_dir: str | Path | None = None,
    skill_paths: Sequence[str | Path] = (),
    include_defaults: bool = True,
) -> LoadSkillsResult:
    """Load skills from all configured locations (mirrors pi's ``loadSkills``).

    Defaults: user skills at ``<agent_dir>/skills`` and project skills at
    ``<cwd>/.pirl/skills``, plus any explicit ``skill_paths`` (files or dirs).
    """
    resolved_cwd = Path(cwd) if cwd is not None else Path.cwd()
    resolved_agent_dir = Path(agent_dir) if agent_dir is not None else default_agent_dir()

    skill_map: Dict[str, Skill] = {}
    realpath_set: set[str] = set()
    all_diagnostics: List[Diagnostic] = []
    collision_diagnostics: List[Diagnostic] = []

    def add_skills(result: LoadSkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            real = _realpath_key(skill.path)
            # Same physical file reached twice (symlink) — skip silently.
            if real in realpath_set:
                continue

            existing = skill_map.get(skill.name)
            if existing is not None:
                collision_diagnostics.append(
                    Diagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.path,
                        collision=Collision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.path,
                            loser_path=skill.path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                realpath_set.add(real)

    if include_defaults:
        add_skills(discover_skills(resolved_agent_dir / "skills", "user"))
        add_skills(discover_skills(resolved_cwd / CONFIG_DIR_NAME / "skills", "project"))

    user_skills_dir = resolved_agent_dir / "skills"
    project_skills_dir = resolved_cwd / CONFIG_DIR_NAME / "skills"

    def resolve_source(path: Path) -> str:
        if not include_defaults:
            try:
                if path.is_relative_to(user_skills_dir):
                    return "user"
                if path.is_relative_to(project_skills_dir):
                    return "project"
            except (OSError, ValueError):
                pass
        return "path"

    for raw_path in skill_paths:
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            resolved = resolved_cwd / resolved
        resolved = resolved.resolve()

        if not resolved.exists():
            all_diagnostics.append(
                Diagnostic(type="warning", message="skill path does not exist", path=str(resolved))
            )
            continue

        try:
            if resolved.is_dir():
                add_skills(discover_skills(resolved, resolve_source(resolved)))
            elif resolved.is_file() and resolved.suffix == ".md":
                skill, ds = load_skill_from_file(resolved, resolve_source(resolved))
                if skill is not None:
                    add_skills(LoadSkillsResult(skills=[skill], diagnostics=ds))
                else:
                    all_diagnostics.extend(ds)
            else:
                all_diagnostics.append(
                    Diagnostic(
                        type="warning",
                        message="skill path is not a markdown file",
                        path=str(resolved),
                    )
                )
        except OSError as error:
            all_diagnostics.append(
                Diagnostic(type="warning", message=str(error), path=str(resolved))
            )

    return LoadSkillsResult(
        skills=list(skill_map.values()),
        diagnostics=[*all_diagnostics, *collision_diagnostics],
    )
