"""Explicit skill invocation: ``/skill:<name> [args]`` expansion (A5).

Ported from pi's ``_expandSkillCommand``. A ``/skill:name args`` message is
rewritten into a ``<skill>`` block that inlines the full ``SKILL.md`` body so the
model gets the instructions directly, followed by any user-supplied args.

If the skill is unknown or the file cannot be read, the original text is
returned unchanged (the caller is responsible for surfacing any read error).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .frontmatter import strip_frontmatter
from .skills import Skill


def _lookup_skill(skills: Sequence[Skill], name: str) -> Optional[Skill]:
    for skill in skills:
        if skill.name == name:
            return skill
    return None


def expand_skill_command(text: str, skills: Sequence[Skill] | Mapping[str, Skill]) -> str:
    """Expand ``/skill:<name> [args]`` into its full content block.

    Returns ``text`` unchanged when it is not a skill command or the named skill
    is not found / unreadable.
    """
    if not text.startswith("/skill:"):
        return text

    if isinstance(skills, Mapping):
        lookup: Mapping[str, Skill] = skills
    else:
        lookup = {s.name: s for s in skills}

    space_index = text.find(" ")
    skill_name = text[7:] if space_index == -1 else text[7:space_index]
    args = "" if space_index == -1 else text[space_index + 1 :].strip()

    skill = lookup.get(skill_name)
    if skill is None:
        return text  # Unknown skill — pass through unchanged.

    try:
        with open(skill.path, encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return text  # Unreadable — pass through unchanged.

    body = strip_frontmatter(content).strip()
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.base_dir}.\n\n"
        f"{body}\n"
        f"</skill>"
    )
    return skill_block if not args else f"{skill_block}\n\n{args}"
