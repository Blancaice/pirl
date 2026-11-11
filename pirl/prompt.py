"""Progressive disclosure: build the ``<available_skills>`` block (A5).

Ported from pi's ``formatSkillsForPrompt``. Only ``name``, ``description`` and
``location`` are exposed in the system prompt — the full ``SKILL.md`` body is
loaded on demand (via the read tool or an explicit ``/skill:name`` command).

Skills with ``disable_model_invocation=True`` are excluded from the prompt; they
can only be invoked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .skills import Skill

@dataclass(frozen=True)
class PromptSection:
    """One named, stable chunk of the system prompt.

    The system prompt is a fixed ordered list of sections (stable prefix).
    Adding context means appending a new section, never rewriting the persona,
    so the prefix stays byte-stable for any future prompt caching.
    """

    name: str
    text: str


def render_sections(sections: Iterable[PromptSection]) -> str:
    """Concatenate non-empty sections in order, separated by blank lines."""
    return "\n\n".join(s.text.strip() for s in sections if s.text.strip())


FileReadTool = Literal["read", "read_file", "bash", "indirect"]

_INTRO = "\n\nThe following skills provide specialized instructions for specific tasks."

_READ_TOOL_LINES = {
    "read": "Use the read tool (`read_file`) to load a skill's file when the task matches its description.",
    "read_file": "Use the read tool (`read_file`) to load a skill's file when the task matches its description.",
    "bash": "Use bash to load a skill's file when the task matches its description.",
    "indirect": "Load a skill's file when the task matches its description.",
}


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(
    skills: Iterable[Skill],
    file_read_tool: FileReadTool = "read",
) -> str:
    """Render ``skills`` as the progressive-disclosure XML block.

    Returns an empty string when there is nothing to show.
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        _INTRO,
        _READ_TOOL_LINES[file_read_tool],
        "Resolve relative paths against the skill's directory (parent of SKILL.md). "
        "Use absolute paths in tool calls.",
        "",
        "<available_skills>",
    ]

    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)
