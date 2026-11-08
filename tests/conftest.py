"""Shared pytest helpers for building skill fixtures on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest


def write_skill(
    root: Path,
    name: str,
    description: str = "A test skill.",
    *,
    body: str = "# Test body",
    disable_model_invocation: Optional[bool] = None,
    extra_frontmatter: Optional[str] = None,
    raw: Optional[str] = None,
    filename: str = "SKILL.md",
) -> Path:
    """Create ``<root>/<name>/SKILL.md`` (or any ``filename``) and return its dir."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    if raw is not None:
        content = raw
    else:
        lines = [f"name: {name}", f"description: {description}"]
        if disable_model_invocation is not None:
            lines.append(
                f"disable-model-invocation: {'true' if disable_model_invocation else 'false'}"
            )
        if extra_frontmatter:
            lines.append(extra_frontmatter)
        content = "---\n" + "\n".join(lines) + f"\n---\n{body}\n"

    (skill_dir / filename).write_text(content, encoding="utf-8")
    return skill_dir


@pytest.fixture
def skill_factory(tmp_path: Path):
    """Return a ``write_skill`` bound to a fresh ``tmp_path``."""
    def factory(
        name: str,
        description: str = "A test skill.",
        **kwargs,
    ) -> Path:
        return write_skill(tmp_path, name, description, **kwargs)

    return factory
