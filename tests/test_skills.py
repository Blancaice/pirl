import textwrap

import pytest

from pirl.skills import (
    discover_skills,
    load_skill_from_file,
    load_skills,
    validate_description,
    validate_name,
)
from tests.conftest import write_skill


# --------------------------------------------------------------------------- #
# Validation (A4)
# --------------------------------------------------------------------------- #


def test_validate_name_accepts_valid_names():
    assert validate_name("pdf-tools") == []
    assert validate_name("a1-b2") == []


def test_validate_name_rejects_bad_names():
    assert validate_name("Pdf-Tools")  # uppercase
    assert validate_name("pdf_tools")  # underscore
    assert validate_name("-lead")  # leading hyphen
    assert validate_name("trail-")  # trailing hyphen
    assert validate_name("two--hyphens")
    assert validate_name("x" * 65)  # too long
    assert validate_name("")  # empty


def test_validate_description():
    assert validate_description("hello") == []
    assert validate_description("")  # required
    assert validate_description(None)  # required
    assert validate_description("x" * 1025)  # too long


# --------------------------------------------------------------------------- #
# Single-file loading (A2/A4)
# --------------------------------------------------------------------------- #


def test_load_skill_from_file_basic(tmp_path):
    write_skill(tmp_path, "pdf-tools", "Extract text from PDFs.", body="# Instructions")
    skill, diagnostics = load_skill_from_file(tmp_path / "pdf-tools" / "SKILL.md", "path")
    assert skill is not None
    assert skill.name == "pdf-tools"
    assert skill.description == "Extract text from PDFs."
    assert skill.base_dir == str(tmp_path / "pdf-tools")
    assert skill.source == "path"
    assert skill.disable_model_invocation is False
    assert diagnostics == []


def test_name_falls_back_to_parent_directory(tmp_path):
    (tmp_path / "my-skill").mkdir()
    (tmp_path / "my-skill" / "SKILL.md").write_text(
        "---\ndescription: desc\n---\n# body", encoding="utf-8"
    )
    skill, _ = load_skill_from_file(tmp_path / "my-skill" / "SKILL.md", "path")
    assert skill.name == "my-skill"


def test_disable_model_invocation_true(tmp_path):
    write_skill(tmp_path, "s", "desc", disable_model_invocation=True)
    skill, _ = load_skill_from_file(tmp_path / "s" / "SKILL.md", "path")
    assert skill.disable_model_invocation is True


def test_missing_description_skips_skill(tmp_path):
    write_skill(tmp_path, "s", raw="---\nname: s\n---\n# body")
    skill, diagnostics = load_skill_from_file(tmp_path / "s" / "SKILL.md", "path")
    assert skill is None
    assert any("description is required" in d.message for d in diagnostics)


def test_bad_name_is_warning_not_fatal(tmp_path):
    write_skill(tmp_path, "Bad_Name", "desc", raw="---\nname: Bad_Name\ndescription: desc\n---\n# body")
    skill, diagnostics = load_skill_from_file(tmp_path / "Bad_Name" / "SKILL.md", "path")
    assert skill is not None  # still loaded
    assert skill.name == "Bad_Name"
    assert any("invalid characters" in d.message for d in diagnostics)


def test_frontmatter_parse_failure_is_warning(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text(
        "---\nname: [unclosed\n---\n# body", encoding="utf-8"
    )
    skill, diagnostics = load_skill_from_file(tmp_path / "broken" / "SKILL.md", "path")
    assert skill is None
    assert any(d.type == "warning" for d in diagnostics)


def test_unreadable_file_is_warning(tmp_path):
    skill, diagnostics = load_skill_from_file(tmp_path / "does-not-exist" / "SKILL.md", "path")
    assert skill is None
    assert any(d.type == "warning" for d in diagnostics)


# --------------------------------------------------------------------------- #
# Discovery (A3)
# --------------------------------------------------------------------------- #


def test_discovers_three_skills(tmp_path):
    write_skill(tmp_path, "pdf-tools", "Extract text from PDFs.")
    write_skill(tmp_path, "csv-tools", "Work with CSV files.")
    write_skill(tmp_path, "git-tools", "Help with git.")
    (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")

    result = discover_skills(tmp_path, "path")
    names = sorted(s.name for s in result.skills)
    assert names == ["csv-tools", "git-tools", "pdf-tools"]
    assert result.diagnostics == []


def test_skill_root_does_not_recurse(tmp_path):
    skill_dir = write_skill(tmp_path, "outer", "desc")
    # A nested SKILL.md inside a skill root must NOT be discovered separately.
    (skill_dir / "references" / "inner").mkdir(parents=True)
    (skill_dir / "references" / "inner" / "SKILL.md").write_text(
        "---\nname: inner\ndescription: nested\n---\n# body", encoding="utf-8"
    )
    result = discover_skills(tmp_path, "path")
    assert [s.name for s in result.skills] == ["outer"]


def test_name_collision_first_wins(tmp_path):
    write_skill(tmp_path, "dup", "first", raw="---\nname: dup\ndescription: first\n---\n# a")
    write_skill(tmp_path, "other", "second", raw="---\nname: dup\ndescription: second\n---\n# b")
    result = discover_skills(tmp_path, "path")
    assert [s.name for s in result.skills] == ["dup"]
    assert result.skills[0].description == "first"
    collisions = [d for d in result.diagnostics if d.type == "collision"]
    assert len(collisions) == 1
    assert collisions[0].collision is not None
    assert collisions[0].collision.winner_path.endswith("dup/SKILL.md")


def test_dot_directories_are_skipped(tmp_path):
    write_skill(tmp_path / ".hidden", "hidden", "desc")
    result = discover_skills(tmp_path, "path")
    assert result.skills == []


def test_gitignore_is_respected(tmp_path):
    write_skill(tmp_path, "keep", "desc")
    write_skill(tmp_path, "ignored", "desc")
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    result = discover_skills(tmp_path, "path")
    assert [s.name for s in result.skills] == ["keep"]


def test_ignored_skill_md_does_not_stop_recursion(tmp_path):
    # A directory whose own SKILL.md is ignored (anchored /SKILL.md) must still
    # be descended so nested skills are discovered, matching pi.
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "SKILL.md").write_text(
        "---\nname: outer-skill\ndescription: x\n---\n# body", encoding="utf-8"
    )
    (outer / ".gitignore").write_text("/SKILL.md\n", encoding="utf-8")
    (inner / "SKILL.md").write_text(
        "---\nname: inner-skill\ndescription: inner\n---\n# body", encoding="utf-8"
    )
    result = discover_skills(tmp_path, "path")
    assert [s.name for s in result.skills] == ["inner-skill"]


def test_no_ignore_option_disables_ignore_files(tmp_path):
    write_skill(tmp_path, "keep", "desc")
    write_skill(tmp_path, "ignored", "desc")
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    result = discover_skills(tmp_path, "path", respect_ignore=False)
    assert sorted(s.name for s in result.skills) == ["ignored", "keep"]


def test_symlink_dedup(tmp_path):
    write_skill(tmp_path, "real", "desc")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    result = discover_skills(tmp_path, "path")
    # The same physical skill must not be loaded twice.
    assert [s.name for s in result.skills] == ["real"]


# --------------------------------------------------------------------------- #
# Top-level load_skills
# --------------------------------------------------------------------------- #


def test_load_skills_explicit_paths(tmp_path):
    write_skill(tmp_path, "pdf-tools", "Extract text from PDFs.")
    result = load_skills(cwd=tmp_path, agent_dir=tmp_path / "agent", skill_paths=[tmp_path])
    names = sorted(s.name for s in result.skills)
    assert "pdf-tools" in names


def test_load_skills_defaults_are_optional(tmp_path):
    # No skill dirs exist yet -> no errors, no skills.
    result = load_skills(cwd=tmp_path, agent_dir=tmp_path / "agent", include_defaults=True)
    assert result.skills == []
    assert result.diagnostics == []
