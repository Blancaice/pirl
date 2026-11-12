from pirl.commands import expand_skill_command
from pirl.plugins.skills import Skill

from tests.conftest import write_skill


def make_skill(path, name="pdf-tools", **kw):
    defaults = dict(
        name=name,
        description="desc",
        path=str(path),
        base_dir=str(path.parent),
        source="path",
    )
    defaults.update(kw)
    return Skill(**defaults)


def test_not_a_skill_command_passes_through():
    assert expand_skill_command("hello world", []) == "hello world"
    assert expand_skill_command("/other:foo", []) == "/other:foo"


def test_unknown_skill_passes_through(tmp_path):
    skill = make_skill(write_skill(tmp_path, "known") / "SKILL.md")
    assert expand_skill_command("/skill:missing", [skill]) == "/skill:missing"


def test_expands_skill_body(tmp_path):
    skill_dir = write_skill(tmp_path, "pdf-tools", "desc", body="# Extract PDFs\n\nDo the thing.")
    skill = make_skill(skill_dir / "SKILL.md")
    output = expand_skill_command("/skill:pdf-tools", [skill])
    assert output.startswith('<skill name="pdf-tools"')
    assert f'location="{skill.path}"' in output
    assert f"References are relative to {skill.base_dir}." in output
    assert "# Extract PDFs" in output
    # Frontmatter must be stripped — no name/description leakage into the block.
    assert "name: pdf-tools" not in output
    assert output.rstrip().endswith("</skill>")


def test_expands_with_args(tmp_path):
    skill_dir = write_skill(tmp_path, "pdf-tools", "desc", body="# Body")
    skill = make_skill(skill_dir / "SKILL.md")
    output = expand_skill_command("/skill:pdf-tools convert foo.pdf", [skill])
    assert output.endswith("\n\nconvert foo.pdf")
    assert "# Body" in output


def test_accepts_mapping_of_skills(tmp_path):
    skill_dir = write_skill(tmp_path, "pdf-tools", "desc", body="# Body")
    skill = make_skill(skill_dir / "SKILL.md")
    output = expand_skill_command("/skill:pdf-tools", {"pdf-tools": skill})
    assert "# Body" in output


def test_unreadable_skill_passes_through(tmp_path):
    skill = make_skill(tmp_path / "missing" / "SKILL.md")
    assert expand_skill_command("/skill:missing-file", [skill]) == "/skill:missing-file"
