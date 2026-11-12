from pirl.prompt import PromptSection, format_skills_for_prompt, render_sections
from pirl.plugins.skills import Skill


def test_render_sections_joins_and_drops_empty():
    sections = [
        PromptSection("a", "one"),
        PromptSection("b", "  "),  # empty -> dropped
        PromptSection("c", "two"),
    ]
    assert render_sections(sections) == "one\n\ntwo"


def make_skill(name="pdf-tools", description="Extract text from PDFs.", **kw):
    defaults = dict(
        name=name,
        description=description,
        path=f"/home/u/.pirl/skills/{name}/SKILL.md",
        base_dir=f"/home/u/.pirl/skills/{name}",
        source="user",
    )
    defaults.update(kw)
    return Skill(**defaults)


def test_empty_list_returns_empty_string():
    assert format_skills_for_prompt([]) == ""


def test_contains_only_name_description_location():
    skill = make_skill()
    output = format_skills_for_prompt([skill])
    assert "<available_skills>" in output
    assert "<name>pdf-tools</name>" in output
    assert "<description>Extract text from PDFs.</description>" in output
    assert "<location>/home/u/.pirl/skills/pdf-tools/SKILL.md</location>" in output
    # No body leakage — progressive disclosure.
    assert "Instructions" not in output


def test_escapes_xml_special_characters():
    skill = make_skill(name="a&b<c>", description='use "quotes" & <tags>')
    output = format_skills_for_prompt([skill])
    assert "a&amp;b&lt;c&gt;" in output
    assert "&quot;" in output
    assert "&lt;tags&gt;" in output


def test_filters_disable_model_invocation():
    visible = make_skill("visible", "shown")
    hidden = make_skill("hidden", "hidden", disable_model_invocation=True)
    output = format_skills_for_prompt([visible, hidden])
    assert "visible" in output
    assert "hidden" not in output


def test_all_hidden_returns_empty():
    hidden = make_skill("hidden", "hidden", disable_model_invocation=True)
    assert format_skills_for_prompt([hidden]) == ""


def test_file_read_tool_variants():
    skill = make_skill()
    assert "read tool" in format_skills_for_prompt([skill], "read")
    assert "Use bash" in format_skills_for_prompt([skill], "bash")
    assert "read tool" not in format_skills_for_prompt([skill], "indirect")
