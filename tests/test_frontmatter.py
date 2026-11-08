import textwrap

import pytest

from pirl.frontmatter import (
    extract_frontmatter,
    normalize_newlines,
    parse_frontmatter,
    strip_bom,
    strip_frontmatter,
)


def test_no_frontmatter_returns_empty_and_body():
    # Faithful to pi: when there is no frontmatter the body is returned
    # untrimmed (the trim only happens after a frontmatter block is stripped).
    parsed = parse_frontmatter("# just a body\n")
    assert parsed["frontmatter"] == {}
    assert parsed["body"] == "# just a body\n"


def test_parse_basic_frontmatter():
    parsed = parse_frontmatter("---\nname: pdf-tools\ndescription: hello\n---\n# Body\n")
    assert parsed["frontmatter"] == {"name": "pdf-tools", "description": "hello"}
    assert parsed["body"] == "# Body"


def test_body_is_trimmed():
    parsed = parse_frontmatter("---\nname: x\ndescription: y\n---\n\n\nBody text\n\n")
    assert parsed["body"] == "Body text"


def test_crlf_is_normalized():
    parsed = parse_frontmatter(
        "---\r\nname: x\r\ndescription: y\r\n---\r\n# Body\r\n"
    )
    assert parsed["frontmatter"]["name"] == "x"
    assert parsed["body"] == "# Body"


def test_bom_is_stripped():
    parsed = parse_frontmatter("\ufeff---\nname: x\ndescription: y\n---\n# Body\n")
    assert parsed["frontmatter"]["name"] == "x"


def test_missing_closing_delimiter_is_treated_as_body():
    parsed = parse_frontmatter("---\nname: x\n# never closed")
    assert parsed["frontmatter"] == {}
    assert parsed["body"].startswith("---")


def test_closing_delimiter_is_first_newline_dash_dash_dash():
    # Faithful to pi: the first "\n---" ends the frontmatter, even when more
    # characters follow on the same line.
    parsed = parse_frontmatter("---\nname: x\n---inline\n---\nbody")
    assert parsed["frontmatter"] == {"name": "x"}
    assert parsed["body"] == "inline\n---\nbody"


def test_non_mapping_frontmatter_raises():
    with pytest.raises(ValueError):
        parse_frontmatter("---\n- a\n- b\n---\nbody")


def test_invalid_yaml_raises():
    with pytest.raises(Exception):
        parse_frontmatter("---\nname: [unclosed\n---\nbody")


def test_strip_frontmatter_returns_body_only():
    assert strip_frontmatter("---\nname: x\ndescription: y\n---\n# Body") == "# Body"


def test_normalize_newlines_handles_lone_cr():
    assert normalize_newlines("a\rb\rc") == "a\nb\nc"


def test_strip_bom_only_strips_leading_bom():
    assert strip_bom("\ufeffabc") == "abc"
    assert strip_bom("a\ufeffbc") == "a\ufeffbc"


def test_extract_frontmatter_returns_none_without_marker():
    yaml_str, body = extract_frontmatter("no frontmatter here")
    assert yaml_str is None
    assert body == "no frontmatter here"
