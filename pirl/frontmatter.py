"""Frontmatter parsing for ``SKILL.md`` files.

Ported from pi's ``utils/frontmatter.ts``. Behaviour is intentionally kept
identical so the ``SKILL.md`` contract stays compatible with the Agent Skills
spec:

* the file is BOM-stripped and newline-normalized (CRLF / CR -> LF)
* frontmatter is only recognised when the file *starts* with ``---``
* the closing delimiter is a line whose content is exactly ``---``
* a missing closing delimiter means the whole file is body (no frontmatter)
* the parsed YAML must be a mapping; ``body`` is returned trimmed
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import yaml

_BOM = "\ufeff"

# Return type of :func:`parse_frontmatter` — kept as a plain dict to mirror the
# TypeScript shape ``{ frontmatter, body }`` exactly.
ParsedFrontmatter = Dict[str, Any]


def strip_bom(value: str) -> str:
    """Remove a single leading UTF-8 byte-order mark, if present."""
    if value.startswith(_BOM):
        return value[1:]
    return value


def normalize_newlines(value: str) -> str:
    """Normalize CRLF and lone CR to LF."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def extract_frontmatter(content: str) -> Tuple[str | None, str]:
    """Split raw file content into ``(yaml_string, body)``.

    ``yaml_string`` is ``None`` when the content does not carry frontmatter.
    """
    normalized = normalize_newlines(strip_bom(content))

    if not normalized.startswith("---"):
        return None, normalized

    # pi starts searching from index 3 so the opening ``---`` is skipped.
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return None, normalized

    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4:].strip()
    return yaml_string, body


def parse_frontmatter(content: str) -> ParsedFrontmatter:
    """Parse ``content`` and return ``{"frontmatter": {...}, "body": "..."}``.

    Raises ``ValueError`` when the frontmatter block is present but is not a
    YAML mapping, or when the YAML itself is invalid. Callers surface that as a
    warning (never fatal) for declared ``SKILL.md`` files.
    """
    yaml_string, body = extract_frontmatter(content)
    if yaml_string is None:
        return {"frontmatter": {}, "body": body}

    parsed = yaml.safe_load(yaml_string)
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return {"frontmatter": parsed, "body": body}


def strip_frontmatter(content: str) -> str:
    """Return the body of ``content`` with any frontmatter removed."""
    return parse_frontmatter(content)["body"]
