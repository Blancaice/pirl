"""A dependency-free gitignore-style matcher.

pi relies on the npm ``ignore`` package. To keep this Python port minimal
(``yaml`` is the only runtime dependency), we implement the common subset of
gitignore semantics ourselves:

* blank lines and ``#`` comments are skipped (``\\#`` is a literal ``#``)
* ``!`` negates a pattern (``\\!`` is a literal ``!``)
* a trailing ``/`` restricts the pattern to directories
* a leading ``/`` anchors the pattern to the ignore file's directory
* ``*`` matches anything except ``/``; ``?`` matches a single non-``/`` char
* ``**`` matches across directories (``**/foo``, ``foo/**``, ``a/**/b``)
* ``[abc]`` / ``[!abc]`` character classes
* the last matching rule wins (so later ``!`` rules can re-include)

All paths are expected as POSIX-style strings relative to the scan root.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class _Rule:
    """One compiled ignore rule."""

    regex: "re.Pattern[str]"
    negated: bool
    dir_only: bool
    base_dir: str  # POSIX path of the ignore file's directory, "" for root

    def matches(self, path: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False

        # Constrain the rule to the subtree that owns the ignore file.
        if self.base_dir:
            if path == self.base_dir:
                rel = ""
            elif path.startswith(self.base_dir + "/"):
                rel = path[len(self.base_dir) + 1 :]
            else:
                return False
        else:
            rel = path

        return bool(self.regex.fullmatch(rel))


def _glob_to_regex(pattern: str) -> str:
    """Translate a single gitignore glob (relative, no dir markers) to regex."""
    i = 0
    n = len(pattern)
    out: List[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            j = i
            while j < n and pattern[j] == "*":
                j += 1
            stars = j - i
            if stars >= 2:
                if j < n and pattern[j] == "/":
                    # "**/" — zero or more leading directories
                    out.append(r"(?:.*/)?")
                    i = j + 1
                elif j == n:
                    # "**" at end — match everything that follows
                    out.append(r".*")
                    i = j
                else:
                    # "**" not adjacent to a slash behaves like "*"
                    out.append(r"[^/]*")
                    i += 1
            else:
                out.append(r"[^/]*")
                i += 1
        elif c == "?":
            out.append(r"[^/]")
            i += 1
        elif c == "[":
            i = _append_char_class(pattern, i, out)
        elif c == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _append_char_class(pattern: str, start: int, out: List[str]) -> int:
    """Translate ``[...]`` starting at ``start``; return the next index."""
    i = start + 1
    negated = False
    if i < len(pattern) and pattern[i] in "!^":
        negated = True
        i += 1
    # A ']' immediately after '[' (or '[!') is a literal ']'.
    if i < len(pattern) and pattern[i] == "]":
        i += 1

    # Inside a character class regex treats most chars literally; only '\',
    # ']' and (as a range separator) '-' are special. We keep '-' as-is so
    # ranges like [0-9] survive the translation.
    body: List[str] = []
    while i < len(pattern) and pattern[i] != "]":
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            body.append("\\" + nxt if nxt in "\\]" else nxt)
            i += 2
            continue
        if c == "\\":
            body.append("\\\\")
        elif c == "]":
            body.append("\\]")
        else:
            body.append(c)
        i += 1

    if i >= len(pattern):
        # Unterminated class — treat the '[' literally and move past it.
        out.append(re.escape("["))
        return start + 1

    out.append("[" + ("^" if negated else "") + "".join(body) + "]")
    return i + 1


def _parse_pattern(line: str, base_dir: str) -> Optional[_Rule]:
    """Parse one raw ignore-file line into a compiled :class:`_Rule`."""
    # Trailing spaces are ignored (a lone backslash escapes them; handled by
    # the common case of a trailing backslash being rare — we keep it simple).
    line = line.rstrip()
    if not line:
        return None
    if line.startswith("#"):
        return None
    if line.startswith("\\#"):
        line = line[1:]
    if line.startswith("\\!"):
        line = line[1:]
        negated = False
    elif line.startswith("!"):
        negated = True
        line = line[1:]
    else:
        negated = False

    dir_only = False
    if line.endswith("/"):
        dir_only = True
        line = line.rstrip("/")
    if not line:
        return None

    anchored = line.startswith("/")
    if anchored:
        line = line[1:]

    # Patterns without a slash match the basename at any depth below the
    # ignore file's directory (gitignore rule).
    has_slash = "/" in line
    if has_slash or anchored:
        regex = _glob_to_regex(line)
    else:
        regex = r"(?:.*/)?" + _glob_to_regex(line)

    return _Rule(
        regex=re.compile(regex),
        negated=negated,
        dir_only=dir_only,
        base_dir=base_dir,
    )


class IgnoreMatcher:
    """Accumulates ignore rules and answers ``ignores(path, is_dir)``."""

    def __init__(self) -> None:
        self._rules: List[_Rule] = []

    def add(self, patterns: Iterable[str], base_dir: str = "") -> None:
        """Add raw pattern lines from an ignore file located at ``base_dir``."""
        for line in patterns:
            rule = _parse_pattern(line, base_dir)
            if rule is not None:
                self._rules.append(rule)

    def ignores(self, path: str, is_dir: bool = False) -> bool:
        """Return ``True`` when ``path`` should be ignored (last rule wins)."""
        ignored = False
        for rule in self._rules:
            if rule.matches(path, is_dir):
                ignored = not rule.negated
        return ignored

    def __bool__(self) -> bool:
        return bool(self._rules)
