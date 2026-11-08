import pytest

from pirl.ignore import IgnoreMatcher


def m(*patterns, base_dir=""):
    ig = IgnoreMatcher()
    ig.add(patterns, base_dir=base_dir)
    return ig


def test_blank_lines_and_comments_are_skipped():
    ig = m("", "# a comment", "*.log")
    assert ig.ignores("x.log")
    assert not ig.ignores("x.txt")


def test_star_matches_within_one_segment_only():
    ig = m("*.log")
    assert ig.ignores("a.log")
    assert ig.ignores("deep/a.log")
    assert not ig.ignores("a.log/b.txt")  # not a dir; '*' doesn't cross '/'


def test_basename_pattern_matches_any_depth():
    ig = m("node_modules")
    assert ig.ignores("node_modules", is_dir=True)
    assert ig.ignores("a/b/node_modules", is_dir=True)
    assert not ig.ignores("a/node_modules_x", is_dir=True)


def test_negation_reincludes():
    ig = m("*.log", "!keep.log")
    assert ig.ignores("a.log")
    assert not ig.ignores("keep.log")


def test_leading_slash_anchors_to_ignore_dir():
    ig = m("/build")
    assert ig.ignores("build", is_dir=True)
    assert not ig.ignores("sub/build", is_dir=True)


def test_dir_only_pattern():
    ig = m("build/")
    assert ig.ignores("build", is_dir=True)
    assert ig.ignores("sub/build", is_dir=True)
    assert not ig.ignores("build", is_dir=False)  # a *file* named build survives


def test_double_star_matches_across_directories():
    assert m("**/foo").ignores("foo")
    assert m("**/foo").ignores("a/b/foo")
    assert m("a/**/b").ignores("a/b")
    assert m("a/**/b").ignores("a/x/y/b")
    assert m("a/**").ignores("a/x")
    assert m("a/**").ignores("a/x/y")


def test_question_mark_matches_single_char():
    ig = m("file?.txt")
    assert ig.ignores("file1.txt")
    assert not ig.ignores("file12.txt")


def test_character_class():
    ig = m("file[0-9].txt")
    assert ig.ignores("file5.txt")
    assert not ig.ignores("filex.txt")

    neg = m("file[!0-9].txt")
    assert neg.ignores("filex.txt")
    assert not neg.ignores("file5.txt")


def test_last_matching_rule_wins():
    ig = m("*.log", "!a.log", "a.log")
    assert ig.ignores("a.log")


def test_nested_ignore_file_is_scoped_to_its_directory():
    ig = IgnoreMatcher()
    ig.add(["*.tmp"], base_dir="")  # root ignore
    ig.add(["*.tmp"], base_dir="sub")  # sub ignore
    # Root rule applies everywhere; sub rule only under sub/.
    assert ig.ignores("x.tmp")
    assert ig.ignores("sub/y.tmp")
    # A root pattern can be re-included inside sub/.
    ig2 = IgnoreMatcher()
    ig2.add(["*.tmp"], base_dir="")
    ig2.add(["!keep.tmp"], base_dir="sub")
    assert ig2.ignores("sub/keep.tmp") is False  # re-included inside sub
    assert ig2.ignores("keep.tmp") is True  # re-include only applies under sub/
    assert ig2.ignores("other/keep.tmp") is True  # outside sub still ignored


def test_scoped_pattern_does_not_match_sibling_dirs():
    ig = IgnoreMatcher()
    ig.add(["secret.txt"], base_dir="sub")
    assert ig.ignores("sub/secret.txt")
    assert not ig.ignores("secret.txt")
    assert not ig.ignores("other/secret.txt")
