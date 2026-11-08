from pirl.tools import ToolRegistry, builtin_tools


def _registry(tmp_path):
    return ToolRegistry(tmp_path, builtin_tools())


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    result = _registry(tmp_path).execute("list_files", {"path": "."})
    assert result.error == ""
    assert "a.txt" in result.content
    assert "d sub" in result.content


def test_read_file_with_line_range(tmp_path):
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = _registry(tmp_path).execute(
        "read_file", {"path": "f.txt", "start_line": 2, "end_line": 2}
    )
    assert result.error == ""
    assert "two" in result.content
    assert "one" not in result.content


def test_write_file_then_read_back(tmp_path):
    reg = _registry(tmp_path)
    w = reg.execute("write_file", {"path": "out.txt", "content": "hello\n"})
    assert w.error == ""
    r = reg.execute("read_file", {"path": "out.txt"})
    assert "hello" in r.content


def test_write_file_escapes_workspace(tmp_path, monkeypatch):
    outside = tmp_path / ".."
    reg = _registry(tmp_path)
    result = reg.execute("write_file", {"path": str(outside / "evil.txt"), "content": "x"})
    assert result.error != ""
    assert "escapes workspace" in result.error


def test_search(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    result = _registry(tmp_path).execute("search", {"pattern": "def foo"})
    assert result.error == ""
    assert "a.py:1" in result.content


def test_bash(tmp_path):
    result = _registry(tmp_path).execute("bash", {"command": "echo hello"})
    assert result.error == ""
    assert "hello" in result.content


def test_unknown_tool(tmp_path):
    result = _registry(tmp_path).execute("nope", {})
    assert result.error == "unknown tool: nope"
