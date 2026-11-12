import json

from pirl.tools import ExecRequest, ExecutionPolicy, ToolRegistry, builtin_tools


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


def test_bash_policy_rejects_escape_and_destructive_commands(tmp_path):
    reg = _registry(tmp_path)
    assert reg.execute("bash", {"command": "cat /etc/passwd"}).error
    assert reg.execute("bash", {"command": "rm -rf ."}).error


def test_bash_can_be_disabled_by_host_policy(tmp_path):
    reg = ToolRegistry(tmp_path, builtin_tools(), policy=ExecutionPolicy(allow_bash=False))
    result = reg.execute("bash", {"command": "echo hello"})
    assert "disabled by policy" in result.error


def test_exec_uses_structured_argv_without_shell(tmp_path):
    reg = ToolRegistry(tmp_path, builtin_tools(), approval_mode="auto")
    result = reg.execute("exec", {"argv": ["printf", "%s", "hello; not shell"]})
    assert result.error == ""
    assert result.content == "hello; not shell"


def test_exec_requests_host_approval(tmp_path):
    requests: list[ExecRequest] = []

    def approve(request: ExecRequest) -> bool:
        requests.append(request)
        return True

    reg = ToolRegistry(tmp_path, builtin_tools(), approver=approve)
    result = reg.execute(
        "exec",
        {"argv": ["printf", "ok"], "reason": "verify output", "timeout_seconds": 5},
    )
    assert result.error == ""
    assert requests == [ExecRequest(("printf", "ok"), tmp_path.resolve(), "verify output", 5)]


def test_exec_rejected_without_approval(tmp_path):
    reg = ToolRegistry(tmp_path, builtin_tools())
    result = reg.execute("exec", {"argv": ["printf", "no"]})
    assert "not approved" in result.error


def test_exec_auto_skips_approver(tmp_path):
    reg = ToolRegistry(
        tmp_path,
        builtin_tools(),
        approval_mode="auto",
        approver=lambda _request: (_ for _ in ()).throw(AssertionError("should not ask")),
    )
    result = reg.execute("exec", {"argv": ["printf", "ok"]})
    assert result.error == ""


def test_exec_never_mode_denies(tmp_path):
    reg = ToolRegistry(tmp_path, builtin_tools(), approval_mode="never")
    result = reg.execute("exec", {"argv": ["printf", "no"]})
    assert "denied by approval policy" in result.error


def test_exec_validates_argv_and_timeout(tmp_path):
    reg = ToolRegistry(tmp_path, builtin_tools(), approval_mode="auto")
    assert reg.execute("exec", {"argv": "printf ok"}).error
    assert reg.execute("exec", {"argv": ["printf", "ok"], "timeout_seconds": 0}).error


def test_apply_patch_is_workspace_scoped(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    patch = """diff --git a/a.txt b/a.txt
index 3367afd..3b18e13 100644
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
"""
    result = _registry(tmp_path).execute("apply_patch", {"patch": patch})
    assert result.error == ""
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new\n"


def test_tool_result_has_machine_readable_envelope(tmp_path):
    result = _registry(tmp_path).execute("nope", {})
    payload = json.loads(result.to_message())
    assert payload == {"ok": False, "content": "", "error": "unknown tool: nope"}


def test_list_runs_tool(tmp_path):
    from pirl.run_history import RunRecord, RunStore

    RunStore(tmp_path).write_run(
        RunRecord(run_id="run_0001", task="t", outcome="success", answer="a", started_at="x")
    )
    result = _registry(tmp_path).execute("list_runs", {})
    assert result.error == ""
    assert "run_0001" in result.content


def test_unknown_tool(tmp_path):
    result = _registry(tmp_path).execute("nope", {})
    assert result.error == "unknown tool: nope"
