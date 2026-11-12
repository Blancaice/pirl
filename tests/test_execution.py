import pytest

from pirl.execution import (
    ApprovalExecutor,
    AuditExecutor,
    DirectExecutor,
    DisabledExecutor,
    ExecRequest,
    ExecResult,
    ExecutionInfo,
    ExecutorRegistry,
    PolicyExecutor,
)


class RecordingExecutor:
    info = ExecutionInfo("recording", "test", "test", "test")

    def __init__(self):
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return ExecResult(0, stdout="ok", execution_scope="recording")


def request(tmp_path, *argv):
    return ExecRequest(tuple(argv), tmp_path)


def test_direct_executor_runs_argv_without_shell(tmp_path):
    result = DirectExecutor().execute(request(tmp_path, "printf", "%s", "a; b"))
    assert result.ok
    assert result.stdout == "a; b"


def test_disabled_executor_denies(tmp_path):
    result = DisabledExecutor().execute(request(tmp_path, "printf", "x"))
    assert result.denied
    assert not result.ok


def test_disabled_executor_does_not_prompt(tmp_path):
    executor = ApprovalExecutor(
        DisabledExecutor(),
        "on-request",
        lambda _req: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )
    assert executor.execute(request(tmp_path, "true")).denied


def test_policy_runs_before_approval_when_composed(tmp_path):
    base = RecordingExecutor()
    approvals = []
    executor = PolicyExecutor(
        ApprovalExecutor(base, "on-request", lambda req: approvals.append(req) or True),
        lambda _req: "blocked",
    )
    result = executor.execute(request(tmp_path, "bad"))
    assert result.denied
    assert approvals == []
    assert base.requests == []


def test_host_preapproved_request_bypasses_interactive_approval(tmp_path):
    base = RecordingExecutor()
    executor = ApprovalExecutor(base, "on-request", lambda _req: False)
    req = ExecRequest(("true",), tmp_path, approved_by_host=True)
    assert executor.execute(req).ok


def test_audit_wraps_denied_and_successful_results(tmp_path):
    events = []
    executor = AuditExecutor(
        ApprovalExecutor(RecordingExecutor(), "never"),
        lambda phase, req, result: events.append((phase, req, result)),
    )
    result = executor.execute(request(tmp_path, "true"))
    assert result.denied
    assert [event[0] for event in events] == ["requested", "completed"]
    assert events[-1][2] is result


def test_executor_registry_creates_selected_backend(tmp_path):
    registry = ExecutorRegistry()
    registry.register("recording", lambda _root: RecordingExecutor())
    assert registry.create("recording", tmp_path).info.name == "recording"
    with pytest.raises(ValueError, match="unknown executor"):
        registry.create("missing", tmp_path)


def test_executor_registry_loads_only_selected_entry_point(tmp_path, monkeypatch):
    class EntryPoint:
        name = "third-party"

        def load(self):
            return lambda _root: RecordingExecutor()

    class EntryPoints:
        def select(self, *, group, name):
            assert group == "pirl.executors"
            return [EntryPoint()] if name == "third-party" else []

    monkeypatch.setattr("pirl.execution.metadata.entry_points", lambda: EntryPoints())
    registry = ExecutorRegistry()
    assert registry.load_entry_point("third-party")
    assert registry.create("third-party", tmp_path).info.name == "recording"


def test_approval_info_describes_composed_topology():
    executor = PolicyExecutor(ApprovalExecutor(DirectExecutor(), "auto"), lambda _req: "")
    assert executor.info.name == "direct"
    assert executor.info.isolation == "none"
    assert executor.info.approval == "auto"
