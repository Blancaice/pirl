from pathlib import Path

from pirl.harness import TaskHarness, TaskSpec
from pirl.execution import ApprovalExecutor, DirectExecutor
from pirl.run_history import RunStore


class FakeUsage:
    prompt_tokens = 1
    completion_tokens = 2
    total_tokens = 3


class FakeResult:
    outcome = "success"
    answer = "done"
    message = ""
    usage = FakeUsage()
    trace = []


class FakeAgent:
    def __init__(self):
        self.tasks = []

    def run(self, task):
        self.tasks.append(task)
        return FakeResult()


class FakeTools:
    executor = ApprovalExecutor(DirectExecutor(), "auto")


def test_task_harness_persists_run_boundary(tmp_path: Path):
    spec = TaskSpec.from_values("fix it", tmp_path)
    agent = FakeAgent()
    result = TaskHarness(tmp_path).run(agent, spec)

    assert agent.tasks == ["fix it"]
    assert result.record_path.is_file()
    assert result.run_id


def test_task_harness_records_execution_topology(tmp_path: Path):
    spec = TaskSpec.from_values("fix it", tmp_path)
    agent = FakeAgent()
    agent.tools = FakeTools()
    result = TaskHarness(tmp_path).run(agent, spec)
    record = RunStore(tmp_path).read_run(result.run_id)
    assert record is not None
    assert record.execution == {
        "backend": "direct",
        "isolation": "none",
        "filesystem_boundary": "host user",
        "network_boundary": "host",
        "approval": "auto",
    }
