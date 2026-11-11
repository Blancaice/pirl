from pathlib import Path

from pirl.harness import TaskHarness, TaskSpec


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


def test_task_harness_persists_run_boundary(tmp_path: Path):
    spec = TaskSpec.from_values("fix it", tmp_path)
    agent = FakeAgent()
    result = TaskHarness(tmp_path).run(agent, spec)

    assert agent.tasks == ["fix it"]
    assert result.record_path.is_file()
    assert result.run_id
