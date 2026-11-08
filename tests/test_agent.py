from pirl.agent import Agent, RunResult
from pirl.llm import Completion, ToolCall
from pirl.proposal import ToolCall as TraceCall
from pirl.skills import Skill
from pirl.tools import ToolRegistry, builtin_tools


class _ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def _agent(tmp_path, responses, **kwargs):
    client = _ScriptedClient(responses)
    registry = ToolRegistry(tmp_path, builtin_tools())
    return Agent(client, tools=registry, **kwargs)


def test_run_success(tmp_path):
    agent = _agent(
        tmp_path,
        [
            Completion("", [ToolCall("list_files", '{"path": "."}', "call_1")]),
            Completion("done!", []),
        ],
    )
    result = agent.run("list the files")
    assert result.outcome == "success"
    assert result.answer == "done!"
    assert result.trace == [TraceCall("list_files", '{"path": "."}')]


def test_run_max_steps_is_failure(tmp_path):
    agent = _agent(
        tmp_path,
        [Completion("", [ToolCall("list_files", '{"path": "."}')])] * 10,
        max_steps=3,
    )
    result = agent.run("x")
    assert result.outcome == "failure"
    assert result.message == "max steps exceeded"
    assert len(result.trace) == 3


def test_run_unknown_tool_reports_error_and_continues(tmp_path):
    agent = _agent(
        tmp_path,
        [
            Completion("", [ToolCall("nope", "{}", "call_1")]),
            Completion("ok", []),
        ],
    )
    result = agent.run("x")
    assert result.outcome == "success"
    # The error was fed back to the model as a tool message, then it answered.
    assert result.answer == "ok"


def test_system_message_includes_skills():
    skill = Skill("pdf-tools", "Extract text from PDFs.", "/tmp/x/SKILL.md", "/tmp/x", "path")
    agent = Agent(_ScriptedClient([]), skills=[skill])
    system = agent._system_message()
    assert "pdf-tools" in system
    assert "Extract text from PDFs." in system


def test_self_gen_called_at_end(tmp_path):
    calls = []

    class _FakeSelfGen:
        def on_run_end(self, trace, outcome):
            calls.append((list(trace), outcome))
            return None

    agent = _agent(
        tmp_path,
        [Completion("finished", [])],
        self_gen=_FakeSelfGen(),
    )
    result = agent.run("task")
    assert result.outcome == "success"
    assert calls == [([], "success")]
