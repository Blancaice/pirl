from pirl.agent import Agent
from pirl.llm import Completion, ToolCall, Usage
from pirl.tools import ToolRegistry, builtin_tools


class Client:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, tools=None):
        return self.responses.pop(0)


def test_default_has_no_step_cap_but_repeat_guard_stops(tmp_path):
    client = Client([Completion("", [ToolCall("list_files", '{"path":"."}')])] * 10)
    agent = Agent(client, tools=ToolRegistry(tmp_path, builtin_tools()))
    result = agent.run("x")
    assert result.outcome == "failure"
    assert result.message == "repeated tool call detected"


def test_total_token_budget_stops_before_next_turn(tmp_path):
    client = Client([
        Completion("", [ToolCall("list_files", '{"path":"."}')], usage=Usage(total_tokens=5)),
    ])
    agent = Agent(
        client, tools=ToolRegistry(tmp_path, builtin_tools()), max_total_tokens=5
    )
    result = agent.run("x")
    assert result.outcome == "failure"
    assert result.message == "max total tokens exceeded (5)"


def test_read_only_exploration_does_not_count_as_stagnation(tmp_path):
    client = Client([
        Completion("", [ToolCall("list_files", '{"path":"."}')]),
        Completion("", [ToolCall("read_file", '{"path":"calc.py"}')]),
        Completion("", [ToolCall("read_file", '{"path":"test_calc.py"}')]),
        Completion("The files look correct.", []),
    ])
    agent = Agent(client, tools=ToolRegistry(tmp_path, builtin_tools()))
    result = agent.run("inspect the project")
    assert result.outcome == "success"
    assert result.answer == "The files look correct."
