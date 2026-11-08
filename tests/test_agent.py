import json

from pirl.agent import Agent, RunResult, StepStart, ToolCalled
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


def test_finish_success(tmp_path):
    agent = _agent(
        tmp_path,
        [Completion("", [ToolCall("finish", json.dumps({"success": True, "summary": "done"}), "c1")])],
    )
    result = agent.run("x")
    assert result.outcome == "success"
    assert result.answer == "done"


def test_finish_failure(tmp_path):
    agent = _agent(
        tmp_path,
        [Completion("", [ToolCall("finish", json.dumps({"success": False, "summary": "cannot"}), "c1")])],
    )
    result = agent.run("x")
    assert result.outcome == "failure"
    assert result.answer == "cannot"


def test_finish_not_in_trace(tmp_path):
    agent = _agent(
        tmp_path,
        [Completion("", [ToolCall("finish", json.dumps({"success": True, "summary": "d"}), "c1")])],
    )
    result = agent.run("x")
    assert result.trace == []


def test_finish_tool_is_advertised(tmp_path):
    agent = _agent(tmp_path, [Completion("ok", [])])
    assert "finish" in [s.name for s in agent._tool_specs()]


def _activate_skill(store, name="foo"):
    (store.candidate_dir / name).mkdir(parents=True)
    (store.candidate_dir / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n# b", encoding="utf-8"
    )
    store.activate(name, reviewer="t", reason="test")
    return str(store.active_dir / name / "SKILL.md")


def test_skill_usage_is_tracked(tmp_path):
    from pirl.lifecycle import SkillStore

    store = SkillStore(tmp_path / "skills")
    skill_path = _activate_skill(store)
    client = _ScriptedClient([
        Completion("", [ToolCall("read_file", json.dumps({"path": skill_path}), "c1")]),
        Completion("", [ToolCall("finish", json.dumps({"success": True, "summary": "ok"}), "c2")]),
    ])
    agent = Agent(client, tools=ToolRegistry(tmp_path, builtin_tools()), skill_store=store)
    result = agent.run("use foo")
    assert result.outcome == "success"
    record = store.get("foo")
    assert record is not None
    assert record.retrievals == 1
    assert record.adoptions == 1
    assert record.failures == 0


def test_failed_skill_use_records_failure(tmp_path):
    from pirl.lifecycle import SkillStore

    store = SkillStore(tmp_path / "skills")
    skill_path = _activate_skill(store)
    client = _ScriptedClient([
        Completion("", [ToolCall("read_file", json.dumps({"path": skill_path}), "c1")]),
        Completion("", [ToolCall("finish", json.dumps({"success": False, "summary": "nope"}), "c2")]),
    ])
    agent = Agent(client, tools=ToolRegistry(tmp_path, builtin_tools()), skill_store=store)
    agent.run("use foo")
    record = store.get("foo")
    assert record.adoptions == 1
    assert record.failures == 1


def test_reasoning_is_collected(tmp_path):
    agent = _agent(
        tmp_path,
        [
            Completion("", [ToolCall("list_files", '{"path": "."}')], reasoning="think 1"),
            Completion("done", [], reasoning="think 2"),
        ],
    )
    result = agent.run("x")
    assert result.reasoning == ["think 1", "think 2"]


def test_system_prompt_is_sectioned():
    agent = Agent(_ScriptedClient([]), system_prompt="extra rule")
    names = [s.name for s in agent._system_sections()]
    assert names == ["persona", "rules"]


def test_skills_become_a_section():
    skill = Skill("pdf-tools", "Extract text from PDFs.", "/tmp/x/SKILL.md", "/tmp/x", "path")
    agent = Agent(_ScriptedClient([]), skills=[skill])
    names = [s.name for s in agent._system_sections()]
    assert names == ["persona", "skills"]


def test_on_step_callback_fires(tmp_path):
    events = []
    agent = _agent(
        tmp_path,
        [
            Completion("", [ToolCall("list_files", '{"path": "."}')]),
            Completion("done", []),
        ],
        on_step=events.append,
    )
    agent.run("x")
    assert any(isinstance(e, StepStart) for e in events)
    assert any(isinstance(e, ToolCalled) and e.name == "list_files" for e in events)


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
