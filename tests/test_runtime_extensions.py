import threading

from pirl.agent import Agent
from pirl.checkpoint import CheckpointStore
from pirl.llm import Completion
from pirl.tools import ToolRegistry, builtin_tools
from pirl.verification import CommandVerifier


class Client:
    def complete(self, messages, tools=None):
        return Completion("done", [])


def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save("r1", {"step": 2, "messages": []})
    assert store.load("r1")["step"] == 2


def test_command_verifier_is_injectable(tmp_path):
    assert CommandVerifier("true", tmp_path)()[0]
    assert not CommandVerifier("false", tmp_path)()[0]


def test_agent_honors_cancel_event(tmp_path):
    event = threading.Event()
    event.set()
    result = Agent(Client(), tools=ToolRegistry(tmp_path, builtin_tools()), cancel_event=event).run("x")
    assert result.outcome == "failure"
    assert result.message == "cancelled"


def test_checkpoint_resume_uses_saved_messages(tmp_path):
    state = {
        "messages": [{"role": "system", "content": "x"}],
        "trace": [],
    }
    result = Agent(Client(), tools=ToolRegistry(tmp_path, builtin_tools())).run("x", resume_state=state)
    assert result.answer == "done"


def test_tool_registry_only_marks_read_tools_parallel_safe(tmp_path):
    registry = ToolRegistry(tmp_path, builtin_tools())
    assert registry.can_parallel(["read_file", "search"])
    assert not registry.can_parallel(["read_file", "write_file"])
