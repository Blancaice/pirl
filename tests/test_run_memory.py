from pirl.plugins.memory import memory_path, read_memory, write_memory


def test_read_missing_returns_empty(tmp_path):
    assert read_memory(tmp_path) == ""


def test_write_then_read_roundtrip(tmp_path):
    write_memory(tmp_path, "success", "fixed calc.py")
    assert read_memory(tmp_path) == "outcome: success\n\nfixed calc.py"


def test_memory_lives_under_dot_pirl(tmp_path):
    assert memory_path(tmp_path) == tmp_path / ".pirl" / "last-run.md"
