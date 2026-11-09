import subprocess

from pirl.git_snapshot import is_git, restore, snapshot


def _init_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)


def test_snapshot_non_git_returns_none(tmp_path):
    assert snapshot(tmp_path) is None


def test_snapshot_records_head(tmp_path):
    _init_git(tmp_path)
    snap = snapshot(tmp_path)
    assert snap is not None
    assert snap.head


def test_restore_reverts_changes(tmp_path):
    _init_git(tmp_path)
    snap = snapshot(tmp_path)
    (tmp_path / "a.txt").write_text("changed by agent", encoding="utf-8")
    restore(tmp_path, snap)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "v1"


def test_is_git(tmp_path):
    assert not is_git(tmp_path)
    _init_git(tmp_path)
    assert is_git(tmp_path)
