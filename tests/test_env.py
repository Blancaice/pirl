import os

from pirl.env import load_dotenv


def test_load_dotenv_applies_pairs(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PIRL_API_KEY=sk-abc\nPIRL_MODEL=deepseek-chat\n", encoding="utf-8")
    monkeypatch.delenv("PIRL_API_KEY", raising=False)
    monkeypatch.delenv("PIRL_MODEL", raising=False)

    applied = load_dotenv(env_file)
    assert applied == {"PIRL_API_KEY": "sk-abc", "PIRL_MODEL": "deepseek-chat"}
    assert os.environ["PIRL_API_KEY"] == "sk-abc"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PIRL_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("PIRL_API_KEY", "from-env")

    applied = load_dotenv(env_file)
    assert "PIRL_API_KEY" not in applied
    assert os.environ["PIRL_API_KEY"] == "from-env"


def test_load_dotenv_values_are_literal(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=$HOME/not-expanded\n", encoding="utf-8")
    monkeypatch.delenv("A", raising=False)

    load_dotenv(env_file)
    assert os.environ["A"] == "$HOME/not-expanded"


def test_load_dotenv_missing_file_returns_empty(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}
