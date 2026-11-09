import pytest

from pirl import ui


def test_confirm_noninteractive_returns_default(monkeypatch):
    monkeypatch.setattr("pirl.ui.is_interactive", lambda: False)
    assert ui.confirm("ok?", default=False) is False
    assert ui.confirm("ok?", default=True) is True


def test_select_noninteractive_returns_default(monkeypatch):
    monkeypatch.setattr("pirl.ui.is_interactive", lambda: False)
    assert ui.select("p", {"a": "activate", "s": "skip"}, default="s") == "s"


def test_select_noninteractive_returns_first_without_default(monkeypatch):
    monkeypatch.setattr("pirl.ui.is_interactive", lambda: False)
    assert ui.select("p", {"a": "activate", "r": "reject"}) == "a"


def test_select_requires_choices():
    with pytest.raises(ValueError):
        ui.select("p", {})
