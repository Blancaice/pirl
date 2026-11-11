from pirl.run_history import RunRecord, RunStore


def _record(run_id="run_0001", task="fix bug", outcome="success", answer="done", trace=None, usage=None):
    return RunRecord(
        run_id=run_id,
        task=task,
        outcome=outcome,
        answer=answer,
        started_at="2026-01-01T00:00:00+00:00",
        trace=trace or [],
        usage=usage or {},
    )


def test_write_and_read_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    store.write_run(_record(trace=[{"tool": "bash", "args": "pytest"}]))
    record = store.read_run("run_0001")
    assert record is not None
    assert record.task == "fix bug"
    assert record.trace == [{"tool": "bash", "args": "pytest"}]


def test_list_runs_newest_first(tmp_path):
    store = RunStore(tmp_path)
    store.write_run(_record(run_id="run_0001", task="one"))
    store.write_run(_record(run_id="run_0002", task="two"))
    assert [r["run_id"] for r in store.list_runs()] == ["run_0002", "run_0001"]


def test_search_matches_task(tmp_path):
    store = RunStore(tmp_path)
    store.write_run(_record(run_id="run_0001", task="fix the auth bug", answer="changed login.py"))
    store.write_run(_record(run_id="run_0002", task="add csv report", answer="wrote report.md"))
    hits = store.search("auth")
    assert [h["run_id"] for h in hits] == ["run_0001"]


def test_read_missing_returns_none(tmp_path):
    assert RunStore(tmp_path).read_run("nope") is None


def test_write_redacts_credentials(tmp_path):
    store = RunStore(tmp_path)
    store.write_run(
        _record(
            task="use api_key=supersecret",
            trace=[{"tool": "bash", "args": "Authorization: Bearer abc123"}],
        )
    )
    raw = next((tmp_path / ".pirl" / "runs").glob("*.json")).read_text()
    assert "supersecret" not in raw
    assert "abc123" not in raw
    assert "[REDACTED]" in raw
