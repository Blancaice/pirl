from pirl.proposal import ToolCall
from pirl.plugins.skill_learning import (
    FailureCandidateStore,
    SkillLearningPlugin,
    classify_failure,
)


def test_candidate_requires_repeated_verifier_backed_failures(tmp_path):
    calls = []
    store = FailureCandidateStore(tmp_path / "failure-candidates.json")
    plugin = SkillLearningPlugin(
        store, lambda trace, message: calls.append((trace, message)) or "proposal",
        threshold=3,
    )
    kwargs = dict(trace=[ToolCall("pytest", "{}")], outcome="failure",
                  message="assertion failed", verification_failed=True)
    plugin.on_run_end(**kwargs, run_id="run-1")
    assert calls == []
    plugin.on_run_end(**kwargs, run_id="run-2")
    assert calls == []

    # A new plugin instance proves the count is persisted across processes/runs.
    plugin = SkillLearningPlugin(store, lambda trace, message: calls.append((trace, message)) or "proposal", threshold=3)
    plugin.on_run_end(**kwargs, run_id="run-3")
    assert len(calls) == 1


def test_candidate_store_deduplicates_same_run(tmp_path):
    store = FailureCandidateStore(tmp_path / "candidates.json")
    assert store.record("sig", "failure", "same-run") == 1
    assert store.record("sig", "failure", "same-run") == 1


def test_transient_failures_are_excluded(tmp_path):
    plugin = SkillLearningPlugin(
        FailureCandidateStore(tmp_path / "candidates.json"),
        lambda trace, message: (_ for _ in ()).throw(AssertionError("must not generate")),
        threshold=2,
    )
    for reason, category in [
        ("connection reset", "infra"),
        ("invalid api key", "auth"),
        ("request timed out", "timeout"),
        ("cancelled", "cancel"),
    ]:
        assert classify_failure(reason) == category
        plugin.on_run_end(trace=[], outcome="failure", message=reason,
                          verification_failed=True, run_id=reason)
    assert not (tmp_path / "candidates.json").exists()


def test_non_verifier_failure_is_not_a_learning_candidate(tmp_path):
    plugin = SkillLearningPlugin(
        FailureCandidateStore(tmp_path / "candidates.json"),
        lambda trace, message: "unexpected",
    )
    plugin.on_run_end(trace=[], outcome="failure", message="agent failed",
                      verification_failed=False, run_id="r1")
    assert not (tmp_path / "candidates.json").exists()
