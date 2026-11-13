"""CLI surface: the temporary step cap and its overrides."""

from pirl.cli import TEMP_MAX_STEPS, _build_parser


def _run_args(*extra):
    return _build_parser().parse_args(["run", "task", *extra])


def test_temporary_step_cap_is_the_default():
    assert TEMP_MAX_STEPS == 20
    assert _run_args().max_steps == TEMP_MAX_STEPS


def test_step_cap_can_be_overridden():
    assert _run_args("--max-steps", "5").max_steps == 5


def test_step_cap_can_be_disabled_with_zero():
    assert _run_args("--max-steps", "0").max_steps == 0
