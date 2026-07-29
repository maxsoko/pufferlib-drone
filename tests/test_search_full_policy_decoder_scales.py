import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "search_full_policy_decoder_scales.py"
)
SPEC = importlib.util.spec_from_file_location(
    "search_full_policy_decoder_scales", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_score_prefers_safe_success_then_course_progress():
    safe = {
        "env/crash": 0.0,
        "env/success_rate": 0.8,
        "env/gates_passed": 3.8,
        "env/avg_terminal_crossing_radial": 1.0,
    }
    unsafe = {**safe, "env/crash": 0.2, "env/success_rate": 0.95}
    more_progress = {**safe, "env/gates_passed": 3.9}

    assert MODULE.candidate_score(safe, 0.1) > MODULE.candidate_score(unsafe, 0.1)
    assert MODULE.candidate_score(more_progress, 0.1) > MODULE.candidate_score(safe, 0.1)


def test_candidate_score_supports_current_terminal_radial_metric():
    wider = {
        "env/crash": 0.0,
        "env/success_rate": 0.8,
        "env/gates_passed": 3.8,
        "env/terminal_crossing_radial": 9.9,
    }
    tighter = {**wider, "env/terminal_crossing_radial": 9.8}
    assert MODULE.candidate_score(tighter, 0.1) > MODULE.candidate_score(wider, 0.1)


def test_passes_requires_both_success_and_crash_thresholds():
    assert MODULE.passes({"env/success_rate": 0.9, "env/crash": 0.1}, 0.9, 0.1)
    assert not MODULE.passes(
        {"env/success_rate": 0.899, "env/crash": 0.0}, 0.9, 0.1
    )
    assert not MODULE.passes(
        {"env/success_rate": 1.0, "env/crash": 0.101}, 0.9, 0.1
    )


def test_scale_tag_is_filename_safe_and_stable():
    assert MODULE.scale_tag(1.125) == "1p125"
    assert MODULE.scale_tag(-0.5) == "m0p5"
