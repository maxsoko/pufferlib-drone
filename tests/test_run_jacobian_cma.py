from pathlib import Path
import json

import numpy as np
import pytest

from scripts.run_jacobian_cma import (
    build_evaluation_command,
    candidate_rank_key,
    evaluation_cache_key,
    experiment_stop_reason,
    has_enough_feasible_parents,
    load_generation_summaries,
    outcome_metrics,
)


def _report(*, success=0.0, gates=2.0, crash=0.0, crossing=0.0, radial=1.0, progress=0.68):
    return {
        "metrics": {
            "env/success_rate": success,
            "env/gates_passed": gates,
            "env/crash": crash,
            "env/gate2_crossing_sampled": crossing,
            "env/gate2_terminal_crossing_sampled": 0.0 if crossing else 1.0,
            "env/avg_gate2_terminal_crossing_radial": radial,
            "env/avg_gate2_crossing_radial": radial,
            "env/final_progress": progress,
        }
    }


def _record(report, feasible=True, drift=0.0005):
    return {
        "trace_feasible": feasible,
        "anchor_actual_feasible": None,
        "development_target": report,
        "anchor_drift": {"full_rms": drift, "focus_rms": drift},
    }


def test_rank_is_lexicographic_and_crossing_beats_radial_only():
    crossing = _record(_report(crossing=0.1, radial=0.7))
    radial = _record(_report(crossing=0.0, radial=0.2))
    infeasible = _record(_report(success=1.0, gates=4.0), feasible=False)
    assert candidate_rank_key(crossing) > candidate_rank_key(radial)
    assert candidate_rank_key(radial) > candidate_rank_key(infeasible)


def test_outcome_metrics_uses_infinity_without_relevant_sample():
    values = outcome_metrics(_report(crossing=0.0, radial=0.8))
    assert values["terminal_radial"] == pytest.approx(0.8)
    assert np.isinf(values["crossing_radial"])


def test_evaluation_command_uses_exact_disjoint_offset(tmp_path):
    command = build_evaluation_command(
        tmp_path / "policy.bin",
        tmp_path / "report.json",
        tmp_path / "report.csv",
        floor=4.5,
        episodes=128,
        episode_offset=1,
        total_agents=128,
        label="heldout",
    )
    assert "--require-exact-episodes" in command
    assert command[command.index("--episode-offset") + 1] == "1"
    assert command[command.index("--env.sitl-gate-transition-from-gate-index") + 1] == "2"
    assert command[command.index("--env.sitl-gate-transition-min-forward-speed") + 1] == "4.5"


def test_cache_key_includes_checkpoint_content_and_configuration(tmp_path):
    checkpoint = tmp_path / "policy.bin"
    checkpoint.write_bytes(b"one")
    first = evaluation_cache_key(checkpoint, {"floor": 4.5})
    second = evaluation_cache_key(checkpoint, {"floor": 4.0})
    checkpoint.write_bytes(b"two")
    third = evaluation_cache_key(checkpoint, {"floor": 4.5})
    assert len({first, second, third}) == 3


def test_rejected_candidates_cannot_fill_cma_parent_set():
    records = [{"trace_feasible": index < 11} for index in range(24)]
    assert not has_enough_feasible_parents(records, 12)
    records[11]["trace_feasible"] = True
    assert has_enough_feasible_parents(records, 12)


def test_stop_reason_distinguishes_constraint_and_budget_exhaustion():
    assert experiment_stop_reason(
        constraint_stopped=True,
        promoted=None,
        generation=3,
        maximum_generations=8,
    ) == "insufficient_feasible_parents_under_anchor_trust_region"
    assert experiment_stop_reason(
        constraint_stopped=False,
        promoted=None,
        generation=8,
        maximum_generations=8,
    ) == "maximum_generation_budget_exhausted_without_promotion"
    assert experiment_stop_reason(
        constraint_stopped=False,
        promoted={"checkpoint": "candidate.bin"},
        generation=8,
        maximum_generations=8,
    ) is None


def test_generation_summary_index_is_reconstructed_across_resumes(tmp_path):
    expected = [{"generation": 0}, {"generation": 1}]
    for summary in reversed(expected):
        path = tmp_path / f"generation_{summary['generation']:03d}" / "summary.json"
        path.parent.mkdir()
        path.write_text(json.dumps(summary), encoding="utf-8")
    assert load_generation_summaries(tmp_path) == expected
