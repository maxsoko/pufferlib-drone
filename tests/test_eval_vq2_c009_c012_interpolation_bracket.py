from __future__ import annotations

from scripts import eval_vq2_c009_c012_interpolation_bracket as bracket


def test_candidate_names_are_stable():
    assert [bracket.candidate_name(alpha) for alpha in bracket.ALPHAS] == [
        "c014_alpha_0p25",
        "c014_alpha_0p50",
        "c014_alpha_0p75",
    ]


def test_admission_requires_perfect_next_gate_completion():
    report = {
        "diagnostic_valid": True,
        "agents": 128,
        "reached_next_gate": 128,
        "failed_before_next_gate": 0,
        "native_metrics_diagnostic_only": {
            "env/crash": 0.0,
            "env/timeout": 0.0,
            "env/missed_gate": 0.0,
            "env/out_of_order": 0.0,
        },
    }
    assert bracket.bracket_passes(report)
    report["reached_next_gate"] = 127
    assert not bracket.bracket_passes(report)
