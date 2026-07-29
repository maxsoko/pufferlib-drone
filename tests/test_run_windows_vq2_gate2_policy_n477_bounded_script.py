from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_n477_bounded.ps1"


def test_n477_launcher_is_puffer_only_calibrated_and_bounded_to_training_gate2():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "policy_callable_vq2_gate2_composite.py" in text
    assert "c39d3c653e33a2dbeda3eb90869fd65ae35a6ca1825380a876a8111b0b8341e6" in text
    assert "b9705f20354106de809f524acb568ada952fa090c4b497a2d46a437c18dabd68" in text
    assert "PUFFER_POLICY_GATE2_FIXED_PREFIX_REPORT_PATH" in text
    assert "PUFFER_POLICY_GATE2_PREDICT_GATE_KINEMATICS" in text
    assert '"--official-reset-on-start"' in text
    assert '"--target-gate-count", "2"' in text
    assert '"--stop-after-official-gate-index", "2"' in text
    assert '"--min-gate-passes", "2"' in text
    assert '"--require-official-race-progress"' in text
    assert "ValidateRange(12, 14)" in text
    assert "--policy-shadow-only" not in text
    assert "Submission" not in text
