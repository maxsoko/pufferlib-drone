from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_vq2_gate2_dagger_iteration.sh"


def test_vq2_gate2_dagger_keeps_privileged_teacher_off_deployment_path():
    text = SCRIPT.read_text()

    assert "collect_full_policy_teacher" in text
    assert "train_full_policy_bc.py" in text
    assert "GATE0_X=10.78 GATE0_Y=0.084 GATE0_Z=0.56" in text
    assert "GATE1_X=25.13 GATE1_Y=8.96 GATE1_Z=1.65" in text
    assert "OBSERVABLE_GATE_INDEX_DENOMINATOR=6" in text
    assert "POLICY_TEACHER=1" in text
    assert "TEACHER_FEEDBACK_FROM_GATE=1" in text
    assert "GATE_POSITION_RANDOMIZE_FROM_INDEX=1" in text
    assert "--min-race-phase 0.166" in text
    assert "--max-race-phase 0.167" in text
    assert "--source-anchor-outside-active" in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
