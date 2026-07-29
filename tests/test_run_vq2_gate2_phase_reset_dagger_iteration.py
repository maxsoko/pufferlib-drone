from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_vq2_gate2_phase_reset_dagger_iteration.sh"


def test_gate2_phase_reset_training_uses_only_native_teacher_state():
    text = SCRIPT.read_text()

    assert "START_GATE_INDEX=1" in text
    assert "START_ELAPSED_TIME=3.25" in text
    assert "START_VX=4.677" in text
    assert "START_QW=0.998520" in text
    assert "GATE1_X=14.74 GATE1_Y=8.70 GATE1_Z=1.095" in text
    assert "OBSERVABLE_GATE_INDEX_DENOMINATOR=6" in text
    assert "POLICY_TEACHER=1" in text
    assert "TEACHER_KEEP_SUCCESSFUL_EPISODES_ONLY" in text
    assert "--min-race-phase 0.166" in text
    assert "--max-race-phase 0.167" in text
    assert "--source-anchor-outside-active" not in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
