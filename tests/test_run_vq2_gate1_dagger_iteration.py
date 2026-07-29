from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_vq2_gate1_dagger_iteration.sh"


def test_vq2_gate1_dagger_confines_teacher_to_native_training():
    text = SCRIPT.read_text()

    assert "collect_full_policy_teacher" in text
    assert "train_full_policy_bc.py" in text
    assert "GATE0_X=10.78 GATE0_Y=0.084 GATE0_Z=0.56" in text
    assert "OBSERVABLE_GATE_INDEX_DENOMINATOR=6" in text
    assert "SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS=4" in text
    assert "TEACHER_GATE_DROPOUT_RANGE=4.25" in text
    assert "START_Z_JITTER=0.15" in text
    assert "GATE_POSITION_JITTER_Z=0.20" in text
    assert "STUDENT_PROBABILITY" in text
    assert "EXTRA_DATASET" in text
    assert "--gate-progress-observation" in text
    assert "--max-race-phase 0" in text
    assert "--source-anchor-outside-active" in text
    assert "--checkpoint-layout-precision-bytes 4" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
