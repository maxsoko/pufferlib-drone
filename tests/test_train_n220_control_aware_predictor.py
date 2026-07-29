from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_n220_control_aware_predictor.sh"


def test_n220_uses_camera_sampling_and_control_aware_prediction():
    source = TRAIN.read_text(encoding="utf-8")
    assert "--env.sitl-gate-obs-sample-interval-steps 4" in source
    assert "--env.sitl-gate-motion-predict-dropout 1" in source
    assert "--env.sitl-gate-motion-control-accel-gain 4.295" in source


def test_n220_intervention_is_explicitly_training_only():
    source = TRAIN.read_text(encoding="utf-8")
    assert 'TEACHER_BLEND="${TEACHER_BLEND:-0.5}"' in source
    assert '--env.teacher-action-blend "$TEACHER_BLEND"' in source
    assert "every promotion screen is unassisted" in source
