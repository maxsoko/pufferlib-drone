from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_n220_dagger_iteration.sh"
)


def test_n220_dagger_uses_exact_tail_sensor_and_teacher_contract():
    source = SCRIPT.read_text(encoding="utf-8")

    required = (
        "START_GATE_INDEX=3",
        "START_X_JITTER=6 START_Y_JITTER=5 START_Z_JITTER=2",
        "GATE5_X=148 GATE5_Y=5.10711 GATE5_Z=-18.24322",
        "SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS=4",
        "SITL_GATE_MOTION_PREDICT_DROPOUT=1",
        "SITL_GATE_MOTION_CONTROL_ACCEL_GAIN=4.295",
        "POLICY_TEACHER=1",
        "TEACHER_SPLINE_LABEL_FROM_GATE=3",
        "TEACHER_SPLINE_FULL_COURSE_ORIGIN=1",
        'TEACHER_SPLINE_CLOSE_LATERAL_POSITION_KP="${TEACHER_SPLINE_CLOSE_LATERAL_POSITION_KP:-5.0}"',
        '"$STUDENT_PROBABILITY" 0 0',
    )
    for expected in required:
        assert expected in source


def test_n220_dagger_defaults_to_mixed_student_teacher_coverage():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'EPISODES="${EPISODES:-256}"' in source
    assert 'STUDENT_PROBABILITY="${STUDENT_PROBABILITY:-0.5}"' in source
    assert "TEACHER_KEEP_SUCCESSFUL_EPISODES_ONLY" not in source
    assert "TEACHER_RECORD_EXECUTED_ACTION" not in source
