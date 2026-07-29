from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_vq2_gate2_assisted_trace.sh"


def test_gate2_assisted_trace_is_native_only_and_parameterized():
    text = SCRIPT.read_text()

    assert "capture_native_policy_trace.py" in text
    assert 'TEACHER_BLEND="${TEACHER_BLEND:-0.20}"' in text
    assert 'GATE_RADIUS="${GATE_RADIUS:-2.0}"' in text
    assert "--env.start-gate-index=1" in text
    assert "--env.gate1-x=14.74" in text
    assert "--env.teacher-thrust-world-frame=1" in text
    assert "--env.teacher-yaw-control=1" in text
    assert "--env.observable-gate-index-denominator=6" in text
    assert "14550" not in text
    assert "5600" not in text
    assert "31000" not in text
