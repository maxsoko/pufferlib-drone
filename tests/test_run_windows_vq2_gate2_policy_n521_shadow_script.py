from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_n521_shadow.ps1"


def test_n521_shadow_pins_corrected_geometry_puffer_and_sends_no_control():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f" in text
    assert "f54c885ec9934a648a06412cee2f36686a50c709d2ed9bac8fa909ffc427f3e0" in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_X = "14.51880584"' in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Y = "0.41894794"' in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Z = "0.47249277"' in text
    assert '$env:PUFFER_POLICY_GATE2_RESEED_BEARING_ON_ASSOCIATION = "0"' in text
    assert '"--policy-shadow-only"' in text
    assert '"--no-arm-on-start"' in text
    assert '"--gate2-checkpoint"' in text
    assert "--official-reset-on-start" not in text
    assert "--target-gate-count" not in text
    assert "Submission" not in text
