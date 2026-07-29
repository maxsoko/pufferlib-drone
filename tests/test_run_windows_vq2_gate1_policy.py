from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate1_policy.ps1"


def test_vq2_gate1_wrapper_pins_policy_and_bounds_control():
    text = SCRIPT.read_text()

    assert 'ValidateSet("Shadow", "Bounded")' in text
    assert "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6" in text
    assert "AI-GP Simulator v1.0.3391\\AIGP_3391\\FlightSim.exe" in text
    assert '"--policy-shadow-only", "--no-arm-on-start"' in text
    assert '"--official-reset-on-start"' in text
    assert '"--stop-after-official-gate-index", "1"' in text
    assert '"--target-gate-count", "1"' in text
    assert "DurationSeconds -gt 12" in text
    assert "refusing to overwrite existing attempt evidence" in text
    assert '"--command-hz", "80"' in text
    assert '"--camera-uptilt-deg", "1.920944634732011"' in text
