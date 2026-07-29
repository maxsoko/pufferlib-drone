from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_n482_shadow.ps1"


def test_n482_shadow_pins_fixed_state_puffer_contract_and_sends_no_control():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "1df2a68feac5e4b3f0c3ed3a25992114d6e6c95e28f4acb33a0cb453257b4ba2" in text
    assert "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_X" in text
    assert "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Z" in text
    assert '"--policy-shadow-only"' in text
    assert '"--no-arm-on-start"' in text
    assert '"--gate2-checkpoint"' in text
    assert "--official-reset-on-start" not in text
    assert "--target-gate-count" not in text
    assert "Submission" not in text
