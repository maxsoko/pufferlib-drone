from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_n483_bounded.ps1"


def test_n483_bounded_pins_fixed_state_whole_puffer_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "1df2a68feac5e4b3f0c3ed3a25992114d6e6c95e28f4acb33a0cb453257b4ba2" in text
    assert "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_X" in text
    assert '"--official-reset-on-start"' in text
    assert '"--target-gate-count", "2"' in text
    assert '"--stop-after-official-gate-index", "2"' in text
    assert '"--min-gate-passes", "2"' in text
    assert "ValidateRange(12, 14)" in text
    assert "--policy-shadow-only" not in text
    assert "Submission" not in text
