from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_vq2_gate2_policy_n522_bounded.ps1"


def test_n522_bounded_pins_corrected_geometry_whole_puffer_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f" in text
    assert "f54c885ec9934a648a06412cee2f36686a50c709d2ed9bac8fa909ffc427f3e0" in text
    assert "92053ff9860a157963cc234844a817647a7eb81af56fab51dc48ba58bdd8c1c0" in text
    assert "88f34aa346a926f165ad6191f258a84ff837962dda3c5a047c090d042ec44a0e" in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_X = "14.51880584"' in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Y = "0.41894794"' in text
    assert '$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Z = "0.47249277"' in text
    assert '$env:PUFFER_POLICY_GATE2_RESEED_BEARING_ON_ASSOCIATION = "0"' in text
    assert '"--official-reset-on-start"' in text
    assert '"--target-gate-count", "2"' in text
    assert '"--stop-after-official-gate-index", "2"' in text
    assert '"--min-gate-passes", "2"' in text
    assert "ValidateRange(12, 14)" in text
    assert "complete recurrent Puffer checkpoints" in text
    assert "--policy-shadow-only" not in text
    assert "Submission" not in text
