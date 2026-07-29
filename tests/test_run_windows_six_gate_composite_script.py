from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_six_gate_composite.ps1"


def test_shadow_is_default_and_sends_no_control_lifecycle_flags() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[string]$Mode = "Shadow"' in source
    assert '"--policy-shadow-only"' in source
    assert '"--no-arm-on-start"' in source
    assert '"--policy-gate-phase-onehot-adapter-observation"' in source
    assert '"--policy-race-phase-denominator", "6"' in source
    assert "[int]$CommandHz = 80" in source
    assert '"--command-hz", "$CommandHz"' in source
    shadow_block = source.split('if ($Mode -eq "Shadow") {', 1)[1].split(
        '$TargetGateCount =', 1
    )[0]
    assert '"--send-sim-reset"' not in shadow_block
    assert '"--official-reset-on-start"' not in shadow_block
    assert "run_windows_full_policy.ps1" not in shadow_block


def test_artifacts_are_frozen_and_flight_modes_are_explicit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[ValidateSet("Shadow", "BoundedGate4", "FullLap")]' in source
    assert source.count("Assert-SHA256 $") == 4
    assert "9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca" in source
    assert "e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30" in source
    assert "4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354" in source
    assert "9eaab39694f67fdae6ebb6be52af9a282974e6778822fd84cd194ac5ade5d2a3" in source
    assert '$TargetGateCount = if ($Mode -eq "BoundedGate4") { 4 } else { 6 }' in source
    assert '$StopAfterOfficialGateIndex = if ($Mode -eq "BoundedGate4") { 4 } else { -1 }' in source
    assert "PolicyLayoutPrecisionBytes = 4" in source
    assert "SixGateComposite = $true" in source
    assert "CommandHz = $CommandHz" in source
