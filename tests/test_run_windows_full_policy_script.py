from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_full_policy.ps1"


def test_checkpoint_paths_are_native_paths_for_wsl_unc_worktrees() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "function Resolve-NativeFilePath" in source
    assert "$resolved.ProviderPath" in source
    assert (
        "$env:PUFFER_POLICY_CHECKPOINT_PATH = "
        "Resolve-NativeFilePath $PrimaryCheckpoint"
    ) in source
    assert "$env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = " in source
    assert '$env:PUFFER_POLICY_NATIVE_BF16 = "0"' in source
    assert "(Resolve-Path $PolicyCheckpoint).Path" not in source
    assert source.count("Resolve-NativeFilePath $") == 11


def test_six_gate_progress_and_action_bias_are_forwarded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "[switch]$PolicyGateProgressAdapterObservation" in source
    assert "[switch]$PolicyGatePhaseOnehotAdapterObservation" in source
    assert "[int]$PolicyGateProgressDenominator = 6" in source
    assert "[int]$PolicyActionBiasGateIndex = -1" in source
    assert '[string]$PolicyActionBiasTableJson = ""' in source
    assert '$env:PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX = "23"' in source
    assert '$env:PUFFER_POLICY_ACTION_BIAS_ROLL = "$PolicyActionBiasRoll"' in source
    assert '$env:PUFFER_POLICY_ACTION_BIAS_THRUST = "$PolicyActionBiasThrust"' in source
    assert "$env:PUFFER_POLICY_ACTION_BIAS_TABLE_JSON = $PolicyActionBiasTableJson" in source
    assert '"--policy-gate-progress-adapter-observation"' in source
    assert '"--policy-gate-phase-onehot-adapter-observation"' in source
    assert '"--policy-race-phase-denominator", "$PolicyGateProgressDenominator"' in source
    assert "[int]$CommandHz = 60" in source
    assert '"--command-hz", "$CommandHz"' in source
    assert "[int]$PolicyStateHz = 0" in source
    assert '"--policy-state-hz", "$PolicyStateHz"' in source


def test_six_gate_composite_is_fail_closed_and_bounded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "[switch]$SixGateComposite" in source
    assert '[string]$Gate5Checkpoint = ""' in source
    assert '[string]$Gate6Checkpoint = ""' in source
    assert '[int]$StopAfterOfficialGateIndex = -1' in source
    assert "SixGateComposite requires Gate5Checkpoint" in source
    assert "SixGateComposite requires Gate6Checkpoint" in source
    assert "SixGateComposite requires FP32 checkpoint layout precision (4 bytes)" in source
    assert "external action bias is forbidden" in source
    assert (
        '$policyCallable = "scripts/policy_callable_six_gate_composite.py:infer"'
        in source
    )
    assert (
        '$env:PUFFER_POLICY_GATE4_CHECKPOINT_PATH =\n'
        '        Resolve-NativeFilePath $PolicyCheckpoint'
    ) in source
    assert (
        '$env:PUFFER_POLICY_GATE5_CHECKPOINT_PATH =\n'
        '        Resolve-NativeFilePath $Gate5Checkpoint'
    ) in source
    assert (
        '$env:PUFFER_POLICY_GATE6_CHECKPOINT_PATH =\n'
        '        Resolve-NativeFilePath $Gate6Checkpoint'
    ) in source
    assert '"--stop-after-official-gate-index", "$StopAfterOfficialGateIndex"' in source


def test_hybrid_prefix_is_explicit_and_forwards_exact_confidence_bridge() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[string]$HybridPrefixCheckpoint = ""' in source
    assert '[string]$HybridPolicyCallable = ""' in source
    assert "HybridPrefixCheckpoint requires SixGateComposite" in source
    assert "HybridPolicyCallable requires HybridPrefixCheckpoint" in source
    assert "$env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH =" in source
    assert "Resolve-NativeFilePath $HybridPrefixCheckpoint" in source
    assert '"scripts/policy_callable_six_gate_hybrid.py:infer"' in source
    assert "$HybridPolicyCallable" in source
    assert '"--policy-hybrid-prefix-confidence-observation"' in source
