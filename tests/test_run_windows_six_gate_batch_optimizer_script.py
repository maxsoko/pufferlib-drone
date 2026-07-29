from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_six_gate_batch_optimizer.ps1"


def test_batch_is_six_gate_risk_sensitive_and_dry_by_default():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '[switch]$Execute' in source
    assert 'if (-not $Execute)' in source
    assert 'Mode = "FullLap"' in source
    assert 'target_gate_count = 6' in source
    assert '"valid_finish_count"' in source
    assert '"minimum_official_gate_count"' in source
    assert '"median_official_gate_count"' in source
    assert '"collision_free_count"' in source
    assert '"lap_or_attempt_time"' in source


def test_batch_uses_best_frozen_variants_and_one_lifecycle_per_attempt():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '@("N195", "N197", "N198")' in source
    assert 'Gate2FrozenObservationDropoutN194' in source
    assert 'Gate3ZeroConfidenceRecoveryN196' in source
    assert 'Gate3FinalTerminalLevelN197' in source
    assert 'Gate3TightTerminalLevelN201' in source
    assert 'Gate3CounterHysteresis08N202' in source
    assert 'Gate3CounterHysteresis12N202' in source
    assert 'Gate3CounterHysteresis16N202' in source
    assert 'Gate4YawSignN203' in source
    assert 'Gate4CoherentTargetHoldN203' in source
    assert 'Gate4LearnedTailN203' in source
    assert 'Gate4TerminalCrossingN206' in source
    assert 'Gate4FilteredTerminalCrossingN206' in source
    assert 'PF = "Gate4PhasePredictiveTerminalN206"' in source
    assert 'RF = "Gate4TerminalCrossingN206"' in source
    assert 'MO = "Gate4VisualMpcN232"' in source
    assert 'MC = "Gate4VisualMpcFixedN233"' in source
    assert 'MR = "Gate4ReacquiringMpcN234"' in source
    assert 'MS = "Gate4SearchMpcN235"' in source
    assert 'MB = "Gate4StagedMpcN236"' in source
    assert 'MI = "Gate4FreshProjectedInterceptN237"' in source
    assert 'IL = "Gate4IdentityLockedInterceptN238"' in source
    assert 'LM = "Gate4IdentityBodyMpcN239"' in source
    assert 'policy_state_hz_by_variant = [ordered]@{ PF = 60; RF = 60; MO = 0; MC = 60; MR = 60; MS = 60; MB = 60; MI = 60; IL = 60; LM = 60 }' in source
    assert '$plan.variant -in @("PF", "RF", "MC", "MR", "MS", "MB", "MI", "IL", "LM")' in source
    assert '$runArgs.PolicyStateHz = 60' in source
    assert 'Repeats = 1' in source
    assert 'MinValidRuns = 1' in source
    assert 'run_windows_six_gate_hybrid.ps1' in source
    assert 'debug_official_reset_snapshot.py' in source
    assert '"--duration-s", "5"' in source
    assert '"--output-dir", $poststopRelative' in source
    assert 'send-sim-reset' not in source.split('$probeArgs = @(', 1)[1].split(')', 1)[0]


def test_batch_is_resumable_and_stops_on_first_valid_finish():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'Skipping already recorded attempt' in source
    assert '[switch]$ContinueAfterFinish' in source
    assert '[int]$leader.valid_finishes -gt 0' in source
    assert 'Valid six-gate finish found; stopping batch immediately.' in source
    assert 'score_official_six_gate_batch.py' in source
    assert 'Assert-SHA256 $HybridRunner' in source
    assert 'Assert-SHA256 $PoststopProbe' in source
    assert 'Assert-SHA256 $Scorer' in source
