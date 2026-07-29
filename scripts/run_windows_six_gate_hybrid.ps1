param(
    [ValidateSet("Shadow", "BoundedGate3", "BoundedGate4", "FullLap")]
    [string]$Mode = "Shadow",
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
    [string]$Tag = "n184_six_gate_hybrid",
    [int]$ShadowDuration = 10,
    [int]$FlightDuration = 45,
    [int]$CommandHz = 80,
    [int]$PolicyStateHz = 0,
    [int]$Repeats = 1,
    [int]$MinValidRuns = 1,
    [switch]$Gate3ProjectedPathPD,
    [switch]$Gate3SevereAcquisition,
    [switch]$Gate3CompositeAcquisitionPD,
    [switch]$Gate2VerticalMarginN189,
    [switch]$Gate2SevereVerticalN190,
    [switch]$Gate3TerminalLevelN191,
    [switch]$Gate2RapidVerticalDeficitN192,
    [switch]$Gate3FrozenObservationRecoveryN193,
    [switch]$Gate2FrozenObservationDropoutN194,
    [switch]$Gate3LaterTerminalLevelN195,
    [switch]$Gate3ZeroConfidenceRecoveryN196,
    [switch]$Gate3FinalTerminalLevelN197,
    [switch]$Gate3LowConfidenceLatchedRecoveryN198,
    [switch]$Gate3RestoredTerminalLevelN199,
    [switch]$Gate1HighDownThrustCeilingN200,
    [switch]$Gate3TightTerminalLevelN201,
    [switch]$Gate3CounterHysteresis08N202,
    [switch]$Gate3CounterHysteresis12N202,
    [switch]$Gate3CounterHysteresis16N202,
    [switch]$Gate4YawSignN203,
    [switch]$Gate4CoherentTargetHoldN203,
    [switch]$Gate4LearnedTailN203,
    [switch]$Gate4TerminalCrossingN206,
    [switch]$Gate4FilteredTerminalCrossingN206,
    [switch]$Gate4PhasePredictiveTerminalN206,
    [switch]$Gate4VisualMpcN232,
    [switch]$Gate4VisualMpcFixedN233,
    [switch]$Gate4ReacquiringMpcN234,
    [switch]$Gate4SearchMpcN235,
    [switch]$Gate4StagedMpcN236,
    [switch]$Gate4FreshProjectedInterceptN237,
    [switch]$Gate4IdentityLockedInterceptN238,
    [switch]$Gate4IdentityBodyMpcN239,
    [switch]$Gate4RollProbeN241,
    [switch]$ForceRelaunch
)

# Frozen N184 deployment entry point. Gates 1--3 use the historically proven
# 23-input v6c prefix, with exact legacy confidence and elapsed time carried in
# reserved slots 30/31 while current field 22 preserves the recurrent policy's
# previous yaw action. Gates 4--6 use the independently reset N145 32-input
# tail. Shadow is the default and sends no reset, arm, setpoint, or disarm
# command.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$SelectedPolicyVariants = @(
    $Gate3ProjectedPathPD,
    $Gate3SevereAcquisition,
    $Gate3CompositeAcquisitionPD,
    $Gate2VerticalMarginN189,
    $Gate2SevereVerticalN190,
    $Gate3TerminalLevelN191,
    $Gate2RapidVerticalDeficitN192,
    $Gate3FrozenObservationRecoveryN193,
    $Gate2FrozenObservationDropoutN194,
    $Gate3LaterTerminalLevelN195,
    $Gate3ZeroConfidenceRecoveryN196,
    $Gate3FinalTerminalLevelN197,
    $Gate3LowConfidenceLatchedRecoveryN198,
    $Gate3RestoredTerminalLevelN199,
    $Gate1HighDownThrustCeilingN200,
    $Gate3TightTerminalLevelN201,
    $Gate3CounterHysteresis08N202,
    $Gate3CounterHysteresis12N202,
    $Gate3CounterHysteresis16N202,
    $Gate4YawSignN203,
    $Gate4CoherentTargetHoldN203,
    $Gate4LearnedTailN203,
    $Gate4TerminalCrossingN206,
    $Gate4FilteredTerminalCrossingN206,
    $Gate4PhasePredictiveTerminalN206,
    $Gate4VisualMpcN232,
    $Gate4VisualMpcFixedN233,
    $Gate4ReacquiringMpcN234,
    $Gate4SearchMpcN235,
    $Gate4StagedMpcN236,
    $Gate4FreshProjectedInterceptN237,
    $Gate4IdentityLockedInterceptN238,
    $Gate4IdentityBodyMpcN239,
    $Gate4RollProbeN241
) | Where-Object { $_ }
if ($SelectedPolicyVariants.Count -gt 1) {
    throw "Select only one policy variant"
}

if (
    ($Gate3ProjectedPathPD -and $Gate3SevereAcquisition) -or
    ($Gate3ProjectedPathPD -and $Gate3CompositeAcquisitionPD) -or
    ($Gate3SevereAcquisition -and $Gate3CompositeAcquisitionPD) -or
    ($Gate2VerticalMarginN189 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate2SevereVerticalN190 -or
        $Gate3TerminalLevelN191 -or
        $Gate2RapidVerticalDeficitN192 -or
        $Gate3FrozenObservationRecoveryN193 -or
        $Gate2FrozenObservationDropoutN194
    )) -or
    ($Gate2SevereVerticalN190 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate3TerminalLevelN191 -or
        $Gate2RapidVerticalDeficitN192 -or
        $Gate3FrozenObservationRecoveryN193 -or
        $Gate2FrozenObservationDropoutN194
    )) -or
    ($Gate3TerminalLevelN191 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate2RapidVerticalDeficitN192 -or
        $Gate3FrozenObservationRecoveryN193 -or
        $Gate2FrozenObservationDropoutN194
    )) -or
    ($Gate2RapidVerticalDeficitN192 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate3FrozenObservationRecoveryN193 -or
        $Gate2FrozenObservationDropoutN194
    )) -or
    ($Gate3FrozenObservationRecoveryN193 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate2VerticalMarginN189 -or
        $Gate2SevereVerticalN190 -or
        $Gate3TerminalLevelN191 -or
        $Gate2RapidVerticalDeficitN192 -or
        $Gate2FrozenObservationDropoutN194
    )) -or
    ($Gate2FrozenObservationDropoutN194 -and (
        $Gate3ProjectedPathPD -or
        $Gate3SevereAcquisition -or
        $Gate3CompositeAcquisitionPD -or
        $Gate2VerticalMarginN189 -or
        $Gate2SevereVerticalN190 -or
        $Gate3TerminalLevelN191 -or
        $Gate2RapidVerticalDeficitN192 -or
        $Gate3FrozenObservationRecoveryN193
    ))
) {
    throw "Select only one policy variant"
}

function Resolve-NativeFilePath([string]$Path) {
    $resolved = Resolve-Path -LiteralPath $Path
    if ($resolved.ProviderPath) {
        return $resolved.ProviderPath
    }
    return $resolved.Path
}

function Assert-SHA256([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required deployment artifact not found: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected, got $actual"
    }
}

if (
    $Gate4YawSignN203 -or
    $Gate4CoherentTargetHoldN203 -or
    $Gate4LearnedTailN203 -or
    $Gate4TerminalCrossingN206 -or
    $Gate4FilteredTerminalCrossingN206 -or
    $Gate4PhasePredictiveTerminalN206 -or
    $Gate4VisualMpcN232 -or
    $Gate4VisualMpcFixedN233 -or
    $Gate4ReacquiringMpcN234 -or
    $Gate4SearchMpcN235 -or
    $Gate4StagedMpcN236 -or
    $Gate4FreshProjectedInterceptN237 -or
    $Gate4IdentityLockedInterceptN238 -or
    $Gate4IdentityBodyMpcN239 -or
    $Gate4RollProbeN241
) {
    $env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.2"
} elseif ($Gate3CounterHysteresis08N202) {
    $env:PUFFER_GATE3_COUNTER_ENTRY_M = "-0.8"
} elseif ($Gate3CounterHysteresis12N202) {
    $env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.2"
} elseif ($Gate3CounterHysteresis16N202) {
    $env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.6"
} else {
    Remove-Item Env:PUFFER_GATE3_COUNTER_ENTRY_M -ErrorAction SilentlyContinue
}

$PrefixCheckpoint = (
    "C:\Users\anon\code\pufferlib-drone\checkpoints\" +
    "drone_race_full_policy_gate3_visual\" +
    "v6c_latest_obsmatch_r135_dropout_e10.bin"
)
$Gate4Checkpoint = Join-Path $RepoRoot (
    "logs\drone_race_full_policy_six_gate_bootstrap\n112_live_gate3_parent\" +
    "checkpoints\drone_race_full_policy_six_gate_bootstrap\1784314517029\" +
    "0000000000294912.bin"
)
$Gate5Checkpoint = Join-Path $RepoRoot (
    "logs\drone_race_full_policy_six_gate_bootstrap\n118_measured_gate5_aperture\" +
    "checkpoints\drone_race_full_policy_six_gate_bootstrap\1784317785919\" +
    "0000000000065536.bin"
)
$Gate6Checkpoint = Join-Path $RepoRoot (
    "logs\drone_race_full_policy_six_gate_bootstrap\n115_backward_tail_gate6\" +
    "checkpoints\drone_race_full_policy_six_gate_bootstrap\1784316585526\" +
    "0000000000294912.bin"
)
$Gate4DynamicsModel = Join-Path $RepoRoot (
    "logs\sitl\n232_official_visual_dynamics_fit_20260719.json"
)
$Gate4IdentityDynamicsModel = Join-Path $RepoRoot (
    "logs\sitl\n239_gate4_identity_body_dynamics_fit_20260719.json"
)
$PolicyCallable = if (
    $Gate4RollProbeN241
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_roll_probe_n241.py"
} elseif (
    $Gate4IdentityBodyMpcN239
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_identity_body_mpc_n239.py"
} elseif (
    $Gate4IdentityLockedInterceptN238
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_identity_locked_intercept_n238.py"
} elseif (
    $Gate4FreshProjectedInterceptN237
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_fresh_projected_intercept_n237.py"
} elseif (
    $Gate4StagedMpcN236
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_staged_mpc_n236.py"
} elseif (
    $Gate4SearchMpcN235
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_search_mpc_n235.py"
} elseif (
    $Gate4ReacquiringMpcN234
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_reacquiring_mpc_n234.py"
} elseif (
    $Gate4VisualMpcFixedN233
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_mpc_n233.py"
} elseif (
    $Gate4VisualMpcN232
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_mpc_n232.py"
} elseif (
    $Gate4TerminalCrossingN206 -or
    $Gate4FilteredTerminalCrossingN206 -or
    $Gate4PhasePredictiveTerminalN206
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_terminal_crossing_n206.py"
} elseif ($Gate4LearnedTailN203) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_learned_tail_n203.py"
} elseif ($Gate4CoherentTargetHoldN203) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_coherent_target_hold_n203.py"
} elseif ($Gate4YawSignN203) {
    Join-Path $RepoRoot "scripts\policy_callable_gate4_yaw_sign_n203.py"
} elseif (
    $Gate3CounterHysteresis08N202 -or
    $Gate3CounterHysteresis12N202 -or
    $Gate3CounterHysteresis16N202
) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_counter_hysteresis_n202.py"
} elseif ($Gate3TightTerminalLevelN201) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_tight_terminal_level_n201.py"
} elseif ($Gate1HighDownThrustCeilingN200) {
    Join-Path $RepoRoot "scripts\policy_callable_gate1_high_down_thrust_ceiling_n200.py"
} elseif ($Gate3RestoredTerminalLevelN199) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_restored_terminal_level_n199.py"
} elseif ($Gate3LowConfidenceLatchedRecoveryN198) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_low_confidence_latched_recovery_n198.py"
} elseif ($Gate3FinalTerminalLevelN197) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_final_terminal_level_n197.py"
} elseif ($Gate3ZeroConfidenceRecoveryN196) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_zero_confidence_recovery_n196.py"
} elseif ($Gate3LaterTerminalLevelN195) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_later_terminal_level_n195.py"
} elseif ($Gate2FrozenObservationDropoutN194) {
    Join-Path $RepoRoot "scripts\policy_callable_gate2_frozen_observation_dropout_n194.py"
} elseif ($Gate3FrozenObservationRecoveryN193) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_frozen_observation_recovery_n193.py"
} elseif ($Gate2RapidVerticalDeficitN192) {
    Join-Path $RepoRoot "scripts\policy_callable_gate2_rapid_vertical_deficit_n192.py"
} elseif ($Gate3TerminalLevelN191) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_terminal_level_n191.py"
} elseif ($Gate2SevereVerticalN190) {
    Join-Path $RepoRoot "scripts\policy_callable_gate2_severe_vertical_n190.py"
} elseif ($Gate2VerticalMarginN189) {
    Join-Path $RepoRoot "scripts\policy_callable_gate2_vertical_margin_n189.py"
} elseif ($Gate3CompositeAcquisitionPD) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_composite_acquisition_pd.py"
} elseif ($Gate3SevereAcquisition) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_severe_acquisition.py"
} elseif ($Gate3ProjectedPathPD) {
    Join-Path $RepoRoot "scripts\policy_callable_gate3_projected_path_pd.py"
} else {
    Join-Path $RepoRoot "scripts\policy_callable_six_gate_hybrid.py"
}
$PolicyCallableSpec = if (
    $Gate4RollProbeN241
) {
    "scripts/policy_callable_gate4_roll_probe_n241.py:infer"
} elseif (
    $Gate4IdentityBodyMpcN239
) {
    "scripts/policy_callable_gate4_identity_body_mpc_n239.py:infer"
} elseif (
    $Gate4IdentityLockedInterceptN238
) {
    "scripts/policy_callable_gate4_identity_locked_intercept_n238.py:infer"
} elseif (
    $Gate4FreshProjectedInterceptN237
) {
    "scripts/policy_callable_gate4_fresh_projected_intercept_n237.py:infer"
} elseif (
    $Gate4StagedMpcN236
) {
    "scripts/policy_callable_gate4_staged_mpc_n236.py:infer"
} elseif (
    $Gate4SearchMpcN235
) {
    "scripts/policy_callable_gate4_search_mpc_n235.py:infer"
} elseif (
    $Gate4ReacquiringMpcN234
) {
    "scripts/policy_callable_gate4_reacquiring_mpc_n234.py:infer"
} elseif (
    $Gate4VisualMpcFixedN233
) {
    "scripts/policy_callable_gate4_mpc_n233.py:infer"
} elseif (
    $Gate4VisualMpcN232
) {
    "scripts/policy_callable_gate4_mpc_n232.py:infer"
} elseif (
    $Gate4TerminalCrossingN206 -or
    $Gate4FilteredTerminalCrossingN206 -or
    $Gate4PhasePredictiveTerminalN206
) {
    "scripts/policy_callable_gate4_terminal_crossing_n206.py:infer"
} elseif ($Gate4LearnedTailN203) {
    "scripts/policy_callable_gate4_learned_tail_n203.py:infer"
} elseif ($Gate4CoherentTargetHoldN203) {
    "scripts/policy_callable_gate4_coherent_target_hold_n203.py:infer"
} elseif ($Gate4YawSignN203) {
    "scripts/policy_callable_gate4_yaw_sign_n203.py:infer"
} elseif (
    $Gate3CounterHysteresis08N202 -or
    $Gate3CounterHysteresis12N202 -or
    $Gate3CounterHysteresis16N202
) {
    "scripts/policy_callable_gate3_counter_hysteresis_n202.py:infer"
} elseif ($Gate3TightTerminalLevelN201) {
    "scripts/policy_callable_gate3_tight_terminal_level_n201.py:infer"
} elseif ($Gate1HighDownThrustCeilingN200) {
    "scripts/policy_callable_gate1_high_down_thrust_ceiling_n200.py:infer"
} elseif ($Gate3RestoredTerminalLevelN199) {
    "scripts/policy_callable_gate3_restored_terminal_level_n199.py:infer"
} elseif ($Gate3LowConfidenceLatchedRecoveryN198) {
    "scripts/policy_callable_gate3_low_confidence_latched_recovery_n198.py:infer"
} elseif ($Gate3FinalTerminalLevelN197) {
    "scripts/policy_callable_gate3_final_terminal_level_n197.py:infer"
} elseif ($Gate3ZeroConfidenceRecoveryN196) {
    "scripts/policy_callable_gate3_zero_confidence_recovery_n196.py:infer"
} elseif ($Gate3LaterTerminalLevelN195) {
    "scripts/policy_callable_gate3_later_terminal_level_n195.py:infer"
} elseif ($Gate2FrozenObservationDropoutN194) {
    "scripts/policy_callable_gate2_frozen_observation_dropout_n194.py:infer"
} elseif ($Gate3FrozenObservationRecoveryN193) {
    "scripts/policy_callable_gate3_frozen_observation_recovery_n193.py:infer"
} elseif ($Gate2RapidVerticalDeficitN192) {
    "scripts/policy_callable_gate2_rapid_vertical_deficit_n192.py:infer"
} elseif ($Gate3TerminalLevelN191) {
    "scripts/policy_callable_gate3_terminal_level_n191.py:infer"
} elseif ($Gate2SevereVerticalN190) {
    "scripts/policy_callable_gate2_severe_vertical_n190.py:infer"
} elseif ($Gate2VerticalMarginN189) {
    "scripts/policy_callable_gate2_vertical_margin_n189.py:infer"
} elseif ($Gate3CompositeAcquisitionPD) {
    "scripts/policy_callable_gate3_composite_acquisition_pd.py:infer"
} elseif ($Gate3SevereAcquisition) {
    "scripts/policy_callable_gate3_severe_acquisition.py:infer"
} elseif ($Gate3ProjectedPathPD) {
    "scripts/policy_callable_gate3_projected_path_pd.py:infer"
} else {
    "scripts/policy_callable_six_gate_hybrid.py:infer"
}
$PolicyCallableExpectedSHA256 = if (
    $Gate4RollProbeN241
) {
    "205ab6380a92c4770d42793c6b0450f16294d21ce9b8e5efabe6708fd3f8301a"
} elseif (
    $Gate4IdentityBodyMpcN239
) {
    "716352e7b2ec603e0316e59f0d97cc6fe96ca5bdde4f5f7923b0be2fb6b75f4c"
} elseif (
    $Gate4IdentityLockedInterceptN238
) {
    "2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc"
} elseif (
    $Gate4FreshProjectedInterceptN237
) {
    "6fc71223f11246bc90d0c70df41cf67e5bc3e721ae83e6b66c05fcfe1720cb5d"
} elseif (
    $Gate4StagedMpcN236
) {
    "fe7984c17b203ade4b3967ff0e6166be17c4d9a89a827231e2e34be6e28862f4"
} elseif (
    $Gate4SearchMpcN235
) {
    "dd8cd8d89f070c9ca51820c18a9565bb61eb478b1dcd9cce464d4fd4674c569a"
} elseif (
    $Gate4ReacquiringMpcN234
) {
    "84c08dac68be66039567173a7319831232601c22790792588a044401d7671530"
} elseif (
    $Gate4VisualMpcFixedN233
) {
    "4a3f0c7a0c8b73053f6aa7da8995b4fffe1e2c8ed2f430c79135476ca8f1c4ba"
} elseif (
    $Gate4VisualMpcN232
) {
    "0c291b21ca49cd71281744a1cf1da770f42a36b81dec916691d965e91f5228cf"
} elseif (
    $Gate4TerminalCrossingN206 -or
    $Gate4FilteredTerminalCrossingN206 -or
    $Gate4PhasePredictiveTerminalN206
) {
    "b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63"
} elseif ($Gate4LearnedTailN203) {
    "4a976df88dc95af46d2bd022fd8497608ff9cbe3a47c2265c4215e8243e775af"
} elseif ($Gate4CoherentTargetHoldN203) {
    "23100ecad829add77d9de88f4778d8ec9d920d3a2d03ce93c460b47ff484d511"
} elseif ($Gate4YawSignN203) {
    "a06c41623d7dd154d63450ff028f3de274d107e59a8c2d7407d88b39c2628db9"
} elseif (
    $Gate3CounterHysteresis08N202 -or
    $Gate3CounterHysteresis12N202 -or
    $Gate3CounterHysteresis16N202
) {
    "b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe"
} elseif ($Gate3TightTerminalLevelN201) {
    "892f38e34da75f29d7b20a42638ff26a19773518ecab751e4f408842733538dd"
} elseif ($Gate1HighDownThrustCeilingN200) {
    "1c555692497ebb2e0ec35c309158ee55813d5caad299709483439bf636c4c1e9"
} elseif ($Gate3RestoredTerminalLevelN199) {
    "acf17a600e25e2c4c43e928052789c03dfa934e865e7bbd7e7d39ddd9a056885"
} elseif ($Gate3LowConfidenceLatchedRecoveryN198) {
    "9deaecd3957d02b11dabdc7b1fb36ce649f1e5575fd11fc1e190332508742d30"
} elseif ($Gate3FinalTerminalLevelN197) {
    "cfd9308910dbe63a857a67a3b96d53204a2ad929c9ad9b1a93787ba9392e0724"
} elseif ($Gate3ZeroConfidenceRecoveryN196) {
    "ca2ceb12ecbc62a721bf3f34b7cd6b2675592626604899153c3ab7b92754b210"
} elseif ($Gate3LaterTerminalLevelN195) {
    "3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df"
} elseif ($Gate2FrozenObservationDropoutN194) {
    "46a075eafd49a1e44a14e32af458a26e8d411e4cb507b510c6aaaa7ee6b97b4a"
} elseif ($Gate3FrozenObservationRecoveryN193) {
    "a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7"
} elseif ($Gate2RapidVerticalDeficitN192) {
    "9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2"
} elseif ($Gate3TerminalLevelN191) {
    "3750e1620f4a7d244b93c370f31ec135a89572e64a60888d7ea032c244949fae"
} elseif ($Gate2SevereVerticalN190) {
    "f301b682edc10efffc3c131910ea31a479cd323beb888c1d3ddccfd1f13bd73a"
} elseif ($Gate2VerticalMarginN189) {
    "70292925b0748e959eb654febf994afd623143b319bb241ec2e93312ef9e24fb"
} elseif ($Gate3CompositeAcquisitionPD) {
    "ce0bdde0ec4d2cb3b46df0ee35d1ede3f805391899fa24ba19b2df756d42b267"
} elseif ($Gate3SevereAcquisition) {
    "fbbf16d8fc61610f180e914658dac3fd504ef856e4a456b2ba64e9dc394097bd"
} elseif ($Gate3ProjectedPathPD) {
    "9f63f47ee3132c76dfb0a987fb97e9d697a50ce990cdb39a6df3ff69a9bbb883"
} else {
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
}
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Verifier = Join-Path $RepoRoot "scripts\verify_live_policy_shadow.py"
$FullPolicyRunner = Join-Path $RepoRoot "scripts\run_windows_full_policy.ps1"
$RaceStartup = Join-Path $RepoRoot "scripts\start_windows_aigp_race.ps1"
$GateValidation = Join-Path $RepoRoot "scripts\run_official_gate1_validation.py"
$PolicyValidation = Join-Path $RepoRoot "scripts\run_official_policy_validation.py"

Assert-SHA256 $PrefixCheckpoint "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
Assert-SHA256 $Gate4Checkpoint "9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca"
Assert-SHA256 $Gate5Checkpoint "e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30"
Assert-SHA256 $Gate6Checkpoint "4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354"
Assert-SHA256 $PolicyCallable $PolicyCallableExpectedSHA256
if ($Gate4VisualMpcN232 -or $Gate4VisualMpcFixedN233 -or $Gate4ReacquiringMpcN234 -or $Gate4SearchMpcN235 -or $Gate4StagedMpcN236 -or $Gate4FreshProjectedInterceptN237) {
    Assert-SHA256 $Gate4DynamicsModel "9aaefaae111734314353757adbd49da913f13c9bb558627e382ba3c2eb8cc275"
    $env:PUFFER_POLICY_GATE4_DYNAMICS_PATH = Resolve-NativeFilePath $Gate4DynamicsModel
} else {
    Remove-Item Env:PUFFER_POLICY_GATE4_DYNAMICS_PATH -ErrorAction SilentlyContinue
}
if ($Gate4IdentityBodyMpcN239) {
    Assert-SHA256 $Gate4IdentityDynamicsModel "a40e9279bc2d34f69a2296350971ee27eb1393f64e5527b37f2bdc1a545677df"
    $env:PUFFER_POLICY_GATE4_IDENTITY_DYNAMICS_PATH = Resolve-NativeFilePath $Gate4IdentityDynamicsModel
} else {
    Remove-Item Env:PUFFER_POLICY_GATE4_IDENTITY_DYNAMICS_PATH -ErrorAction SilentlyContinue
}
Assert-SHA256 $Runner "e6f2d717138456abba5b9a95905f91144541a8a6bd939a3155fc95881dfdb3b3"
Assert-SHA256 $Verifier "7538094e7065300c42495cd5d380fe7e967daf94065cbf85caa88cc35d253198"
Assert-SHA256 $FullPolicyRunner "892a78ce01a90f8230d1c81271d4f05a71105e4e5b626d7c5febff39cde89d46"
Assert-SHA256 $RaceStartup "d9e97ff6ac38c7a476eebc5ddfa5fd6f47d705c11f613922c17e9690f9031f7d"
Assert-SHA256 $GateValidation "59ed6aec27645a19a166ffa26588a158b17be9c82fcf8b51ae73d1ab1fc2048a"
Assert-SHA256 $PolicyValidation "d36e52e4a19c2b24c8f0075d76ea3b61e559fa10b5af506c7fadcb15625df5db"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found at $Python"
}
if ($ShadowDuration -le 0 -or $FlightDuration -le 0) {
    throw "ShadowDuration and FlightDuration must be positive"
}
if ($CommandHz -lt 50 -or $CommandHz -ge 100) {
    throw "CommandHz must be in [50,100)"
}
if ($PolicyStateHz -lt 0 -or $PolicyStateHz -ge 1000) {
    throw "PolicyStateHz must be in [0,1000)"
}
if ($Gate4VisualMpcN232 -and $PolicyStateHz -ne 0) {
    throw "Gate4VisualMpcN232 requires PolicyStateHz=0 (one state step per live update)"
}
if ($Gate4VisualMpcFixedN233 -and $PolicyStateHz -ne 60) {
    throw "Gate4VisualMpcFixedN233 requires PolicyStateHz=60"
}
if ($Gate4ReacquiringMpcN234 -and $PolicyStateHz -ne 60) {
    throw "Gate4ReacquiringMpcN234 requires PolicyStateHz=60"
}
if ($Gate4SearchMpcN235 -and $PolicyStateHz -ne 60) {
    throw "Gate4SearchMpcN235 requires PolicyStateHz=60"
}
if ($Gate4StagedMpcN236 -and $PolicyStateHz -ne 60) {
    throw "Gate4StagedMpcN236 requires PolicyStateHz=60"
}
if ($Gate4FreshProjectedInterceptN237 -and $PolicyStateHz -ne 60) {
    throw "Gate4FreshProjectedInterceptN237 requires PolicyStateHz=60"
}
if ($Gate4IdentityLockedInterceptN238 -and $PolicyStateHz -ne 60) {
    throw "Gate4IdentityLockedInterceptN238 requires PolicyStateHz=60"
}
if ($Gate4IdentityBodyMpcN239 -and $PolicyStateHz -ne 60) {
    throw "Gate4IdentityBodyMpcN239 requires PolicyStateHz=60"
}
if ($Gate4RollProbeN241 -and $PolicyStateHz -ne 60) {
    throw "Gate4RollProbeN241 requires PolicyStateHz=60"
}

# Historical manual Gate-4 variants use raw camera-relative pose because their
# callable owns target association. The learned filtered-tail variant restores
# the standard observable association/rate contract used by checkpoint training.
if ($Gate4FilteredTerminalCrossingN206 -or $Gate4PhasePredictiveTerminalN206) {
    Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX -ErrorAction SilentlyContinue
} else {
    $env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX = "3"
    $env:PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX = "3"
}

# N229 keeps the live-proven H12/N203 prefix non-predictive and enables
# constant-velocity association propagation only for the learned Gates 4--6
# tail. Historical variants explicitly retain their original no-predictor
# contract so parent evidence cannot depend on inherited shell state.
if ($Gate4PhasePredictiveTerminalN206) {
    $env:PUFFER_POLICY_PREDICT_GATE_DROPOUT = "1"
    $env:PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX = "3"
} else {
    Remove-Item Env:PUFFER_POLICY_PREDICT_GATE_DROPOUT -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX -ErrorAction SilentlyContinue
}
Remove-Item Env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR -ErrorAction SilentlyContinue
Remove-Item Env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX -ErrorAction SilentlyContinue

if ($Mode -eq "Shadow") {
    $simProcess = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
    if ($null -eq $simProcess -or -not $simProcess.Responding) {
        throw "Shadow mode requires an already-running responsive simulator"
    }

    $env:PUFFER_POLICY_CHECKPOINT_PATH = Resolve-NativeFilePath $PrefixCheckpoint
    $env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH = Resolve-NativeFilePath $PrefixCheckpoint
    $env:PUFFER_POLICY_GATE4_CHECKPOINT_PATH = Resolve-NativeFilePath $Gate4Checkpoint
    $env:PUFFER_POLICY_GATE5_CHECKPOINT_PATH = Resolve-NativeFilePath $Gate5Checkpoint
    $env:PUFFER_POLICY_GATE6_CHECKPOINT_PATH = Resolve-NativeFilePath $Gate6Checkpoint
    $env:PUFFER_POLICY_INPUT_DIM = "32"
    $env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "4"
    $env:PUFFER_POLICY_NATIVE_BF16 = "0"

    $ShadowJson = Join-Path $RepoRoot "logs\sitl\${Tag}_shadow.json"
    $ParityJson = Join-Path $RepoRoot "logs\sitl\${Tag}_shadow_parity.json"
    $OutputDirectory = Split-Path -Parent $ShadowJson
    if (-not (Test-Path -LiteralPath $OutputDirectory)) {
        New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    }
    $NativeOutputDirectory = Resolve-NativeFilePath $OutputDirectory
    $NativeShadowJson = Join-Path $NativeOutputDirectory (Split-Path -Leaf $ShadowJson)
    $NativeParityJson = Join-Path $NativeOutputDirectory (Split-Path -Leaf $ParityJson)
    $shadowArgs = @(
        "scripts/drone_sitl_competition_smoke.py"
        "--endpoint", "udpin:0.0.0.0:14550"
        "--camera-host", "0.0.0.0"
        "--camera-port", "5600"
        "--duration", "$ShadowDuration"
        "--command-hz", "$CommandHz"
        "--policy-state-hz", "$PolicyStateHz"
        "--control-mode", "policy-attitude"
        "--policy-callable", $PolicyCallableSpec
        "--policy-shadow-only"
        "--no-arm-on-start"
        "--policy-shadow-calibration-timeout-s", "10"
        "--policy-trace-hz", "1000"
        "--policy-trace-max-samples", "4000"
        "--policy-gate-phase-onehot-adapter-observation"
        "--policy-hybrid-prefix-confidence-observation"
        "--policy-race-phase-denominator", "6"
        "--acceptance-config", "config/sitl_multigate_acceptance.json"
        "--require-telemetry"
        "--require-camera"
        "--json-path", $NativeShadowJson
    )
    & $Python @shadowArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Six-gate hybrid passive shadow failed with exit code $LASTEXITCODE"
    }

    $verifyArgs = @(
        "scripts/verify_live_policy_shadow.py"
        (Resolve-NativeFilePath $ShadowJson)
        (Resolve-NativeFilePath $PrefixCheckpoint)
        "--gate4-checkpoint", (Resolve-NativeFilePath $Gate4Checkpoint)
        "--gate5-checkpoint", (Resolve-NativeFilePath $Gate5Checkpoint)
        "--gate6-checkpoint", (Resolve-NativeFilePath $Gate6Checkpoint)
        "--policy-callable", (Resolve-NativeFilePath $PolicyCallable)
        "--hybrid-prefix-confidence"
        "--min-inference-hz", "10.0"
        "--json-path", $NativeParityJson
    )
    & $Python @verifyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Six-gate hybrid passive shadow parity failed with exit code $LASTEXITCODE"
    }
    Write-Host "Passive six-gate hybrid shadow and parity passed: $ParityJson"
    exit 0
}

$TargetGateCount = if ($Mode -eq "BoundedGate3") {
    3
} elseif ($Mode -eq "BoundedGate4") {
    4
} else {
    6
}
$StopAfterOfficialGateIndex = if ($Mode -eq "BoundedGate3") {
    3
} elseif ($Mode -eq "BoundedGate4") {
    4
} else {
    -1
}
$runArgs = @{
    PolicyCheckpoint = $Gate4Checkpoint
    Gate5Checkpoint = $Gate5Checkpoint
    Gate6Checkpoint = $Gate6Checkpoint
    SixGateComposite = $true
    HybridPrefixCheckpoint = $PrefixCheckpoint
    HybridPolicyCallable = $PolicyCallableSpec
    Python = $Python
    SimRoot = $SimRoot
    Tag = $Tag
    SmokeDuration = $FlightDuration
    CommandHz = $CommandHz
    PolicyStateHz = $PolicyStateHz
    TargetGateCount = $TargetGateCount
    StopAfterOfficialGateIndex = $StopAfterOfficialGateIndex
    Repeats = $Repeats
    MinValidRuns = $MinValidRuns
    PolicyLayoutPrecisionBytes = 4
}
if ($ForceRelaunch) {
    $runArgs.ForceRelaunch = $true
}
& "$RepoRoot\scripts\run_windows_full_policy.ps1" @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "Six-gate hybrid $Mode validation failed with exit code $LASTEXITCODE"
}
