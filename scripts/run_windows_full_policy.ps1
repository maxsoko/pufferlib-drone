param(
    [Parameter(Mandatory = $true)]
    [string]$PolicyCheckpoint,
    [string]$Gate5Checkpoint = "",
    [string]$Gate6Checkpoint = "",
    [switch]$SixGateComposite,
    [string]$HybridPrefixCheckpoint = "",
    [string]$HybridPolicyCallable = "",
    [string]$PreviousCheckpoint = "",
    [string]$ResidualCheckpoint = "",
    [string]$FinalCheckpoint = "",
    [string]$GateHeadMinConfidenceJson = "",
    [string]$ResidualLowConfidenceCheckpoint = "",
    [string]$FinalLowConfidenceCheckpoint = "",
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
    [string]$Tag = "full_policy_gate1",
    [int]$SmokeDuration = 30,
    [int]$CommandHz = 60,
    [int]$PolicyStateHz = 0,
    [int]$TargetGateCount = 1,
    [int]$StopAfterOfficialGateIndex = -1,
    [int]$Repeats = 1,
    [int]$MinValidRuns = 0,
    [ValidateSet(2, 4)]
    [int]$PolicyLayoutPrecisionBytes = 2,
    [switch]$PolicyPhaseAdapterObservation,
    [switch]$PolicyGateProgressAdapterObservation,
    [switch]$PolicyGatePhaseOnehotAdapterObservation,
    [int]$PolicyGateProgressDenominator = 6,
    [int]$PolicyActionBiasGateIndex = -1,
    [double]$PolicyActionBiasPitch = 0.0,
    [double]$PolicyActionBiasRoll = 0.0,
    [double]$PolicyActionBiasThrust = 0.0,
    [double]$PolicyActionBiasYaw = 0.0,
    [string]$PolicyActionBiasTableJson = "",
    [switch]$ForceRelaunch
)

# Full-policy official validation. Deterministic code manages lifecycle and
# safety only; the recurrent policy owns every timed-flight attitude command.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Resolve-NativeFilePath([string]$Path) {
    # Resolve-Path.Path includes the PowerShell provider prefix when this script
    # is launched from a WSL UNC working tree, for example:
    #   Microsoft.PowerShell.Core\FileSystem::\\wsl.localhost\...
    # Native Windows programs (including NumPy's fromfile) cannot open that
    # provider-qualified spelling. ProviderPath is the ordinary UNC path.
    $resolved = Resolve-Path -LiteralPath $Path
    if ($resolved.ProviderPath) {
        return $resolved.ProviderPath
    }
    return $resolved.Path
}

if (-not (Test-Path $Python)) {
    throw "Python runtime not found at $Python"
}
if (-not (Test-Path $PolicyCheckpoint)) {
    throw "Policy checkpoint not found: $PolicyCheckpoint"
}
if ($SixGateComposite -and -not $Gate5Checkpoint) {
    throw "SixGateComposite requires Gate5Checkpoint"
}
if ($SixGateComposite -and -not $Gate6Checkpoint) {
    throw "SixGateComposite requires Gate6Checkpoint"
}
if ($HybridPrefixCheckpoint -and -not $SixGateComposite) {
    throw "HybridPrefixCheckpoint requires SixGateComposite"
}
if ($HybridPolicyCallable -and -not $HybridPrefixCheckpoint) {
    throw "HybridPolicyCallable requires HybridPrefixCheckpoint"
}
if ($HybridPrefixCheckpoint -and -not (Test-Path $HybridPrefixCheckpoint)) {
    throw "Hybrid prefix checkpoint not found: $HybridPrefixCheckpoint"
}
if ($Gate5Checkpoint -and -not (Test-Path $Gate5Checkpoint)) {
    throw "Gate 5 checkpoint not found: $Gate5Checkpoint"
}
if ($Gate6Checkpoint -and -not (Test-Path $Gate6Checkpoint)) {
    throw "Gate 6 checkpoint not found: $Gate6Checkpoint"
}
if (-not $SixGateComposite -and ($Gate5Checkpoint -or $Gate6Checkpoint)) {
    throw "Gate5Checkpoint and Gate6Checkpoint require SixGateComposite"
}
if ($PreviousCheckpoint -and -not (Test-Path $PreviousCheckpoint)) {
    throw "Previous checkpoint not found: $PreviousCheckpoint"
}
if ($ResidualCheckpoint -and -not (Test-Path $ResidualCheckpoint)) {
    throw "Residual checkpoint not found: $ResidualCheckpoint"
}
if ($FinalCheckpoint -and -not $ResidualCheckpoint) {
    throw "FinalCheckpoint requires ResidualCheckpoint"
}
if ($FinalCheckpoint -and -not (Test-Path $FinalCheckpoint)) {
    throw "Final checkpoint not found: $FinalCheckpoint"
}
if ($ResidualLowConfidenceCheckpoint -and -not (Test-Path $ResidualLowConfidenceCheckpoint)) {
    throw "Low-confidence residual checkpoint not found: $ResidualLowConfidenceCheckpoint"
}
if ($FinalLowConfidenceCheckpoint -and -not (Test-Path $FinalLowConfidenceCheckpoint)) {
    throw "Low-confidence final checkpoint not found: $FinalLowConfidenceCheckpoint"
}
if ($TargetGateCount -lt 1 -or $TargetGateCount -gt 6) {
    throw "TargetGateCount must be between 1 and 6"
}
if ($CommandHz -lt 1 -or $CommandHz -ge 100) {
    throw "CommandHz must be between 1 and 99"
}
if ($PolicyStateHz -lt 0 -or $PolicyStateHz -ge 1000) {
    throw "PolicyStateHz must be between 0 and 999"
}
if ($StopAfterOfficialGateIndex -lt -1 -or $StopAfterOfficialGateIndex -gt 6) {
    throw "StopAfterOfficialGateIndex must be -1 or between 0 and 6"
}
if ($StopAfterOfficialGateIndex -ge 0 -and $StopAfterOfficialGateIndex -lt $TargetGateCount) {
    throw "StopAfterOfficialGateIndex cannot be below TargetGateCount"
}
if ($SixGateComposite -and ($PreviousCheckpoint -or $ResidualCheckpoint -or $FinalCheckpoint)) {
    throw "SixGateComposite cannot be combined with previous/residual/final policy modes"
}
if ($SixGateComposite -and (
    $GateHeadMinConfidenceJson -or
    $ResidualLowConfidenceCheckpoint -or
    $FinalLowConfidenceCheckpoint
)) {
    throw "SixGateComposite cannot be combined with residual confidence routing"
}
if ($SixGateComposite -and ($PolicyPhaseAdapterObservation -or $PolicyGateProgressAdapterObservation)) {
    throw "SixGateComposite owns the six-gate one-hot observation adapter"
}
if ($SixGateComposite -and $PolicyLayoutPrecisionBytes -ne 4) {
    throw "SixGateComposite requires FP32 checkpoint layout precision (4 bytes)"
}
if ($SixGateComposite -and (
    $PolicyActionBiasGateIndex -ge 0 -or $PolicyActionBiasTableJson
)) {
    throw "SixGateComposite owns all phase action adapters; external action bias is forbidden"
}
if ($PolicyPhaseAdapterObservation -and ($PreviousCheckpoint -or $ResidualCheckpoint)) {
    throw "PolicyPhaseAdapterObservation requires the single-checkpoint policy callable"
}
if ($PolicyGateProgressAdapterObservation -and ($PreviousCheckpoint -or $ResidualCheckpoint)) {
    throw "PolicyGateProgressAdapterObservation requires the single-checkpoint policy callable"
}
if ($PolicyGatePhaseOnehotAdapterObservation -and ($PreviousCheckpoint -or $ResidualCheckpoint)) {
    throw "PolicyGatePhaseOnehotAdapterObservation requires the single-checkpoint policy callable"
}
$PolicyObservationAdapterCount = @(
    $PolicyPhaseAdapterObservation,
    $PolicyGateProgressAdapterObservation,
    ($PolicyGatePhaseOnehotAdapterObservation -or $SixGateComposite)
).Where({ $_ }).Count
if ($PolicyObservationAdapterCount -gt 1) {
    throw "Choose only one policy observation adapter"
}
if ($PolicyGateProgressDenominator -le 0) {
    throw "PolicyGateProgressDenominator must be positive"
}
if ($PolicyActionBiasGateIndex -lt -1) {
    throw "PolicyActionBiasGateIndex must be -1 or nonnegative"
}
if (($PolicyActionBiasGateIndex -ge 0 -or $PolicyActionBiasTableJson) -and -not $PolicyGateProgressAdapterObservation) {
    throw "Policy action bias requires PolicyGateProgressAdapterObservation"
}

$simProcess = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
if ($ForceRelaunch -or $null -eq $simProcess -or -not $simProcess.Responding) {
    $startArgs = @{
        Python = $Python
        SimRoot = $SimRoot
        Tag = "${Tag}_startup"
    }
    if ($ForceRelaunch) {
        $startArgs.ForceRelaunch = $true
    }
    & "$RepoRoot\scripts\start_windows_aigp_race.ps1" @startArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Simulator startup failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "Reusing healthy simulator process $($simProcess.Id)"
    # A healthy process can still be parked at the inactive pre-race
    # "THROTTLE DOWN" screen, where command 31000 is ignored.  Let the startup
    # helper confirm the active race or recover through the UI without ever
    # relaunching the process.
    & "$RepoRoot\scripts\start_windows_aigp_race.ps1" `
        -Python $Python `
        -SimRoot $SimRoot `
        -Tag "${Tag}_reuse_check" `
        -NoRelaunch `
        -SkipLoginClick
    if ($LASTEXITCODE -ne 0) {
        throw "Healthy simulator race-state recovery failed with exit code $LASTEXITCODE"
    }
}

$PrimaryCheckpoint = if ($HybridPrefixCheckpoint) {
    $HybridPrefixCheckpoint
} else {
    $PolicyCheckpoint
}
$env:PUFFER_POLICY_CHECKPOINT_PATH = Resolve-NativeFilePath $PrimaryCheckpoint
$env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "$PolicyLayoutPrecisionBytes"
$env:PUFFER_POLICY_NATIVE_BF16 = "0"
if ($PolicyPhaseAdapterObservation -or $PolicyGateProgressAdapterObservation -or $PolicyGatePhaseOnehotAdapterObservation -or $SixGateComposite) {
    $env:PUFFER_POLICY_INPUT_DIM = "32"
} else {
    Remove-Item Env:PUFFER_POLICY_INPUT_DIM -ErrorAction SilentlyContinue
}
if ($PolicyActionBiasGateIndex -ge 0) {
    $env:PUFFER_POLICY_ACTION_BIAS_GATE_INDEX = "$PolicyActionBiasGateIndex"
    $env:PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX = "23"
    $env:PUFFER_POLICY_ACTION_BIAS_DENOMINATOR = "$PolicyGateProgressDenominator"
    $env:PUFFER_POLICY_ACTION_BIAS_PITCH = "$PolicyActionBiasPitch"
    $env:PUFFER_POLICY_ACTION_BIAS_ROLL = "$PolicyActionBiasRoll"
    $env:PUFFER_POLICY_ACTION_BIAS_THRUST = "$PolicyActionBiasThrust"
    $env:PUFFER_POLICY_ACTION_BIAS_YAW = "$PolicyActionBiasYaw"
} else {
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_GATE_INDEX -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_DENOMINATOR -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_PITCH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_ROLL -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_THRUST -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_YAW -ErrorAction SilentlyContinue
}
if ($PolicyActionBiasTableJson) {
    $env:PUFFER_POLICY_ACTION_BIAS_TABLE_JSON = $PolicyActionBiasTableJson
} else {
    Remove-Item Env:PUFFER_POLICY_ACTION_BIAS_TABLE_JSON -ErrorAction SilentlyContinue
}
$policyCallable = "scripts/policy_callable_checkpoint.py:infer"
if ($SixGateComposite) {
    if ($HybridPrefixCheckpoint) {
        $env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH =
            Resolve-NativeFilePath $HybridPrefixCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    $env:PUFFER_POLICY_GATE4_CHECKPOINT_PATH =
        Resolve-NativeFilePath $PolicyCheckpoint
    $env:PUFFER_POLICY_GATE5_CHECKPOINT_PATH =
        Resolve-NativeFilePath $Gate5Checkpoint
    $env:PUFFER_POLICY_GATE6_CHECKPOINT_PATH =
        Resolve-NativeFilePath $Gate6Checkpoint
    Remove-Item Env:PUFFER_POLICY_BASE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_FINAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RACE_PHASE_DENOMINATOR -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    $policyCallable = "scripts/policy_callable_six_gate_composite.py:infer"
    if ($HybridPrefixCheckpoint) {
        $policyCallable = if ($HybridPolicyCallable) {
            $HybridPolicyCallable
        } else {
            "scripts/policy_callable_six_gate_hybrid.py:infer"
        }
    }
} elseif ($PreviousCheckpoint -or $ResidualCheckpoint) {
    $env:PUFFER_POLICY_BASE_CHECKPOINT_PATH = Resolve-NativeFilePath $PolicyCheckpoint
    if ($PreviousCheckpoint) {
        $env:PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH =
            Resolve-NativeFilePath $PreviousCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    if ($ResidualCheckpoint) {
        $env:PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH =
            Resolve-NativeFilePath $ResidualCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    if ($FinalCheckpoint) {
        $env:PUFFER_POLICY_FINAL_CHECKPOINT_PATH = Resolve-NativeFilePath $FinalCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_FINAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    $env:PUFFER_POLICY_RACE_PHASE_DENOMINATOR = "3"
    if ($GateHeadMinConfidenceJson) {
        $env:PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON = $GateHeadMinConfidenceJson
    } else {
        Remove-Item Env:PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON -ErrorAction SilentlyContinue
    }
    if ($ResidualLowConfidenceCheckpoint) {
        $env:PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH =
            Resolve-NativeFilePath $ResidualLowConfidenceCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    if ($FinalLowConfidenceCheckpoint) {
        $env:PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH =
            Resolve-NativeFilePath $FinalLowConfidenceCheckpoint
    } else {
        Remove-Item Env:PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    }
    $policyCallable = "scripts/policy_callable_phase_residual.py:infer"
} else {
    Remove-Item Env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE4_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE5_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE6_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_BASE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_FINAL_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RACE_PHASE_DENOMINATOR -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH -ErrorAction SilentlyContinue
}
$validationArgs = @(
    "scripts/run_official_policy_validation.py"
    "--n-repeats", "$Repeats"
    "--tag", $Tag
    "--control-mode", "policy-attitude"
    "--policy-callable", $policyCallable
    "--send-sim-reset"
    "--policy-ready-reset"
    "--post-reset-sleep-s", "5"
    "--probe-duration", "5"
    "--smoke-duration", "$SmokeDuration"
    "--command-hz", "$CommandHz"
    "--policy-state-hz", "$PolicyStateHz"
    "--target-gate-count", "$TargetGateCount"
    "--min-official-gate-index", "$TargetGateCount"
    "--acceptance-config", "config/sitl_multigate_acceptance.json"
    "--require-official-race-progress"
)
if ($PreviousCheckpoint -or $ResidualCheckpoint) {
    $validationArgs += @(
        "--policy-race-phase-observation"
        "--policy-race-phase-denominator", "3"
    )
}
if ($PolicyPhaseAdapterObservation) {
    $validationArgs += @("--policy-phase-adapter-observation")
}
if ($PolicyGateProgressAdapterObservation) {
    $validationArgs += @(
        "--policy-gate-progress-adapter-observation"
        "--policy-race-phase-denominator", "$PolicyGateProgressDenominator"
    )
}
if ($PolicyGatePhaseOnehotAdapterObservation -or $SixGateComposite) {
    $validationArgs += @(
        "--policy-gate-phase-onehot-adapter-observation"
        "--policy-race-phase-denominator", "$PolicyGateProgressDenominator"
    )
}
if ($HybridPrefixCheckpoint) {
    $validationArgs += @("--policy-hybrid-prefix-confidence-observation")
}
if ($StopAfterOfficialGateIndex -ge 0) {
    $validationArgs += @(
        "--stop-after-official-gate-index", "$StopAfterOfficialGateIndex"
    )
}
if ($MinValidRuns -gt 0) {
    $validationArgs += @("--min-valid-runs", "$MinValidRuns")
}
Write-Host "Running full policy from race start; retries use MAVLink 31000"
& $Python @validationArgs
if ($LASTEXITCODE -ne 0) {
    throw "Full-policy validation failed with exit code $LASTEXITCODE"
}
