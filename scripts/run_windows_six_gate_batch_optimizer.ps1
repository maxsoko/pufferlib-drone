param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
    [string]$Tag = "competitive_six_gate_batch_001",
    [ValidateSet("N195", "N197", "N198", "N202", "H08", "H12", "H16", "YP", "CH", "LT", "TP", "FT", "PP", "PF", "RF", "MO", "MC", "MR", "MS", "MB", "MI", "IL", "LM")]
    [string[]]$Variants = @("N195", "N197", "N198"),
    [int]$RunsPerVariant = 2,
    [int]$FlightDuration = 45,
    [int]$CommandHz = 80,
    [switch]$Execute,
    [switch]$ContinueAfterFinish
)

# Risk-sensitive official-simulator policy tournament. Each child invocation
# owns one exact reset/arm/flight/disarm lifecycle. A fresh child invocation on
# the next attempt performs same-process UI recovery when a collision left the
# event inactive. Every attempt gets a command-free, no-reset post-stop proof.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Assert-SHA256([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required batch artifact not found: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected, got $actual"
    }
}

if ($RunsPerVariant -le 0) {
    throw "RunsPerVariant must be positive"
}
if ($FlightDuration -le 0) {
    throw "FlightDuration must be positive"
}
if ($CommandHz -lt 50 -or $CommandHz -ge 100) {
    throw "CommandHz must be in [50,100)"
}
if (-not $Variants -or $Variants.Count -eq 0) {
    throw "At least one policy variant is required"
}

$HybridRunner = Join-Path $RepoRoot "scripts\run_windows_six_gate_hybrid.ps1"
$PoststopProbe = Join-Path $RepoRoot "scripts\debug_official_reset_snapshot.py"
$Scorer = Join-Path $RepoRoot "scripts\score_official_six_gate_batch.py"
Assert-SHA256 $HybridRunner "6ca53a4e3945402b770175180a14818755ebce137e28ccc613b4ecaba8d5f85c"
Assert-SHA256 $PoststopProbe "9b223f8fa3f439016a044ecf50a440fd54bd1656e92c08a73f01e05e265e9048"
Assert-SHA256 $Scorer "538161ed73bf27fff14c2cd8eb6b59dbfc75981477c10e060073112d3724a48c"

$VariantSwitch = @{
    N195 = "Gate2FrozenObservationDropoutN194"
    N197 = "Gate3ZeroConfidenceRecoveryN196"
    N198 = "Gate3FinalTerminalLevelN197"
    N202 = "Gate3TightTerminalLevelN201"
    H08 = "Gate3CounterHysteresis08N202"
    H12 = "Gate3CounterHysteresis12N202"
    H16 = "Gate3CounterHysteresis16N202"
    YP = "Gate4YawSignN203"
    CH = "Gate4CoherentTargetHoldN203"
    LT = "Gate4LearnedTailN203"
    TP = "Gate4TerminalCrossingN206"
    FT = "Gate4FilteredTerminalCrossingN206"
    PP = "Gate4PhasePredictiveTerminalN206"
    PF = "Gate4PhasePredictiveTerminalN206"
    RF = "Gate4TerminalCrossingN206"
    MO = "Gate4VisualMpcN232"
    MC = "Gate4VisualMpcFixedN233"
    MR = "Gate4ReacquiringMpcN234"
    MS = "Gate4SearchMpcN235"
    MB = "Gate4StagedMpcN236"
    MI = "Gate4FreshProjectedInterceptN237"
    IL = "Gate4IdentityLockedInterceptN238"
    LM = "Gate4IdentityBodyMpcN239"
}
$OutputDir = Join-Path $RepoRoot "logs\sitl"
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$ManifestPath = Join-Path $OutputDir "${Tag}_batch_manifest.json"
$ScorePath = Join-Path $OutputDir "${Tag}_batch_score.json"

$planned = @()
for ($run = 1; $run -le $RunsPerVariant; $run++) {
    foreach ($variant in $Variants) {
        $attemptTag = "{0}_{1}_run_{2:D3}" -f $Tag, $variant.ToLowerInvariant(), $run
        $planned += [ordered]@{
            variant = $variant
            run = $run
            tag = $attemptTag
        }
    }
}

if (Test-Path -LiteralPath $ManifestPath) {
    $manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    if ([int]$manifest.target_gate_count -ne 6) {
        throw "Existing batch manifest is not a six-gate manifest: $ManifestPath"
    }
    $records = @($manifest.attempts)
} else {
    $records = @()
}

function Write-Manifest {
    $payload = [ordered]@{
        schema_version = 1
        tag = $Tag
        target_gate_count = 6
        variants = @($Variants)
        runs_per_variant = $RunsPerVariant
        flight_duration_s = $FlightDuration
        command_hz = $CommandHz
        policy_state_hz_by_variant = [ordered]@{ PF = 60; RF = 60; MO = 0; MC = 60; MR = 60; MS = 60; MB = 60; MI = 60; IL = 60; LM = 60 }
        objective_order = @(
            "valid_finish_count",
            "minimum_official_gate_count",
            "median_official_gate_count",
            "collision_free_count",
            "collision_count",
            "lap_or_attempt_time"
        )
        planned = $planned
        attempts = @($records)
    }
    $payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 -LiteralPath $ManifestPath
}

Write-Manifest
if (-not $Execute) {
    Write-Host "Dry run only. Planned official six-gate batch:"
    foreach ($item in $planned) {
        Write-Host ("  {0} run {1}: {2}" -f $item.variant, $item.run, $item.tag)
    }
    Write-Host "Re-run with -Execute to mutate simulator state. Manifest: $ManifestPath"
    exit 0
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Windows Python runtime not found: $Python"
}

foreach ($plan in $planned) {
    if (@($records | Where-Object { $_.tag -eq $plan.tag }).Count -gt 0) {
        Write-Host "Skipping already recorded attempt $($plan.tag)"
        continue
    }

    Write-Host "==> six-gate batch attempt $($plan.tag)"
    $runArgs = @{
        Mode = "FullLap"
        Python = $Python
        SimRoot = $SimRoot
        Tag = $plan.tag
        FlightDuration = $FlightDuration
        CommandHz = $CommandHz
        Repeats = 1
        MinValidRuns = 1
    }
    $runArgs[$VariantSwitch[$plan.variant]] = $true
    if ($plan.variant -in @("PF", "RF", "MC", "MR", "MS", "MB", "MI", "IL", "LM")) {
        $runArgs.PolicyStateHz = 60
    }
    $childExit = 0
    $childError = ""
    try {
        & $HybridRunner @runArgs
        $childExit = [int]$LASTEXITCODE
    } catch {
        $childExit = 1
        $childError = $_.Exception.Message
        Write-Warning "Attempt $($plan.tag) rejected: $childError"
    }

    $poststopRelative = "logs/sitl/{0}_poststop" -f $plan.tag
    $probeArgs = @(
        "scripts/debug_official_reset_snapshot.py",
        "--duration-s", "5",
        "--output-dir", $poststopRelative
    )
    & $Python @probeArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Command-free post-stop probe failed for $($plan.tag)"
    }

    $summaryRelative = "logs/sitl/official_policy_validation_summary_{0}.json" -f $plan.tag
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $summaryRelative))) {
        throw "Missing official validation summary for $($plan.tag)"
    }
    $records += [ordered]@{
        variant = $plan.variant
        run = $plan.run
        tag = $plan.tag
        child_exit_code = $childExit
        child_error = $childError
        summary_path = $summaryRelative
        poststop_path = "${poststopRelative}/snapshot_summary.json"
    }
    Write-Manifest

    & $Python $Scorer $ManifestPath --output $ScorePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Six-gate batch scoring failed"
    }
    $score = Get-Content -Raw -LiteralPath $ScorePath | ConvertFrom-Json
    $leader = $score.variants | Select-Object -First 1
    Write-Host (
        "Batch leader {0}: finishes={1}, min/median gates={2}/{3}, collision-free={4}/{5}" -f
        $leader.variant,
        $leader.valid_finishes,
        $leader.minimum_official_gate_count,
        $leader.median_official_gate_count,
        $leader.collision_free_attempts,
        $leader.attempt_count
    )
    if (-not $ContinueAfterFinish -and [int]$leader.valid_finishes -gt 0) {
        Write-Host "Valid six-gate finish found; stopping batch immediately."
        break
    }
}

& $Python $Scorer $ManifestPath --output $ScorePath
if ($LASTEXITCODE -ne 0) {
    throw "Final six-gate batch scoring failed"
}
Write-Host "Official six-gate batch complete: $ScorePath"
