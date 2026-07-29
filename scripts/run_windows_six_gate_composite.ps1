param(
    [ValidateSet("Shadow", "BoundedGate4", "FullLap")]
    [string]$Mode = "Shadow",
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
    [string]$Tag = "n145_six_gate_composite",
    [int]$ShadowDuration = 10,
    [int]$FlightDuration = 45,
    [int]$CommandHz = 80,
    [int]$Repeats = 1,
    [int]$MinValidRuns = 1,
    [switch]$ForceRelaunch
)

# Frozen N145 deployment entry point. Shadow is deliberately the default and
# sends no reset, arm, setpoint, or disarm command. Flight modes are explicit.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

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
$PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_six_gate_composite.py"

Assert-SHA256 $Gate4Checkpoint "9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca"
Assert-SHA256 $Gate5Checkpoint "e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30"
Assert-SHA256 $Gate6Checkpoint "4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354"
Assert-SHA256 $PolicyCallable "9eaab39694f67fdae6ebb6be52af9a282974e6778822fd84cd194ac5ade5d2a3"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found at $Python"
}
if ($ShadowDuration -le 0 -or $FlightDuration -le 0) {
    throw "ShadowDuration and FlightDuration must be positive"
}
if ($CommandHz -lt 50 -or $CommandHz -ge 100) {
    throw "CommandHz must be in [50,100)"
}

if ($Mode -eq "Shadow") {
    $simProcess = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
    if ($null -eq $simProcess -or -not $simProcess.Responding) {
        throw "Shadow mode requires an already-running responsive simulator"
    }

    $env:PUFFER_POLICY_CHECKPOINT_PATH = Resolve-NativeFilePath $Gate4Checkpoint
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
        "--control-mode", "policy-attitude"
        "--policy-callable", "scripts/policy_callable_six_gate_composite.py:infer"
        "--policy-shadow-only"
        "--no-arm-on-start"
        "--policy-shadow-calibration-timeout-s", "10"
        "--policy-trace-hz", "1000"
        "--policy-trace-max-samples", "4000"
        "--policy-gate-phase-onehot-adapter-observation"
        "--policy-race-phase-denominator", "6"
        "--acceptance-config", "config/sitl_multigate_acceptance.json"
        "--require-telemetry"
        "--require-camera"
        "--json-path", $NativeShadowJson
    )
    & $Python @shadowArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Six-gate passive shadow failed with exit code $LASTEXITCODE"
    }

    $verifyArgs = @(
        "scripts/verify_live_policy_shadow.py"
        (Resolve-NativeFilePath $ShadowJson)
        (Resolve-NativeFilePath $Gate4Checkpoint)
        "--gate5-checkpoint", (Resolve-NativeFilePath $Gate5Checkpoint)
        "--gate6-checkpoint", (Resolve-NativeFilePath $Gate6Checkpoint)
        "--policy-callable", (Resolve-NativeFilePath $PolicyCallable)
        "--json-path", $NativeParityJson
    )
    & $Python @verifyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Six-gate passive shadow parity failed with exit code $LASTEXITCODE"
    }
    Write-Host "Passive six-gate shadow and parity passed: $ParityJson"
    exit 0
}

$TargetGateCount = if ($Mode -eq "BoundedGate4") { 4 } else { 6 }
$StopAfterOfficialGateIndex = if ($Mode -eq "BoundedGate4") { 4 } else { -1 }
$runArgs = @{
    PolicyCheckpoint = $Gate4Checkpoint
    Gate5Checkpoint = $Gate5Checkpoint
    Gate6Checkpoint = $Gate6Checkpoint
    SixGateComposite = $true
    Python = $Python
    SimRoot = $SimRoot
    Tag = $Tag
    SmokeDuration = $FlightDuration
    CommandHz = $CommandHz
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
    throw "Six-gate $Mode validation failed with exit code $LASTEXITCODE"
}
