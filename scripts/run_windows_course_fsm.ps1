param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
    [string]$Tag = "course_fsm_gate2",
    [int]$SmokeDuration = 90,
    [int]$TargetGateCount = 2,
    [string]$PolicyCheckpoint = "",
    [int]$PolicyActivateGateIndex = 1,
    [switch]$ForceRelaunch,
    [switch]$RunTrackBTrain
)

# One-command official course-FSM attempt.
#
# Startup policy:
# - Reuse a running simulator and active race.
# - Launch the simulator + automate login/event/RACE only when it is absent or
#   not yet in a race.
# - Never kill the simulator unless -ForceRelaunch is explicitly supplied.
#
# Attempt-reset policy:
# - prd_loop_runner -> run_official_gate1_validation sends MAVLink command 31000.
# - Supported simulator builds reset and auto-start the race in place, avoiding
#   process/UI startup on normal retries.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    throw "Python runtime not found at $Python"
}

$startArgs = @{
    Python = $Python
    SimRoot = $SimRoot
    Tag = "${Tag}_startup"
}
if ($ForceRelaunch) {
    $startArgs.ForceRelaunch = $true
}

$simProcess = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
if ($ForceRelaunch -or $null -eq $simProcess -or -not $simProcess.Responding) {
    Write-Host "Simulator startup is required; ensuring process and initial race are available..."
    & "$RepoRoot\scripts\start_windows_aigp_race.ps1" @startArgs
} else {
    # Normal retry: the validator below waits for a heartbeat, addresses the
    # simulator's learned component (including component 0), and sends 31000.
    # Avoid a redundant race-status/UI probe on every attempt.
    Write-Host "Reusing healthy simulator process $($simProcess.Id); skipping startup/UI checks"
}

Write-Host "Running course FSM; this attempt uses MAVLink 31000 in-place reset..."
$loopArgs = @(
    "scripts/prd_loop_runner.py",
    "--tag", $Tag,
    "--smoke-duration", "$SmokeDuration",
    "--target-gate-count", "$TargetGateCount"
)
if ($RunTrackBTrain) {
    $loopArgs += "--run-track-b-train"
}
if ($PolicyCheckpoint) {
    if (-not (Test-Path $PolicyCheckpoint)) {
        throw "Policy checkpoint not found: $PolicyCheckpoint"
    }
    $env:PUFFER_POLICY_CHECKPOINT_PATH = (Resolve-Path $PolicyCheckpoint).Path
    $loopArgs += @(
        "--policy-callable", "scripts/policy_callable_checkpoint.py:infer",
        "--policy-activate-gate-index", "$PolicyActivateGateIndex"
    )
    Write-Host "Hybrid mode: FSM through gate $PolicyActivateGateIndex, then checkpoint policy"
}
& $Python @loopArgs
if ($LASTEXITCODE -ne 0) {
    throw "Course-FSM runner failed with exit code $LASTEXITCODE"
}
