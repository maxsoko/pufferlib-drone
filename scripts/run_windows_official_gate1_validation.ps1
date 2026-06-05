param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$Tag = "windows_local_reset",
    [double]$ProbeDuration = 8.0,
    [double]$SmokeDuration = 30.0,
    [int]$CameraMaxPacketsPerLoop = 512,
    [string]$ControlMode = "visual-servo",
    [string]$PolicyCallable = "",
    [string]$PolicyActionJson = "[0.0, 0.0, 0.0, 0.0]",
    [switch]$SendSimReset,
    [switch]$RequireOfficialRaceProgress,
    [double]$PostResetSleepS = 2.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$argsList = @(
    "scripts/run_official_gate1_validation.py",
    "--host", "0.0.0.0",
    "--mavlink-port", "14550",
    "--camera-port", "5600",
    "--endpoint", "udpin:0.0.0.0:14550",
    "--camera-host", "0.0.0.0",
    "--probe-duration", "$ProbeDuration",
    "--smoke-duration", "$SmokeDuration",
    "--camera-max-packets-per-loop", "$CameraMaxPacketsPerLoop",
    "--post-reset-sleep-s", "$PostResetSleepS",
    "--control-mode", "$ControlMode",
    "--policy-action-json", "$PolicyActionJson",
    "--summary-json-path", "logs/sitl/official_gate1_validation_summary_$Tag.json",
    "--probe-json-path", "logs/sitl/stream_probe_official_$Tag.json",
    "--smoke-json-path", "logs/sitl/competition_smoke_gate1_official_$Tag.json",
    "--smoke-csv-path", "logs/sitl/competition_smoke_gate1_official_$Tag.csv"
)

if ($PolicyCallable -ne "") {
    $argsList += @("--policy-callable", $PolicyCallable)
}
if ($SendSimReset) {
    $argsList += @("--send-sim-reset")
}
if ($RequireOfficialRaceProgress) {
    $argsList += @("--require-official-race-progress")
}

& $Python @argsList
