param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$Tag = "windows_gate1_auto",
    [string]$SimRoot = "",
    [string]$SmokeTag = "",
    [int]$SmokeDuration = 45,
    [double]$ProbeDuration = 8.0,
    [string]$ControlMode = "visual-servo-attitude",
    [string]$CommandFrame = "local_ned",
    [string]$CommandYawMode = "yaw_and_rate",
    [string]$PolicyCallable = "",
    [string]$PolicyActionJson = "[0.0, 0.0, 0.0, 0.0]",
    [switch]$RequireOfficialRaceProgress,
    [int]$RaceCheckS = 10,
    [int]$RaceAttempts = 4,
    [switch]$NoRelaunch,
    [switch]$ForceRelaunch,
    [switch]$NoUiClicks
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if ($SmokeTag -eq "") {
    $SmokeTag = "$Tag"
}

if (-not (Test-Path $Python)) {
    throw "Python runtime not found at $Python"
}

Write-Host "Starting simulator/race reset flow..."
$startArgs = @{
    Python = $Python
    SimRoot = $SimRoot
    Tag = "${Tag}_start"
    RaceCheckS = $RaceCheckS
    UiAttempts = $RaceAttempts
}
if ($NoRelaunch) { $startArgs.NoRelaunch = $true }
if ($ForceRelaunch) { $startArgs.ForceRelaunch = $true }
if ($NoUiClicks) { $startArgs.NoUiClicks = $true }

& "$RepoRoot\scripts\start_windows_aigp_race.ps1" @startArgs
Write-Host "Race start confirmed."

Write-Host "Launching strict official gate-1 validation..."
$valArgs = @{
    Python = $Python
    Tag = $SmokeTag
    ProbeDuration = $ProbeDuration
    SmokeDuration = $SmokeDuration
    ControlMode = $ControlMode
    CommandFrame = $CommandFrame
    CommandYawMode = $CommandYawMode
    PolicyActionJson = $PolicyActionJson
}

if ($RequireOfficialRaceProgress.IsPresent) {
    $valArgs.RequireOfficialRaceProgress = $true
}
if ($PolicyCallable -ne "") {
    $valArgs.PolicyCallable = $PolicyCallable
}

& "$RepoRoot\scripts\run_windows_official_gate1_validation.ps1" @valArgs
