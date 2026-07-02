param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "$env:USERPROFILE\Desktop\AI-GP Simulator v1.0.3379\AIGP_3379",
    [string]$Endpoint = "udpin:0.0.0.0:14550",
    [string]$Tag = "auto_race_start",
    [int]$WaitRaceStartS = 90,
    [double]$LaunchWaitS = 18.0,
    [double]$AfterLoginWaitS = 5.0,
    [double]$AfterEventWaitS = 3.0,
    [double]$AfterRaceWaitS = 1.0,
    [int]$LoginX = 1450,
    [int]$LoginY = 716,
    [int]$EventX = 900,
    [int]$EventY = 232,
    [int]$RaceX = 1605,
    [int]$RaceY = 830,
    [int]$UiAttempts = 3,
    [int]$RaceCheckS = 5,
    [switch]$NoRelaunch,
    [switch]$NoUiClicks,
    [switch]$SkipLoginClick
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$FlightSimExe = Join-Path $SimRoot "FlightSim.exe"
if (-not (Test-Path $FlightSimExe)) {
    throw "FlightSim.exe not found at $FlightSimExe"
}
if (-not (Test-Path $Python)) {
    throw "Python runtime not found at $Python"
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class AigpWinCtl {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
}
"@

function Wait-Sec([double]$Seconds) {
    if ($Seconds -gt 0) {
        Start-Sleep -Milliseconds ([int]([Math]::Round($Seconds * 1000.0)))
    }
}

function Get-SimProcess {
    return Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
}

function Focus-SimWindow {
    $proc = Get-SimProcess
    if ($null -eq $proc) {
        throw "DCGame-Win64-Shipping is not running"
    }
    [AigpWinCtl]::ShowWindow($proc.MainWindowHandle, 9) | Out-Null
    [AigpWinCtl]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
    Wait-Sec 0.5
}

function Invoke-Click([int]$X, [int]$Y) {
    [AigpWinCtl]::SetCursorPos($X, $Y) | Out-Null
    Wait-Sec 0.1
    [AigpWinCtl]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    Wait-Sec 0.1
    [AigpWinCtl]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
}

function Wait-ReportFile([string]$Path, [int]$TimeoutS) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while (-not (Test-Path $Path)) {
        if ((Get-Date) -ge $deadline) {
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not (Test-Path $Path)) {
        throw "Race-start waiter did not write $Path"
    }

    $report = $null
    $readDeadline = (Get-Date).AddSeconds(5)
    while ($null -eq $report) {
        try {
            $report = Get-Content -Raw $Path | ConvertFrom-Json
        } catch {
            if ((Get-Date) -ge $readDeadline) {
                throw
            }
            Start-Sleep -Milliseconds 250
        }
    }
    return $report
}

function Invoke-RaceStartWait([string]$Path, [int]$DurationS) {
    Remove-Item $Path -ErrorAction SilentlyContinue
    & $Python "scripts/wait_official_race_start.py" `
        "--endpoint" $Endpoint `
        "--duration" "$DurationS" `
        "--json-path" $Path
    return Wait-ReportFile $Path ($DurationS + 10)
}

function Invoke-UiRaceStartSequence {
    Focus-SimWindow
    if (-not $SkipLoginClick) {
        Invoke-Click $LoginX $LoginY
        Wait-Sec $AfterLoginWaitS
        Focus-SimWindow
    }
    Invoke-Click $EventX $EventY
    Wait-Sec $AfterEventWaitS
    Focus-SimWindow
    Invoke-Click $RaceX $RaceY
    Wait-Sec $AfterRaceWaitS
}

if (-not $NoRelaunch) {
    Get-Process DCGame-Win64-Shipping,FlightSim -ErrorAction SilentlyContinue | Stop-Process -Force
    Wait-Sec 2.0
    Start-Process -FilePath $FlightSimExe -WorkingDirectory $SimRoot
    Wait-Sec $LaunchWaitS
}

Focus-SimWindow

$raceJson = Join-Path "logs/sitl" ("race_start_" + $Tag + ".json")
$raceReport = $null

if (-not $NoUiClicks) {
    if ($UiAttempts -le 0) {
        throw "UiAttempts must be positive when UI clicks are enabled"
    }
    for ($attempt = 1; $attempt -le $UiAttempts; $attempt++) {
        Invoke-UiRaceStartSequence
        $raceReport = Invoke-RaceStartWait $raceJson $RaceCheckS
        if ($raceReport.race_started -eq $true) {
            break
        }
    }
} else {
    $raceReport = Invoke-RaceStartWait $raceJson $WaitRaceStartS
}

if ($raceReport.race_started -ne $true -and -not $NoUiClicks) {
    $raceReport = Invoke-RaceStartWait $raceJson $WaitRaceStartS
}
$raceReport | ConvertTo-Json -Depth 8
if ($raceReport.race_started -ne $true) {
    throw "Race did not start before timeout; see $raceJson"
}
