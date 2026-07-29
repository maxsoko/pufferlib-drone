param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "",
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
    [int]$EventY = 277,
    [int]$RaceX = 1605,
    [int]$RaceY = 872,
    [int]$UiAttempts = 3,
    [int]$RaceCheckS = 5,
    [double]$MaxReusableRaceAgeS = 300.0,
    [switch]$NoRelaunch,
    [switch]$ForceRelaunch,
    [switch]$DiscoverOnly,
    [switch]$NoUiClicks,
    [switch]$SkipLoginClick
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if ($NoRelaunch -and $ForceRelaunch) {
    throw "-NoRelaunch and -ForceRelaunch are mutually exclusive"
}
if ($MaxReusableRaceAgeS -le 0.0) {
    throw "MaxReusableRaceAgeS must be positive"
}

function Find-FlightSimExe([string]$RequestedPath) {
    if ($RequestedPath -ne "") {
        if ((Test-Path $RequestedPath -PathType Leaf) -and ([IO.Path]::GetFileName($RequestedPath) -ieq "FlightSim.exe")) {
            return (Resolve-Path $RequestedPath).Path
        }
        if (Test-Path $RequestedPath -PathType Container) {
            $direct = Join-Path $RequestedPath "FlightSim.exe"
            if (Test-Path $direct -PathType Leaf) {
                return (Resolve-Path $direct).Path
            }
            $nested = Get-ChildItem -Path $RequestedPath -Filter "FlightSim.exe" -File -Recurse -ErrorAction SilentlyContinue |
                Sort-Object FullName |
                Select-Object -First 1
            if ($null -ne $nested) {
                return $nested.FullName
            }
        }
        throw "FlightSim.exe not found from requested path $RequestedPath"
    }

    $desktop = [Environment]::GetFolderPath("Desktop")
    $installs = Get-ChildItem -Path $desktop -Directory -Filter "AI-GP Simulator v*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            $versionText = $_.Name -replace '^AI-GP Simulator v', ''
            $parsedVersion = [version]"0.0.0"
            [version]::TryParse($versionText, [ref]$parsedVersion) | Out-Null
            [PSCustomObject]@{
                Directory = $_
                Version = $parsedVersion
            }
        } |
        Sort-Object Version -Descending

    $newest = $installs | Select-Object -First 1
    if ($null -eq $newest) {
        throw "No simulator install found under Desktop\AI-GP Simulator v*; pass -SimRoot explicitly"
    }
    $found = Get-ChildItem -Path $newest.Directory.FullName -Filter "FlightSim.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        Select-Object -First 1
    if ($null -eq $found) {
        throw "Newest simulator v$($newest.Version) is incomplete: no FlightSim.exe under $($newest.Directory.FullName). Finish installation or pass -SimRoot explicitly."
    }
    Write-Host "Auto-selected simulator $($newest.Version): $($found.FullName)"
    return $found.FullName
}

$FlightSimExe = Find-FlightSimExe $SimRoot
$ResolvedSimRoot = Split-Path -Parent $FlightSimExe
if ($DiscoverOnly) {
    $selectedVersion = $null
    if ($FlightSimExe -match 'AI-GP Simulator v([^\\]+)') {
        $selectedVersion = $Matches[1]
    }
    [PSCustomObject]@{
        selected_version = $selectedVersion
        executable = $FlightSimExe
        working_directory = $ResolvedSimRoot
        process_started = $false
    } | ConvertTo-Json -Depth 4
    return
}
if (-not (Test-Path $Python)) {
    throw "Python runtime not found at $Python"
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class AigpWinCtl {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr dpiContext);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
  [DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
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
    # These coordinates are measured physical pixels relative to the
    # simulator's fixed 1920x1080 outer window. Convert them to physical screen
    # coordinates: the competition host may place the window away from the
    # primary display origin. PowerShell is also DPI-virtualized on this host.
    $proc = Get-SimProcess
    if ($null -eq $proc) {
        throw "DCGame-Win64-Shipping is not running"
    }
    $previousDpiContext = [AigpWinCtl]::SetThreadDpiAwarenessContext([IntPtr](-4))
    try {
        $rect = New-Object AigpWinCtl+RECT
        if (-not [AigpWinCtl]::GetWindowRect($proc.MainWindowHandle, [ref]$rect)) {
            throw "GetWindowRect failed for simulator process $($proc.Id)"
        }
        [AigpWinCtl]::SetCursorPos($rect.Left + $X, $rect.Top + $Y) | Out-Null
        Wait-Sec 0.1
        [AigpWinCtl]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
        Wait-Sec 0.1
        [AigpWinCtl]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
    } finally {
        if ($previousDpiContext -ne [IntPtr]::Zero) {
            [AigpWinCtl]::SetThreadDpiAwarenessContext($previousDpiContext) | Out-Null
        }
    }
}

function Invoke-InactiveRaceRecovery {
    # MAVLink 31000 is accepted only after a race session is active.  v3385
    # can remain in an inactive pre-race screen after a controller exits.  The
    # pause menu's Back To Main Menu action clears that stale vehicle/throttle
    # state while preserving the healthy simulator process.
    Focus-SimWindow
    [AigpWinCtl]::keybd_event(0x1B, 0, 0, [UIntPtr]::Zero)
    Wait-Sec 0.1
    [AigpWinCtl]::keybd_event(0x1B, 0, 2, [UIntPtr]::Zero)
    Wait-Sec 0.5
    # Back To Main Menu uses the same measured physical-pixel convention as
    # the login, event, and race controls.
    # v3385's 1942x1136 outer window places the button center at y=805. The
    # former y=755 coordinate selected Toggle HUD and left the pause menu open.
    Invoke-Click 480 805
    Wait-Sec 4.0
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

if ($ForceRelaunch) {
    Get-Process DCGame-Win64-Shipping,FlightSim -ErrorAction SilentlyContinue | Stop-Process -Force
    Wait-Sec 2.0
}

$ExistingSim = Get-SimProcess
if ($null -eq $ExistingSim) {
    if ($NoRelaunch) {
        throw "Simulator is not running and -NoRelaunch was requested"
    }
    Write-Host "Simulator is not running; launching $FlightSimExe"
    Start-Process -FilePath $FlightSimExe -WorkingDirectory $ResolvedSimRoot
    Wait-Sec $LaunchWaitS
} else {
    Write-Host "Reusing simulator process $($ExistingSim.Id); no process restart"
}

Focus-SimWindow

$raceJson = Join-Path "logs/sitl" ("race_start_" + $Tag + ".json")
$raceReport = $null

# A running process may already be inside an active race. Confirm that state
# before clicking the login/event/race UI. Normal attempt restarts are handled
# later with MAVLink command 31000, not by this process helper.
if (-not $ForceRelaunch) {
    $raceReport = Invoke-RaceStartWait $raceJson ([Math]::Max(2, $RaceCheckS))
    $raceAgeS = [double]::PositiveInfinity
    if ($raceReport.race_started -eq $true -and $null -ne $raceReport.race_status) {
        $raceAgeS = ([double]$raceReport.race_status.sim_boot_time_ms - [double]$raceReport.race_status.race_start_boot_time_ms) / 1000.0
    }
    # A long-running race is not stale merely because its boot-time age is
    # large. The policy runner immediately issues MAVLink command 31000, so
    # reuse any live armed/active session that can accept that reset. Candidate
    # 012 proved that opening UI recovery solely at 300 s can park v3385 in an
    # inactive screen while its old race telemetry remains observable.
    $raceResetReady = (
        $raceReport.race_started -eq $true -and
        $raceAgeS -ge 0.0 -and
        [int]$raceReport.base_mode -eq 193 -and
        [int]$raceReport.system_status -eq 4
    )
    if ($raceResetReady) {
        Write-Host (
            "Existing race is MAVLink-31000 reset-ready (age {0:N1}s); leaving simulator process and UI untouched" `
                -f $raceAgeS
        )
        $raceReport | ConvertTo-Json -Depth 8
        return
    }
    if ($null -ne $ExistingSim) {
        if ($raceReport.race_started -eq $true) {
            Write-Host (
                "Race is not MAVLink-31000 reset-ready (age {0:N1}s, base {1}, status {2}); recovering through the UI without relaunch" `
                    -f $raceAgeS, $raceReport.base_mode, $raceReport.system_status
            )
        } else {
            Write-Host "Healthy process has no active race; recovering through the UI without relaunch"
        }
        Invoke-InactiveRaceRecovery
    }
}

if (-not $NoUiClicks) {
    if ($UiAttempts -le 0) {
        throw "UiAttempts must be positive when UI clicks are enabled"
    }
    for ($attempt = 1; $attempt -le $UiAttempts; $attempt++) {
        Invoke-UiRaceStartSequence
        $raceReport = Invoke-RaceStartWait $raceJson $RaceCheckS
        if (
            $raceReport.race_started -eq $true -and
            [int]$raceReport.base_mode -eq 193 -and
            [int]$raceReport.system_status -eq 4
        ) {
            break
        }
    }
} else {
    $raceReport = Invoke-RaceStartWait $raceJson $WaitRaceStartS
}

if (
    (
        $raceReport.race_started -ne $true -or
        [int]$raceReport.base_mode -ne 193 -or
        [int]$raceReport.system_status -ne 4
    ) -and
    -not $NoUiClicks
) {
    $raceReport = Invoke-RaceStartWait $raceJson $WaitRaceStartS
}
$raceReport | ConvertTo-Json -Depth 8
if (
    $raceReport.race_started -ne $true -or
    [int]$raceReport.base_mode -ne 193 -or
    [int]$raceReport.system_status -ne 4
) {
    throw "Race did not become MAVLink-31000 reset-ready before timeout; see $raceJson"
}
