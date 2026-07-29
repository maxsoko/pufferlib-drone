param(
    [string]$Tag = "vq2_n399_gate2_bounded_001",
    [ValidateRange(12, 14)]
    [int]$DurationSeconds = 14
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$PrefixCheckpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n294_full_blend_refine\alpha_0p60.bin"
$Gate2Checkpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n380_gate2_full_policy_reward_bracket\teacher100_align20\update_0001.bin"
$CallablePath = Join-Path $RepoRoot "scripts\policy_callable_vq2_gate2_composite.py"
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Callable = $CallablePath + ":infer"
$Acceptance = Join-Path $RepoRoot "config\sitl_competition_acceptance.json"
$LogRoot = "C:\Users\anon\code\pufferlib-drone\logs\sitl"

function Assert-SHA256([string]$Path, [string]$Expected) {
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "missing deployment artifact: $Path"
    }
    $Actual = (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "deployment hash mismatch for ${Path}: $Actual"
    }
}

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "missing Windows controller Python: $Python"
}
Assert-SHA256 $PrefixCheckpoint "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6"
Assert-SHA256 $Gate2Checkpoint "6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a"
Assert-SHA256 $CallablePath "1998a3370df919bf0eaf16e57d3928de41850fefac9bc0ea026d0ba783d653bc"
Assert-SHA256 $Runner "f8dcf22713637b42474db7fab6d6ebe7769b8cbdacedfe75fd3af058d5dc58c7"

$SimulatorProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "FlightSim.exe" -and $_.ExecutablePath -eq $Simulator
})
if ($SimulatorProcesses.Count -ne 1) {
    throw "expected exactly one authoritative VQ2 FlightSim process"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$JsonPath = Join-Path $LogRoot ($Tag + ".json")
$CsvPath = Join-Path $LogRoot ($Tag + ".csv")
$StdoutPath = Join-Path $LogRoot ($Tag + ".stdout.txt")
if (Test-Path $JsonPath) {
    throw "refusing to overwrite existing attempt evidence: $JsonPath"
}

$env:PUFFER_POLICY_CHECKPOINT_PATH = $PrefixCheckpoint
$env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH = $PrefixCheckpoint
$env:PUFFER_POLICY_GATE2_CHECKPOINT_PATH = $Gate2Checkpoint
$env:PUFFER_POLICY_INPUT_DIM = "32"
$env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "4"
$env:PUFFER_POLICY_NATIVE_BF16 = "0"
$env:PUFFER_POLICY_RACE_PHASE_DENOMINATOR = "6"

$RunnerArgs = @(
    $Runner,
    "--acceptance-config", $Acceptance,
    "--endpoint", "udpin:0.0.0.0:14550",
    "--heartbeat-hz", "2",
    "--command-hz", "80",
    "--duration", [string]$DurationSeconds,
    "--control-mode", "policy-attitude",
    "--attitude-mode", "body_rates",
    "--policy-callable", $Callable,
    "--policy-gate-phase-onehot-adapter-observation",
    "--policy-race-phase-denominator", "6",
    "--policy-trace-hz", "100",
    "--policy-trace-max-samples", "1800",
    "--camera-uptilt-deg", "1.920944634732011",
    "--detector-min-area-px", "300",
    "--detector-max-aspect-error", "0.8",
    "--detector-min-fill-ratio", "0.1",
    "--require-telemetry",
    "--require-camera",
    "--min-telemetry-messages", "1",
    "--min-camera-frames", "1",
    "--max-command-rate-violations", "0",
    "--max-telemetry-dropouts", "2",
    "--official-reset-on-start",
    "--target-gate-count", "2",
    "--stop-after-official-gate-index", "2",
    "--min-gate-passes", "2",
    "--require-official-race-progress",
    "--json-path", $JsonPath,
    "--csv-path", $CsvPath
)

& $Python @RunnerArgs *> $StdoutPath
if ($LASTEXITCODE -ne 0) {
    Get-Content $StdoutPath -Tail 100
    throw "VQ2 Gate-2 bounded attempt failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed) {
    throw "VQ2 Gate-2 bounded report did not pass acceptance"
}
if ([int]$Report.official_active_gate_index -lt 2) {
    throw "bounded attempt did not reach official Gate-2 completion"
}
if (-not [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    -not [bool]$Report.control_inputs.official_reset_start.reset_detected) {
    throw "bounded attempt lacks detected reset proof"
}
if ([int]$Report.sitl.commands_sent -le 0 -or
    [double]$Report.sitl.effective_command_hz -lt 50.0 -or
    [double]$Report.sitl.effective_command_hz -ge 100.0 -or
    [int]$Report.sitl.command_rate_violations -ne 0) {
    throw "bounded attempt command transport is invalid"
}
if ([int]$Report.control_inputs.disarm_commands_sent -ne 1) {
    throw "bounded attempt lacks exactly one exit disarm"
}
Write-Output $JsonPath
