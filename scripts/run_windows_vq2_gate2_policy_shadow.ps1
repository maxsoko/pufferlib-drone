param(
    [string]$Tag = "vq2_n397_gate2_composite_shadow_001",
    [ValidateRange(5, 12)]
    [int]$DurationSeconds = 10
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
    throw "refusing to overwrite existing shadow evidence: $JsonPath"
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
    "--policy-trace-max-samples", "1200",
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
    "--policy-shadow-only",
    "--no-arm-on-start",
    "--json-path", $JsonPath,
    "--csv-path", $CsvPath
)

& $Python @RunnerArgs *> $StdoutPath
if ($LASTEXITCODE -ne 0) {
    Get-Content $StdoutPath -Tail 80
    throw "VQ2 Gate-2 composite shadow failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed) {
    throw "VQ2 Gate-2 shadow report did not pass acceptance"
}
if ([int]$Report.control_inputs.arm_commands_sent -ne 0 -or
    [int]$Report.control_inputs.disarm_commands_sent -ne 0 -or
    [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    [int]$Report.sitl.commands_sent -ne 0 -or
    [int]$Report.control_inputs.policy_shadow_cadence_probe.mavlink_setpoints_sent -ne 0) {
    throw "shadow lifecycle/control counters are nonzero"
}
Write-Output $JsonPath
