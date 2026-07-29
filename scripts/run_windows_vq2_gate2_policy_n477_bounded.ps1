param(
    [string]$Tag = "vq2_n477_gate2_bounded_002",
    [ValidateRange(12, 14)]
    [int]$DurationSeconds = 14
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$PrefixCheckpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n294_full_blend_refine\alpha_0p60.bin"
$Gate2Checkpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n474_full_kinematic_perturbed_decoder\ridge_1e-1.bin"
$FixedPrefixReport = "C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_n295_gate1_bounded_001.json"
$CallablePath = Join-Path $RepoRoot "scripts\policy_callable_vq2_gate2_composite.py"
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
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
Assert-SHA256 $Gate2Checkpoint "c39d3c653e33a2dbeda3eb90869fd65ae35a6ca1825380a876a8111b0b8341e6"
Assert-SHA256 $FixedPrefixReport "5f46b74bd5636ba282c9868cff7f7d17b409965e39987314479cd0e2f5728091"
Assert-SHA256 $CallablePath "b9705f20354106de809f524acb568ada952fa090c4b497a2d46a437c18dabd68"
Assert-SHA256 $Runner "094eb6b14a7e6b2145803fca87e10133b67f351f1d8f2223b91be54a1d8526a8"
Assert-SHA256 $Acceptance "ccbdef1e9b986094d1db4f0320a822cf397c37caba5630e62c6d80a91a4074d4"

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
    throw "refusing to overwrite existing N477 attempt evidence: $JsonPath"
}

$env:PUFFER_POLICY_CHECKPOINT_PATH = $PrefixCheckpoint
$env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH = $PrefixCheckpoint
$env:PUFFER_POLICY_GATE2_CHECKPOINT_PATH = $Gate2Checkpoint
$env:PUFFER_POLICY_GATE2_FIXED_PREFIX_REPORT_PATH = $FixedPrefixReport
$env:PUFFER_POLICY_INPUT_DIM = "32"
$env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "4"
$env:PUFFER_POLICY_NATIVE_BF16 = "0"
$env:PUFFER_POLICY_RACE_PHASE_DENOMINATOR = "6"
$env:PUFFER_POLICY_RUNTIME_DURATION_SECONDS = "14"
$env:PUFFER_POLICY_GATE2_TIME_LIMIT_SECONDS = "20"
$env:PUFFER_POLICY_GATE2_WORLD_FRAME_FEATURES = "1"
$env:PUFFER_POLICY_GATE2_LINEAR_GATE_FEATURES = "1"
$env:PUFFER_POLICY_GATE2_PREDICT_GATE_KINEMATICS = "1"
$env:PUFFER_POLICY_GATE2_INITIAL_FORWARD_GATE_RATE_M_S = "-4.195446884841884"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Y_M_S = "-0.030547706258741657"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Z_M_S = "-0.3521659209057582"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_X = "1.73424389"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Y = "1.72005255"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Z = "2.2471045"
$env:PUFFER_POLICY_GATE2_FORWARD_PREDICTOR_DT_SECONDS = "0.016666666666666666"

$RunnerArgs = @(
    $Runner,
    "--acceptance-config", $Acceptance,
    "--endpoint", "udpin:0.0.0.0:14550",
    "--heartbeat-hz", "2",
    "--command-hz", "80",
    "--duration", [string]$DurationSeconds,
    "--control-mode", "policy-attitude",
    "--attitude-mode", "body_rates",
    "--policy-callable", ($CallablePath + ":infer"),
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
    Get-Content $StdoutPath -Tail 120
    throw "N477 VQ2 Gate-2 bounded attempt failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed -or
    [int]$Report.official_active_gate_index -lt 2 -or
    [int]$Report.official_race_finish_time_ns -ne -1 -or
    [bool]$Report.crash_detected) {
    throw "N477 bounded report did not prove a clean Gate-2 completion"
}
if (-not [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    -not [bool]$Report.control_inputs.official_reset_start.reset_detected) {
    throw "N477 bounded attempt lacks detected reset proof"
}
if ([int]$Report.sitl.commands_sent -le 0 -or
    [double]$Report.sitl.effective_command_hz -lt 50.0 -or
    [double]$Report.sitl.effective_command_hz -ge 100.0 -or
    [int]$Report.sitl.command_rate_violations -ne 0) {
    throw "N477 bounded attempt command transport is invalid"
}
if ([int]$Report.control_inputs.disarm_commands_sent -ne 1) {
    throw "N477 bounded attempt lacks exactly one exit disarm"
}
Write-Output $JsonPath
