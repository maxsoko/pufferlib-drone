param(
    [string]$Tag = "vq2_n521_corrected_geometry_shadow_005",
    [ValidateRange(8, 12)]
    [int]$DurationSeconds = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$PrefixCheckpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n294_full_blend_refine\alpha_0p60.bin"
$Gate2Checkpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n506_corrected_geometry_stress_decoder\stress_only_ridge_1e-7.bin"
$FixedPrefixReport = "C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_n295_gate1_bounded_001.json"
$CallablePath = Join-Path $RepoRoot "scripts\policy_callable_vq2_gate2_composite.py"
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Verifier = Join-Path $RepoRoot "scripts\verify_live_policy_shadow.py"
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
Assert-SHA256 $Gate2Checkpoint "55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f"
Assert-SHA256 $FixedPrefixReport "5f46b74bd5636ba282c9868cff7f7d17b409965e39987314479cd0e2f5728091"
Assert-SHA256 $CallablePath "f54c885ec9934a648a06412cee2f36686a50c709d2ed9bac8fa909ffc427f3e0"
Assert-SHA256 $Runner "094eb6b14a7e6b2145803fca87e10133b67f351f1d8f2223b91be54a1d8526a8"
Assert-SHA256 $Verifier "9e1329e89f5dab9a0c19171ad5c42a0441e7a9f3137faaf5056fab25d799aa40"
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
$ParityPath = Join-Path $LogRoot ($Tag + "_parity.json")
if ((Test-Path $JsonPath) -or (Test-Path $ParityPath)) {
    throw "refusing to overwrite existing N521 shadow evidence"
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
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_X = "14.51880584"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Y = "0.41894794"
$env:PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_Z = "0.47249277"
$env:PUFFER_POLICY_GATE2_ASSOCIATION_JUMP_THRESHOLD_M = "0"
$env:PUFFER_POLICY_GATE2_RESEED_BEARING_ON_ASSOCIATION = "0"
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
    throw "N521 corrected-geometry shadow failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed -or
    [int]$Report.control_inputs.arm_commands_sent -ne 0 -or
    [int]$Report.control_inputs.disarm_commands_sent -ne 0 -or
    [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    [int]$Report.sitl.commands_sent -ne 0 -or
    [int]$Report.control_inputs.policy_shadow_cadence_probe.mavlink_setpoints_sent -ne 0) {
    throw "N521 shadow acceptance or zero-command contract failed"
}

$VerifyArgs = @(
    $Verifier,
    $JsonPath,
    $PrefixCheckpoint,
    "--gate2-checkpoint", $Gate2Checkpoint,
    "--policy-callable", $CallablePath,
    "--json-path", $ParityPath,
    "--min-inference-hz", "50"
)
& $Python @VerifyArgs
if ($LASTEXITCODE -ne 0) {
    throw "N521 corrected-geometry shadow parity failed"
}
Write-Output $JsonPath
Write-Output $ParityPath
