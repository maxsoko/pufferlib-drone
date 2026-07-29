param(
    [string]$Tag = "vq2_n522_corrected_geometry_gate2_bounded_004",
    [ValidateRange(12, 14)]
    [int]$DurationSeconds = 14
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$PrefixCheckpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n294_full_blend_refine\alpha_0p60.bin"
$Gate2Checkpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n506_corrected_geometry_stress_decoder\stress_only_ridge_1e-7.bin"
$FixedPrefixReport = "C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_n295_gate1_bounded_001.json"
$GeometryAnalysis = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n519_n483_corrected_geometry_analysis\report.json"
$ShadowReport = "C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_n521_corrected_geometry_shadow_005.json"
$ShadowParity = "C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_n521_corrected_geometry_shadow_005_parity.json"
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
Assert-SHA256 $Gate2Checkpoint "55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f"
Assert-SHA256 $FixedPrefixReport "5f46b74bd5636ba282c9868cff7f7d17b409965e39987314479cd0e2f5728091"
Assert-SHA256 $GeometryAnalysis "ab6fcb5701bb8477d4ab92d0f649cb64e4643bef62f6e9f14db07af9d2e2a35e"
Assert-SHA256 $ShadowReport "92053ff9860a157963cc234844a817647a7eb81af56fab51dc48ba58bdd8c1c0"
Assert-SHA256 $ShadowParity "88f34aa346a926f165ad6191f258a84ff837962dda3c5a047c090d042ec44a0e"
Assert-SHA256 $CallablePath "f54c885ec9934a648a06412cee2f36686a50c709d2ed9bac8fa909ffc427f3e0"
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
    throw "refusing to overwrite existing N522 evidence: $JsonPath"
}

# Both phases are complete recurrent Puffer checkpoints. The public corrected
# Gate-2 vector seeds policy observations only; it never generates an action.
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
    throw "N522 corrected-geometry bounded attempt failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed -or
    [int]$Report.official_active_gate_index -lt 2 -or
    [int]$Report.official_race_finish_time_ns -ne -1 -or
    [bool]$Report.crash_detected) {
    throw "N522 did not prove a clean Gate-2 completion"
}
if (-not [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    -not [bool]$Report.control_inputs.official_reset_start.reset_detected) {
    throw "N522 lacks detected reset proof"
}
if ([int]$Report.sitl.commands_sent -le 0 -or
    [double]$Report.sitl.effective_command_hz -lt 50.0 -or
    [double]$Report.sitl.effective_command_hz -ge 100.0 -or
    [int]$Report.sitl.command_rate_violations -ne 0) {
    throw "N522 command transport is invalid"
}
if ([int]$Report.control_inputs.disarm_commands_sent -ne 1) {
    throw "N522 lacks exactly one exit disarm"
}
Write-Output $JsonPath
