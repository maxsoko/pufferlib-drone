param(
    [string]$Tag = "vq2_lc221_all24_numpy_shadow_001",
    [ValidateRange(8, 15)]
    [int]$DurationSeconds = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$Checkpoint = Join-Path $RepoRoot "checkpoints\vq2_lc216_all24_action_sequence_numpy.npz"
$CallablePath = Join-Path $RepoRoot "scripts\policy_callable_vq2_all24_sequence.py"
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Mask = Join-Path $RepoRoot "scripts\vq2_soft_red_mask.py"
$Callable = $CallablePath + ":policy"
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
Assert-SHA256 $Checkpoint "248d3574d3354862891b27942d239fb0d123dc8832e8c1e04db5fe12c0f3da1f"
Assert-SHA256 $CallablePath "ce64fc1422fbe29a40c239ab921c023e73cac008fa9511ee0dbc7c9e6aa73773"
Assert-SHA256 $Runner "d2cf7d4203f473561d4e4bbdcc2cfce3108004e0095df57a632015f8a61cc6f2"
Assert-SHA256 $Mask "1c220738ea9105bf35e50631f61df19ac29eb2ddefe338b6caf9530202d6dda9"

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
    throw "refusing to overwrite existing LC221 shadow evidence: $JsonPath"
}

$env:PUFFER_POLICY_CHECKPOINT_PATH = $Checkpoint
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
    "--policy-vq2-visual-observation",
    "--policy-state-hz", "64",
    "--policy-trace-hz", "2",
    "--policy-trace-max-samples", "40",
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
    Get-Content $StdoutPath -Tail 100
    throw "LC221 all-24 NumPy shadow failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed) {
    throw "LC221 shadow report did not pass acceptance"
}
if ([int]$Report.control_inputs.arm_commands_sent -ne 0 -or
    [int]$Report.control_inputs.disarm_commands_sent -ne 0 -or
    [bool]$Report.control_inputs.official_reset_start.reset_sent -or
    [int]$Report.sitl.commands_sent -ne 0 -or
    [int]$Report.control_inputs.policy_shadow_cadence_probe.mavlink_setpoints_sent -ne 0) {
    throw "LC221 shadow lifecycle/control counters are nonzero"
}
if ([int]$Report.policy_trace.inference_ticks -lt (60 * $DurationSeconds)) {
    throw "LC221 shadow did not sustain the recurrent 64 Hz schedule"
}
if ([double]$Report.policy_trace.state_schedule.realized_hz -lt 60.0) {
    throw "LC221 realized inference rate fell below 60 Hz"
}
if (@($Report.policy_trace.samples).Count -lt 10) {
    throw "LC221 shadow retained too few parity samples"
}
foreach ($Sample in @($Report.policy_trace.samples)) {
    if (@($Sample.observation).Count -ne 4119 -or @($Sample.normalized_action).Count -ne 4) {
        throw "LC221 trace ABI changed"
    }
}
$Manifest = $Report.policy_trace.deployment_manifest
if ($Manifest.checkpoint.sha256 -ne "248d3574d3354862891b27942d239fb0d123dc8832e8c1e04db5fe12c0f3da1f" -or
    $Manifest.policy_callable.sha256 -ne "ce64fc1422fbe29a40c239ab921c023e73cac008fa9511ee0dbc7c9e6aa73773" -or
    $Manifest.runner.sha256 -ne "d2cf7d4203f473561d4e4bbdcc2cfce3108004e0095df57a632015f8a61cc6f2" -or
    $Manifest.vq2_soft_red_mask.sha256 -ne "1c220738ea9105bf35e50631f61df19ac29eb2ddefe338b6caf9530202d6dda9") {
    throw "LC221 report deployment manifest changed"
}
Write-Output $JsonPath
