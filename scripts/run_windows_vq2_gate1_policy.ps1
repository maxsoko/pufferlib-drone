param(
    [ValidateSet("Shadow", "Bounded")]
    [string]$Mode = "Shadow",
    [string]$Tag = "vq2_n294_gate1_policy_shadow",
    [int]$DurationSeconds = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = "\\wsl.localhost\Ubuntu\root\pufferlib-drone"
$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe"
$Simulator = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe"
$Checkpoint = Join-Path $RepoRoot "logs\drone_race_full_policy_six_gate_bootstrap\vq2_n294_full_blend_refine\alpha_0p60.bin"
$CheckpointSha256 = "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6"
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Callable = (Join-Path $RepoRoot "scripts\policy_callable_checkpoint.py") + ":infer"
$Acceptance = Join-Path $RepoRoot "config\sitl_competition_acceptance.json"
$LogRoot = "C:\Users\anon\code\pufferlib-drone\logs\sitl"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "missing Windows controller Python: $Python"
}
if (-not (Test-Path $Checkpoint -PathType Leaf)) {
    throw "missing candidate checkpoint: $Checkpoint"
}
$ActualCheckpointSha256 = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualCheckpointSha256 -ne $CheckpointSha256) {
    throw "checkpoint hash mismatch: $ActualCheckpointSha256"
}
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

$env:PUFFER_POLICY_CHECKPOINT_PATH = $Checkpoint
$env:PUFFER_POLICY_INPUT_DIM = "32"
$env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "4"
$env:PUFFER_POLICY_NATIVE_BF16 = "0"

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
    "--json-path", $JsonPath,
    "--csv-path", $CsvPath
)

if ($Mode -eq "Shadow") {
    $RunnerArgs += @("--policy-shadow-only", "--no-arm-on-start")
} else {
    if ($DurationSeconds -gt 12) {
        throw "bounded Gate-1 duration may not exceed 12 seconds"
    }
    $RunnerArgs += @(
        "--official-reset-on-start",
        "--target-gate-count", "1",
        "--stop-after-official-gate-index", "1",
        "--min-gate-passes", "1",
        "--require-official-race-progress"
    )
}

& $Python @RunnerArgs *> $StdoutPath
if ($LASTEXITCODE -ne 0) {
    Get-Content $StdoutPath -Tail 80
    throw "VQ2 $Mode policy run failed"
}
$Report = Get-Content $JsonPath -Raw | ConvertFrom-Json
if (-not $Report.acceptance_passed) {
    throw "VQ2 $Mode report did not pass acceptance"
}
if ($Mode -eq "Bounded" -and [int]$Report.official_active_gate_index -lt 1) {
    throw "bounded attempt did not reach official Gate-1 completion"
}
Write-Output $JsonPath
