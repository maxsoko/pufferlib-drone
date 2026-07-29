param(
    [ValidateSet("Shadow", "FullLap")]
    [string]$Mode = "Shadow",
    [ValidateSet("N209", "N214", "N218", "N219", "N220", "N221", "N222", "N223", "N224", "N225", "N226", "N227", "N228")]
    [string]$TailCandidate = "N209",
    [string]$Python = "C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe",
    [string]$SimRoot = "C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe",
    [string]$Tag = "n210_h12_n209_unified_tail",
    [int]$ShadowDuration = 10,
    [int]$FlightDuration = 45,
    [int]$CommandHz = 80,
    [int]$Repeats = 1,
    [int]$MinValidRuns = 1,
    [switch]$ForceRelaunch
)

# N210 deployment entry point. H12/N203 owns official Gates 1--3 and the
# zero-state N209 recurrent checkpoint owns the entire Gates 4--6 tail.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Resolve-NativeFilePath([string]$Path) {
    $resolved = Resolve-Path -LiteralPath $Path
    if ($resolved.ProviderPath) {
        return $resolved.ProviderPath
    }
    return $resolved.Path
}

function Assert-SHA256([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required deployment artifact not found: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected, got $actual"
    }
}

$PrefixCheckpoint = (
    "C:\Users\anon\code\pufferlib-drone\checkpoints\" +
    "drone_race_full_policy_gate3_visual\" +
    "v6c_latest_obsmatch_r135_dropout_e10.bin"
)
if ($TailCandidate -eq "N228") {
    # N228 preserves N227 Gate-4 lateral control and changes only vertical
    # damping plus the bounded post-plane miss response.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n228_gate4_vertical_pd.py"
    $PolicyCallableSpec = "scripts/policy_callable_n228_gate4_vertical_pd.py:infer"
    $ExpectedPolicyCallableHash = "3531c3be26fda31e6e4355bd2baf30e5fb44d962ce91a58529f5f23c7ddbd74a"
    $DeploymentLabel = "N228"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 3
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N227") {
    # N227 retains N226's live-proven Gates 1--3 phase schedule and replaces
    # only Gate-4 roll/thrust with observable terminal-plane PD control.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n227_gate4_terminal_pd.py"
    $PolicyCallableSpec = "scripts/policy_callable_n227_gate4_terminal_pd.py:infer"
    $ExpectedPolicyCallableHash = "f0fd8ed0017d8130da7bbab40e07f6c9eacdf6f8aa9362582661ad6a6ec2bde4"
    $DeploymentLabel = "N227"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 3
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N226") {
    # N226 restores the historically live-proven, non-predictive H12 prefix
    # observation contract at official Gates 1--3. Dropout propagation and
    # its roll-aware correction begin only at the N220 Gates 4--6 tail.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n225_gate3_vertical_guard.py"
    $PolicyCallableSpec = "scripts/policy_callable_n225_gate3_vertical_guard.py:infer"
    $ExpectedPolicyCallableHash = "c2f104c1264f31df1d22a167dcfb32d47c01b1479229925076ba780e057eb29e"
    $DeploymentLabel = "N226"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 3
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N225") {
    # N225 changes only four failure-specific Gate-3 sub-hover thrust samples
    # on the N224 timeout trace and is invariant on five H12 Gate-3 passes.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n225_gate3_vertical_guard.py"
    $PolicyCallableSpec = "scripts/policy_callable_n225_gate3_vertical_guard.py:infer"
    $ExpectedPolicyCallableHash = "c2f104c1264f31df1d22a167dcfb32d47c01b1479229925076ba780e057eb29e"
    $DeploymentLabel = "N225"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N224") {
    # N224 retains N223's last-good association horizon, but restores the
    # historical propagation timestep for pose/rate filtering. This isolates
    # gate membership from the H12 prefix dynamics that passed Gates 1--3.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n220_observable_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n220_observable_tail.py:infer"
    $ExpectedPolicyCallableHash = "dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746"
    $DeploymentLabel = "N224"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N223") {
    # Preserve the proven H12 Gates 1--3 prefix and N220 observable Gates
    # 4--6 tail. N223 changes only the gate-association clock contract that
    # the failed N220 Gate-2 trace identified as causal. Keep the legacy 30
    # m/s innovation bound and do not add reacquisition behavior to the flight.
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n220_observable_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n220_observable_tail.py:infer"
    $ExpectedPolicyCallableHash = "dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746"
    $DeploymentLabel = "N223"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N222") {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n221_observable_course.py"
    $PolicyCallableSpec = "scripts/policy_callable_n221_observable_course.py:infer"
    $ExpectedPolicyCallableHash = "468eb14d2234d640d644a06c9877759374553905c3116bc47d6e69e7211285c7"
    $DeploymentLabel = "N222"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $true
} elseif ($TailCandidate -eq "N221") {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n221_observable_course.py"
    $PolicyCallableSpec = "scripts/policy_callable_n221_observable_course.py:infer"
    $ExpectedPolicyCallableHash = "468eb14d2234d640d644a06c9877759374553905c3116bc47d6e69e7211285c7"
    $DeploymentLabel = "N221"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N220") {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n220_observable_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n220_observable_tail.py:infer"
    $ExpectedPolicyCallableHash = "dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746"
    $DeploymentLabel = "N220"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $true
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -eq "N219") {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n219_predictor_adaptation\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784418874041\" +
        "0000000000032768.bin"
    )
    $ExpectedTailHash = "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n219_unified_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n219_unified_tail.py:infer"
    $ExpectedPolicyCallableHash = "28cce6118eb9305b2526895c23258f92ad546f6d791da113572ee2668937b76a"
    $DeploymentLabel = "N219"
    $PredictGateDropout = $true
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $false
    $BoundedGateReacquisition = $false
} elseif ($TailCandidate -in @("N214", "N218")) {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n214_teacher_intervention_blend000\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784415204755\" +
        "0000000000065536.bin"
    )
    $ExpectedTailHash = "999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n214_unified_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n214_unified_tail.py:infer"
    $ExpectedPolicyCallableHash = "b13b36d7c22af83eb6f2c0e1faf8f8df340fc02e49df8d555064b8aa1751d4ba"
    $DeploymentLabel = if ($TailCandidate -eq "N218") { "N218" } else { "N215" }
    $PredictGateDropout = $TailCandidate -eq "N218"
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $false
    $BoundedGateReacquisition = $false
} else {
    $TailCheckpoint = Join-Path $RepoRoot (
        "logs\drone_race_full_policy_six_gate_bootstrap\" +
        "n209_measured_entry_mixture\checkpoints\" +
        "drone_race_full_policy_six_gate_bootstrap\1784409692380\" +
        "0000000000589824.bin"
    )
    $ExpectedTailHash = "efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb"
    $PolicyCallable = Join-Path $RepoRoot "scripts\policy_callable_n209_unified_tail.py"
    $PolicyCallableSpec = "scripts/policy_callable_n209_unified_tail.py:infer"
    $ExpectedPolicyCallableHash = "5a00e97df818477e18424cbccb9c21fa53c435a05a8602572390419c420f7769"
    $DeploymentLabel = "N210"
    $PredictGateDropout = $false
    $PredictGateDropoutStartIndex = 0
    $ControlAwareGatePredictor = $false
    $BoundedGateReacquisition = $false
}
$Runner = Join-Path $RepoRoot "scripts\drone_sitl_competition_smoke.py"
$Verifier = Join-Path $RepoRoot "scripts\verify_live_policy_shadow.py"
$FullPolicyRunner = Join-Path $RepoRoot "scripts\run_windows_full_policy.ps1"
$RaceStartup = Join-Path $RepoRoot "scripts\start_windows_aigp_race.ps1"
$PolicyValidation = Join-Path $RepoRoot "scripts\run_official_policy_validation.py"

Assert-SHA256 $PrefixCheckpoint "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
Assert-SHA256 $TailCheckpoint $ExpectedTailHash
Assert-SHA256 $PolicyCallable $ExpectedPolicyCallableHash
Assert-SHA256 $Runner "e6f2d717138456abba5b9a95905f91144541a8a6bd939a3155fc95881dfdb3b3"
Assert-SHA256 $Verifier "7538094e7065300c42495cd5d380fe7e967daf94065cbf85caa88cc35d253198"
Assert-SHA256 $FullPolicyRunner "892a78ce01a90f8230d1c81271d4f05a71105e4e5b626d7c5febff39cde89d46"
Assert-SHA256 $RaceStartup "d9e97ff6ac38c7a476eebc5ddfa5fd6f47d705c11f613922c17e9690f9031f7d"
Assert-SHA256 $PolicyValidation "d36e52e4a19c2b24c8f0075d76ea3b61e559fa10b5af506c7fadcb15625df5db"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found: $Python"
}
if ($ShadowDuration -le 0 -or $FlightDuration -le 0) {
    throw "ShadowDuration and FlightDuration must be positive"
}
if ($CommandHz -lt 50 -or $CommandHz -ge 100) {
    throw "CommandHz must be in [50,100)"
}

# The deployed H12 Gate-3 hysteresis variant is frozen at the promoted 1.2 m
# threshold. N209 receives the standard filtered observation contract.
$env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.2"
Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX -ErrorAction SilentlyContinue
Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX -ErrorAction SilentlyContinue
if ($PredictGateDropout) {
    $env:PUFFER_POLICY_PREDICT_GATE_DROPOUT = "1"
    $env:PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX = "$PredictGateDropoutStartIndex"
} else {
    Remove-Item Env:PUFFER_POLICY_PREDICT_GATE_DROPOUT -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX -ErrorAction SilentlyContinue
}
if ($ControlAwareGatePredictor) {
    $env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR = "1"
    if ($TailCandidate -in @("N220", "N223", "N224", "N225", "N226", "N227", "N228")) {
        # Preserve the proven H12/N219 prefix observation contract. The
        # roll-aware propagation was validated only for official Gates 4--6.
        $env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX = "3"
    } else {
        $env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX = "0"
    }
} else {
    Remove-Item Env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX -ErrorAction SilentlyContinue
}
if ($BoundedGateReacquisition) {
    $env:PUFFER_POLICY_BOUNDED_GATE_REACQUISITION = "1"
    $env:PUFFER_POLICY_GATE_ASSOCIATION_MAX_SPEED_M_S = "10.0"
    $env:PUFFER_POLICY_REACQUIRE_MAX_CANDIDATE_JUMP_M = "3.0"
    $env:PUFFER_POLICY_REACQUIRE_CLOSER_MARGIN_M = "0.5"
} else {
    Remove-Item Env:PUFFER_POLICY_BOUNDED_GATE_REACQUISITION -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_GATE_ASSOCIATION_MAX_SPEED_M_S -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_REACQUIRE_MAX_CANDIDATE_JUMP_M -ErrorAction SilentlyContinue
    Remove-Item Env:PUFFER_POLICY_REACQUIRE_CLOSER_MARGIN_M -ErrorAction SilentlyContinue
}

if ($Mode -eq "Shadow") {
    $simProcess = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
    if ($null -eq $simProcess -or -not $simProcess.Responding) {
        throw "Shadow mode requires an already-running responsive simulator"
    }

    $env:PUFFER_POLICY_CHECKPOINT_PATH = Resolve-NativeFilePath $PrefixCheckpoint
    $env:PUFFER_POLICY_PREFIX_CHECKPOINT_PATH = Resolve-NativeFilePath $PrefixCheckpoint
    $env:PUFFER_POLICY_GATE4_CHECKPOINT_PATH = Resolve-NativeFilePath $TailCheckpoint
    $env:PUFFER_POLICY_GATE5_CHECKPOINT_PATH = Resolve-NativeFilePath $TailCheckpoint
    $env:PUFFER_POLICY_GATE6_CHECKPOINT_PATH = Resolve-NativeFilePath $TailCheckpoint
    $env:PUFFER_POLICY_INPUT_DIM = "32"
    $env:PUFFER_POLICY_LAYOUT_PRECISION_BYTES = "4"
    $env:PUFFER_POLICY_NATIVE_BF16 = "0"

    $ShadowJson = Join-Path $RepoRoot "logs\sitl\${Tag}_shadow.json"
    $ParityJson = Join-Path $RepoRoot "logs\sitl\${Tag}_shadow_parity.json"
    $OutputDirectory = Split-Path -Parent $ShadowJson
    if (-not (Test-Path -LiteralPath $OutputDirectory)) {
        New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    }
    $NativeOutputDirectory = Resolve-NativeFilePath $OutputDirectory
    $NativeShadowJson = Join-Path $NativeOutputDirectory (Split-Path -Leaf $ShadowJson)
    $NativeParityJson = Join-Path $NativeOutputDirectory (Split-Path -Leaf $ParityJson)
    $shadowArgs = @(
        "scripts/drone_sitl_competition_smoke.py"
        "--endpoint", "udpin:0.0.0.0:14550"
        "--camera-host", "0.0.0.0"
        "--camera-port", "5600"
        "--duration", "$ShadowDuration"
        "--command-hz", "$CommandHz"
        "--control-mode", "policy-attitude"
        "--policy-callable", $PolicyCallableSpec
        "--policy-shadow-only"
        "--no-arm-on-start"
        "--policy-shadow-calibration-timeout-s", "10"
        "--policy-trace-hz", "1000"
        "--policy-trace-max-samples", "4000"
        "--policy-gate-phase-onehot-adapter-observation"
        "--policy-hybrid-prefix-confidence-observation"
        "--policy-race-phase-denominator", "6"
        "--acceptance-config", "config/sitl_multigate_acceptance.json"
        "--require-telemetry"
        "--require-camera"
        "--json-path", $NativeShadowJson
    )
    & $Python @shadowArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$DeploymentLabel passive shadow failed with exit code $LASTEXITCODE"
    }

    $verifyArgs = @(
        "scripts/verify_live_policy_shadow.py"
        (Resolve-NativeFilePath $ShadowJson)
        (Resolve-NativeFilePath $PrefixCheckpoint)
        "--gate4-checkpoint", (Resolve-NativeFilePath $TailCheckpoint)
        "--gate5-checkpoint", (Resolve-NativeFilePath $TailCheckpoint)
        "--gate6-checkpoint", (Resolve-NativeFilePath $TailCheckpoint)
        "--policy-callable", (Resolve-NativeFilePath $PolicyCallable)
        "--hybrid-prefix-confidence"
        "--min-inference-hz", "10.0"
        "--json-path", $NativeParityJson
    )
    & $Python @verifyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$DeploymentLabel passive shadow parity failed with exit code $LASTEXITCODE"
    }
    Write-Host "$DeploymentLabel passive shadow and parity passed: $ParityJson"
    exit 0
}

$runArgs = @{
    PolicyCheckpoint = $TailCheckpoint
    Gate5Checkpoint = $TailCheckpoint
    Gate6Checkpoint = $TailCheckpoint
    SixGateComposite = $true
    HybridPrefixCheckpoint = $PrefixCheckpoint
    HybridPolicyCallable = $PolicyCallableSpec
    Python = $Python
    SimRoot = $SimRoot
    Tag = $Tag
    SmokeDuration = $FlightDuration
    CommandHz = $CommandHz
    TargetGateCount = 6
    Repeats = $Repeats
    MinValidRuns = $MinValidRuns
    PolicyLayoutPrecisionBytes = 4
}
if ($ForceRelaunch) {
    $runArgs.ForceRelaunch = $true
}
& $FullPolicyRunner @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "$DeploymentLabel FullLap validation failed with exit code $LASTEXITCODE"
}
