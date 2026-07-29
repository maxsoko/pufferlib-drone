from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_n209_unified_tail.ps1"


def test_runner_pins_exact_sim_python_controller_and_checkpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'C:\\Users\\anon\\Desktop\\AI-GP Simulator v1.0.3385\\AIGP_3385\\FlightSim.exe' in source
    assert 'C:\\Users\\anon\\code\\pufferlib-drone\\.venv-win\\Scripts\\python.exe' in source
    assert "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760" in source
    assert "efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb" in source
    assert "5a00e97df818477e18424cbccb9c21fa53c435a05a8602572390419c420f7769" in source
    assert "policy_callable_n209_unified_tail.py:infer" in source
    assert "999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75" in source
    assert "policy_callable_n214_unified_tail.py:infer" in source
    assert '[ValidateSet("N209", "N214", "N218", "N219", "N220", "N221", "N222", "N223", "N224", "N225", "N226", "N227", "N228")]' in source
    assert '$DeploymentLabel = "N222"' in source
    assert '$DeploymentLabel = "N223"' in source
    assert '$DeploymentLabel = "N224"' in source
    assert '$DeploymentLabel = "N225"' in source
    assert 'policy_callable_n225_gate3_vertical_guard.py:infer' in source
    assert 'c2f104c1264f31df1d22a167dcfb32d47c01b1479229925076ba780e057eb29e' in source
    n223 = source.split('if ($TailCandidate -eq "N223") {', 1)[1].split(
        '} elseif ($TailCandidate -eq "N222") {', 1
    )[0]
    assert '$BoundedGateReacquisition = $false' in n223
    assert 'policy_callable_n221_observable_course.py:infer' in source
    assert '$env:PUFFER_POLICY_BOUNDED_GATE_REACQUISITION = "1"' in source
    assert '$env:PUFFER_POLICY_GATE_ASSOCIATION_MAX_SPEED_M_S = "10.0"' in source
    assert '$env:PUFFER_POLICY_REACQUIRE_MAX_CANDIDATE_JUMP_M = "3.0"' in source
    assert '$env:PUFFER_POLICY_REACQUIRE_CLOSER_MARGIN_M = "0.5"' in source
    assert "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec" in source
    assert "policy_callable_n219_unified_tail.py:infer" in source
    assert "28cce6118eb9305b2526895c23258f92ad546f6d791da113572ee2668937b76a" in source
    assert "policy_callable_n220_observable_tail.py:infer" in source
    assert "dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746" in source
    assert '$env:PUFFER_POLICY_PREDICT_GATE_DROPOUT = "1"' in source
    assert '$env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR = "1"' in source
    assert '$env:PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX = "3"' in source
    assert 'if ($TailCandidate -in @("N220", "N223", "N224", "N225", "N226", "N227", "N228"))' in source
    assert '$env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.2"' in source
    assert "Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX" in source


def test_shadow_is_passive_and_replay_verified():
    source = SCRIPT.read_text(encoding="utf-8")
    shadow = source.split('if ($Mode -eq "Shadow") {', 1)[1].split(
        "$runArgs = @{", 1
    )[0]
    assert '"--policy-shadow-only"' in shadow
    assert '"--no-arm-on-start"' in shadow
    assert '"--policy-gate-phase-onehot-adapter-observation"' in shadow
    assert '"--policy-hybrid-prefix-confidence-observation"' in shadow
    assert '"--send-sim-reset"' not in shadow
    assert "verify_live_policy_shadow.py" in shadow


def test_full_lap_uses_one_checkpoint_for_the_whole_late_tail():
    source = SCRIPT.read_text(encoding="utf-8")
    full = source.split("$runArgs = @{", 1)[1]
    assert "PolicyCheckpoint = $TailCheckpoint" in full
    assert "Gate5Checkpoint = $TailCheckpoint" in full
    assert "Gate6Checkpoint = $TailCheckpoint" in full
    assert "HybridPrefixCheckpoint = $PrefixCheckpoint" in full
    assert "SixGateComposite = $true" in full
    assert "TargetGateCount = 6" in full
    assert "MinValidRuns = $MinValidRuns" in full
