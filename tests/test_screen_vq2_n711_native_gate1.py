import subprocess
import sys
from pathlib import Path

from scripts.screen_vq2_n711_native_gate1 import _native_gates


ROOT = Path(__file__).resolve().parents[1]


def _metrics(passes: int) -> dict[str, object]:
    result: dict[str, object] = {
        "env_gates_passed": passes / 64,
        "env_n": 64.0,
        "total_episodes": 64,
        "episodes_per_agent": 1,
        "episode_offset": 600,
        "evaluation_start_contract": "uninterrupted_full_course",
        "gate_local_start_curriculum": 0,
        "mixed_start_curriculum": 0,
        "env_gate_local_reset_fraction": 0.0,
        "env_crash": 0.0,
        "env_timeout": 0.0,
        "action_count": 100,
        "action_0_mean": -0.02,
    }
    for channel in range(4):
        result[f"action_{channel}_abs_max"] = 0.1
    return result


def test_native_gates_require_paired_frontier_without_regression() -> None:
    baseline = _metrics(11)
    assert all(_native_gates(_metrics(11), baseline).values())
    gates = _native_gates(_metrics(10), baseline)
    assert not gates["at_least_11_gate1_passes"]
    assert not gates["does_not_regress_paired_baseline"]


def test_direct_cli_has_no_flightsim_or_training_surface() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/screen_vq2_n711_native_gate1.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--n712-report" in result.stdout
    assert "--baseline" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "14550" not in result.stdout
    assert "5600" not in result.stdout
    assert "31000" not in result.stdout
