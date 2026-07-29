import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "verify_gate1_promoted_adapter_traces",
    SCRIPTS / "verify_gate1_promoted_adapter_traces.py",
)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def _observation(
    *,
    forward: float,
    right: float,
    forward_rate: float,
    right_rate: float,
    legacy: bool,
) -> list[float]:
    size = 23 if legacy else 32
    values = np.zeros(size, dtype=np.float32)
    values[6] = 1.0
    values[10] = 1.0
    values[11] = np.tanh(forward / 10.0)
    values[12] = np.tanh(right / 5.0)
    values[0] = np.tanh(forward_rate / 5.0)
    values[1] = np.tanh(right_rate / 3.0)
    if not legacy:
        values[24] = 1.0
    return values.tolist()


def _sample(mode: str, *, legacy: bool) -> dict:
    if mode == "base":
        args = (25.0, 0.0, -3.0, 0.0)
        roll = 0.1
    elif mode == "counter":
        args = (5.0, -0.5, -5.0, 0.5)
        roll = 0.2
    elif mode == "adaptive":
        args = (20.0, -4.0, -6.0, 0.0)
        roll = 0.3
    elif mode == "close":
        args = (7.0, -2.0, -3.0, 0.0)
        roll = 0.4
    else:
        raise ValueError(mode)
    return {
        "elapsed_s": 1.0,
        "official_active_gate_index": 0,
        "observation": _observation(
            forward=args[0],
            right=args[1],
            forward_rate=args[2],
            right_rate=args[3],
            legacy=legacy,
        ),
        "normalized_action": [0.2, roll, -0.1, 0.3],
    }


def _write_report(path: Path, *, progress: int, accepted: bool, samples: list[dict]):
    path.write_text(
        json.dumps(
            {
                "acceptance_passed": accepted,
                "official_active_gate_index": progress,
                "policy_trace": {"samples": samples},
            }
        ),
        encoding="utf-8",
    )


def test_verify_separation_preserves_passes_and_exercises_failed_trace(tmp_path):
    accepted_legacy = tmp_path / "accepted_legacy.json"
    accepted_current = tmp_path / "accepted_current.json"
    failed = tmp_path / "failed.json"
    _write_report(
        accepted_legacy,
        progress=1,
        accepted=True,
        samples=[_sample("base", legacy=True), _sample("counter", legacy=True)],
    )
    _write_report(
        accepted_current,
        progress=2,
        accepted=False,
        samples=[_sample("base", legacy=False), _sample("counter", legacy=False)],
    )
    _write_report(
        failed,
        progress=0,
        accepted=False,
        samples=[_sample("adaptive", legacy=False), _sample("close", legacy=False)],
    )

    result = verify.verify_separation(
        [accepted_legacy, accepted_current],
        failed,
        expected_failed_changes=2,
    )

    assert result["passed"] is True
    assert result["accepted"]["gate1_samples"] == 4
    assert result["accepted"]["positive_projection_changes"] == 0
    assert result["accepted"]["full_adapter_counter_changes"] == 2
    assert result["failed"]["positive_projection_changes"] == 2
    assert result["failed"]["positive_non_roll_max_error"] == 0.0
    assert result["missing_trace_formats"] == []
