#!/usr/bin/env python3
"""Collect oracle labels on SF063's exact safe-under-turn distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_measured_two_gate_dagger as iteration
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf065_underturn_dagger_64"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = ROOT / "docs/vq2_sf065_underturn_dagger_preregistration_2026-07-28.md"
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf063_current_dagger_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "c27c28bd26e937fc8023ceca6f8880fa6f3d4320bf6db55ce9837518b1868ebb"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "fa738aee5d5c79080d0e7ada1c69d8cce074ef5ec9ed03d8a5684a0c0357db2e"
)
SCREEN_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf064_measured_two_gate_teacher_free_512/report.json"
)
SCREEN_REPORT_SHA256 = (
    "1f77ef61f929a1bf3bad5a93a638fbbc719939e59f3efee4205399a48fc2ce9e"
)
SEED = 42065
COLLECTION_STEP_LIMIT = 1600


def failure_state_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
) -> bool:
    return bool(
        metrics.get("env/n", 0.0) == float(iteration.EPISODES)
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
        and lengths.shape == (iteration.AGENTS,)
        and np.all((lengths > 0) & (lengths <= COLLECTION_STEP_LIMIT))
        and np.all(terminal_count == 1)
        and np.all(terminal_is_last)
        and labels == int(lengths.sum())
        and executed_action_max_error <= 1e-7
    )


def configure_iteration() -> None:
    iteration.TAG = TAG
    iteration.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    iteration.PREREGISTRATION = PREREGISTRATION
    iteration.CHECKPOINT = CHECKPOINT
    iteration.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    iteration.TRAIN_REPORT = TRAIN_REPORT
    iteration.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    iteration.SF051_REPORT = SCREEN_REPORT
    iteration.SF051_REPORT_SHA256 = SCREEN_REPORT_SHA256
    iteration.SEED = SEED
    iteration.COLLECTION_STEP_LIMIT = COLLECTION_STEP_LIMIT
    iteration.exact_collection_passes = failure_state_collection_passes


def collect(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict:
    configure_iteration()
    report = iteration.collect(output=output, device_name=device_name)
    wrapper = Path(__file__).resolve()
    key = str(wrapper.relative_to(ROOT))
    digest = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][key] = digest
    metadata["action"]["plant_action_source"] = "sf063_phase_actor_mean"
    metadata["genuine_failure_terminals_retained"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["source_sha256"][key] = digest
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["parent_screen_report_sha256"] = SCREEN_REPORT_SHA256
    report["genuine_failure_terminals_retained"] = True
    report["terminal_categories"] = {
        "crash_rate": report["metrics"].get("env/crash", 0.0),
        "missed_gate_rate": report["metrics"].get("env/missed_gate", 0.0),
        "out_of_order_rate": report["metrics"].get("env/out_of_order", 0.0),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(collect(args.output.resolve(), device_name=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
