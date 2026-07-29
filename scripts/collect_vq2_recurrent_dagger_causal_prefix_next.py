#!/usr/bin/env python3
"""Collect a fixed causal DAgger prefix from the SF028 policy distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_recurrent_dagger_next as collector
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.eval_vq2_recurrent_dagger_causal_prefix import load_sf028


TAG = "vq2_sf030_recurrent_dagger_causal_prefix_next_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf030_recurrent_dagger_causal_prefix_next_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf028_recurrent_dagger_causal_prefix_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "b74e2f1c792fbd0e250a06a9c65b4fdaa3cc4f132e5e6bbab0a4c07a240b0403"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "c1ebd267ca0d35ed873287da1378720c316eb8957115465c5813bf284a4c7ccd"
)
SCREEN_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf029_recurrent_dagger_causal_prefix_teacher_free_512/report.json"
)
SCREEN_REPORT_SHA256 = (
    "b95b671c24ddb64ee688d6686f0d28421709bb282dd62bc03135986484d2adc7"
)
AGENTS = 512
EPISODES = 512
SEED = 42030
COLLECTION_STEP_LIMIT = 512


def prefix_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
) -> bool:
    completed = int(terminal_count.sum())
    survivors = terminal_count == 0
    return (
        metrics.get("env/n", 0.0) == float(completed)
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
        and lengths.shape == (AGENTS,)
        and bool(np.all(lengths > 0))
        and bool(np.all(lengths <= COLLECTION_STEP_LIMIT))
        and bool(np.all(terminal_count <= 1))
        and bool(np.all(terminal_is_last[terminal_count == 1]))
        and bool(np.all(~terminal_is_last[survivors]))
        and bool(np.all(lengths[survivors] == COLLECTION_STEP_LIMIT))
        and labels == int(lengths.sum())
        and executed_action_max_error <= 1e-7
    )


def configure_collector_module() -> None:
    collector.TAG = TAG
    collector.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    collector.PREREGISTRATION = PREREGISTRATION
    collector.CHECKPOINT = CHECKPOINT
    collector.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    collector.TRAIN_REPORT = TRAIN_REPORT
    collector.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    collector.SF023_REPORT = SCREEN_REPORT
    collector.SF023_REPORT_SHA256 = SCREEN_REPORT_SHA256
    collector.AGENTS = AGENTS
    collector.EPISODES = EPISODES
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = COLLECTION_STEP_LIMIT
    collector.load_sf022 = load_sf028
    collector.next_dagger_collection_passes = prefix_collection_passes


def collect(
    output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda"
) -> dict:
    configure_collector_module()
    report = collector.collect(output=output, device_name=device_name)

    terminal = np.load(output / "terminal.npy", mmap_mode="r")
    completed = int(terminal.sum())
    survivors = AGENTS - completed
    wrapper_key = str(Path(__file__).resolve().relative_to(ROOT))
    wrapper_sha256 = sha256_path(Path(__file__).resolve())
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["action"]["plant_action_source"] = "sf028_recurrent_actor_mean"
    metadata["prefix_contract"] = {
        "fixed_horizon_steps": COLLECTION_STEP_LIMIT,
        "completed_terminals": completed,
        "horizon_survivors": survivors,
    }
    metadata["source_sha256"][wrapper_key] = wrapper_sha256
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["completed_terminals"] = completed
    report["horizon_survivors"] = survivors
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["collector_wrapper_sha256"] = wrapper_sha256
    report["safety"]["native_crashes_retained_as_terminal_training_states"] = int(
        round(report["metrics"].get("env/crash", 0.0) * completed)
    )
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
