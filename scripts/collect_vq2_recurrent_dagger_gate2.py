#!/usr/bin/env python3
"""Collect legal oracle labels on the SF034 Gate-2 failure distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_recurrent_dagger_next as collector
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.eval_vq2_recurrent_roll_priority import load_sf033


TAG = "vq2_sf035_recurrent_dagger_gate2_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf035_recurrent_dagger_gate2_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
SCREEN_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf034_recurrent_roll_priority_teacher_free_512/report.json"
)
SCREEN_REPORT_SHA256 = (
    "97ce3f49b18ee4d0e6fcb498f3d9098b5f0fbe03436cb26982488ae6c561ac7a"
)
AGENTS = 512
EPISODES = 512
SEED = 42035
COLLECTION_STEP_LIMIT = 2048


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
    collector.load_sf022 = load_sf033


def collect(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict:
    configure_collector_module()
    report = collector.collect(output=output, device_name=device_name)
    wrapper_path = Path(__file__).resolve()
    wrapper_key = str(wrapper_path.relative_to(ROOT))
    wrapper_sha256 = sha256_path(wrapper_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["action"]["plant_action_source"] = "sf033_recurrent_actor_mean"
    metadata["source_screen_report_sha256"] = SCREEN_REPORT_SHA256
    metadata["source_sha256"][wrapper_key] = wrapper_sha256
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["source_screen_report_sha256"] = SCREEN_REPORT_SHA256
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["collector_wrapper_sha256"] = wrapper_sha256
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

