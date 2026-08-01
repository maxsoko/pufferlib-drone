#!/usr/bin/env python3
"""Teacher-free phase-8 PPO on four repeated copies of LC123's frontier."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus as expanded
import scripts.train_vq2_lc127_phase6_onpolicy_ppo as base


BASE_LOAD_CONFIG = base.load_config
BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc130_phase8_repeated_frontier_ppo_001"
SCHEMA = "vq2_lc130_phase8_repeated_frontier_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc130_phase8_repeated_frontier_ppo_checkpoint_v1"
TOTAL_AGENTS = 512
ACTOR_BATCH_SIZE = 256
SEED_GROUP_SIZE = 128
ROLLOUTS = 3
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
LC129_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc129_phase8_9_expanded_rescue_corpus_001/report.json"
)
LC129_REPORT_SHA256 = "cced099979bc18bfd23cee8c5a01c08c5c1d73f8e908b36e5992a641fe1541af"
PREREGISTRATION = ROOT / "docs/vq2_lc130_phase8_repeated_frontier_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc130_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc130_phase8_repeated_frontier_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC129_REPORT: LC129_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC130 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC129_REPORT.read_text())
    items = rejected.get("expanded_child_items", [])
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc129_phase8_9_expanded_rescue_corpus_report_v1"
        or rejected.get("training_dataset_admitted")
        or rejected.get("new_feature_records") != 0
        or rejected.get("failed_admission_predicates") != [
            "new_seed_records_present",
            "new_seed_success_present",
            "combined_splittable_outcomes",
        ]
        or len(items) != 2
        or items[0].get("maximum_raw_index_distribution", {}).get("8") != 3
        or items[0].get("maximum_raw_index_distribution", {}).get("9") != 0
        or items[1].get("target_passes") != 1
        or items[1].get("paired_target_losses_vs_control") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC129 do not authorize LC130")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC129_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def repeated_load_config(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["vec"]["env_seed_group_size"] = SEED_GROUP_SIZE
    return config, [*overrides, "--vec.env-seed-group-size", str(SEED_GROUP_SIZE)]


def split_actor_loader(
    payload: dict[str, Any], device: torch.device
) -> expanded.SplitBatchActor:
    return expanded.SplitBatchActor((
        BASE_ACTOR_LOADER(payload, device),
        BASE_ACTOR_LOADER(payload, device),
    ))


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["actor_execution"] = (
            "two independent complete 256-row Puffer actors over four repeated 128-seed groups"
        )
        corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
        corrected["native_seed_group_size"] = SEED_GROUP_SIZE
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once, base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TOTAL_AGENTS, base.ROLLOUTS = TOTAL_AGENTS, ROLLOUTS
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.load_config = repeated_load_config
    base.write_json_once = corrected_writer
    base.milestone.load_actor = split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once, base.milestone.load_actor,
    ) = originals


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.train(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
    finally:
        restore(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
