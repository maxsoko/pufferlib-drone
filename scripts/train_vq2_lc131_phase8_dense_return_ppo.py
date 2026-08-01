#!/usr/bin/env python3
"""Teacher-free phase-8 PPO with training-only native dense returns."""

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
import scripts.train_vq2_lc127_phase6_onpolicy_ppo as base


BASE_LOAD_CONFIG = base.load_config
BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc131_phase8_dense_return_ppo_001"
SCHEMA = "vq2_lc131_phase8_dense_return_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc131_phase8_dense_return_ppo_checkpoint_v1"
TOTAL_AGENTS = 256
SEED_GROUP_SIZE = 128
ROLLOUTS = 4
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
PHASE_RETURN_ADVANTAGE_WEIGHT = 1.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
LC130_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc130_phase8_repeated_frontier_ppo_001/report.json"
)
LC130_REPORT_SHA256 = "da1a9b3c4d26a1dcd9d06b15e442f876028895504677736dba24ab296722c8d9"
PREREGISTRATION = ROOT / "docs/vq2_lc131_phase8_dense_return_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc131_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc131_phase8_dense_return_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC130_REPORT: LC130_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC131 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC130_REPORT.read_text())
    updates = rejected.get("updates", [])
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc130_phase8_repeated_frontier_ppo_report_v1"
        or not rejected.get("training_admitted")
        or rejected.get("candidate_selected_for_screen") is not None
        or len(rejected.get("rollouts", [])) != 3
        or any(item.get("raw9_or_later") != 0 for item in rejected["rollouts"])
        or len(updates) != 2
        or any(item.get("parameter_delta_l2") != 0.0 for item in updates)
        or any(item.get("queried_score_std") != 0.0 for item in updates)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC130 do not authorize LC131")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC130_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
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


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        frozen = corrected.pop("frozen_non_phase6_state_exact")
        corrected["frozen_non_target_phase_state_exact"] = frozen
        corrected["actor_execution"] = (
            "one complete 256-row Puffer actor over two repeated 128-seed groups"
        )
        corrected["native_seed_group_size"] = SEED_GROUP_SIZE
        updates = corrected.get("updates", [])
        nonzero_updates = bool(
            updates and all(item.get("parameter_delta_l2", 0.0) > 0.0 for item in updates)
        )
        return_variance = bool(
            updates and all(item.get("queried_phase_return_std", 0.0) > 0.0 for item in updates)
        )
        corrected["nonzero_updates"] = nonzero_updates
        corrected["nonzero_phase_return_variance"] = return_variance
        corrected["training_admitted"] = bool(
            corrected.get("training_admitted") and nonzero_updates and return_variance
        )
        corrected["next_authority"] = (
            "Run one reduced deterministic LC123-versus-selected-candidate raw-9 screen; no FlightSim authority."
            if corrected["training_admitted"]
            and corrected.get("candidate_selected_for_screen") is not None
            else "Reject LC131 and retain LC105 as the safe frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TOTAL_AGENTS, base.ROLLOUTS = TOTAL_AGENTS, ROLLOUTS
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.PHASE_RETURN_ADVANTAGE_WEIGHT = PHASE_RETURN_ADVANTAGE_WEIGHT
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
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once,
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
