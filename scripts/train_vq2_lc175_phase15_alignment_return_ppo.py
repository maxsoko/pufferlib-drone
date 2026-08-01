#!/usr/bin/env python3
"""Teacher-free phase-15 PPO with training-only alignment returns."""

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

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus as expanded
import scripts.train_vq2_lc127_phase6_onpolicy_ppo as base


BASE_LOAD_CONFIG = base.load_config
BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc175_phase15_alignment_return_ppo_001"
SCHEMA = "vq2_lc175_phase15_alignment_return_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc175_phase15_alignment_return_ppo_checkpoint_v1"
TOTAL_AGENTS = 512
ACTOR_BATCH_SIZE = 256
SEED = 432_190
SEED_GROUP_SIZE = 1
SEED_INDEX_OFFSET = 15
MAX_STEPS = 27_000
ROLLOUTS = 3
PPO_EPOCHS = 2
LEARNING_RATE = 2e-5
TARGET_PHASE = 15
TARGET_RAW_INDEX = 16
EXPLORATION_STD = (0.25, 0.75, 0.25, 0.002)
PHASE_RETURN_ADVANTAGE_WEIGHT = 1.0
MAXIMUM_UPDATE_DELTA_L2 = 4.0
REWARD_OVERRIDES: dict[str, int | float] = {
    "w_progress": 5.0,
    "w_ordered_gate": 30.0,
    "w_finish": 0.0,
    "w_time": 0.0,
    "w_ctrl": 0.001,
    "w_body_rate": 0.1,
    "w_cross_track": 0.0,
    "w_gate_camera_alignment": 20.0,
    "gate_camera_alignment_from_gate_index": TARGET_PHASE,
    "w_gate_crossing_error": 10.0,
    "invalid_penalty": 150.0,
    "late_invalid_penalty": 150.0,
    "late_invalid_penalty_from_gate_index": TARGET_PHASE,
}
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc173_phase15_failure_state_dagger3_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "e3c65e89ce928bff59ffe05a7b01f2f9abde7467d728c791944b33d7f3924958"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a48a4514e2c0240cb0f375357d6299a6cdeb4bb9162e8223862017d2f3a1ec7d"
LC172_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc172_phase15_failure_state_dagger3_001"
LC172_FEATURES = LC172_DIR / "features.bin"
LC172_FEATURES_SHA256 = "ddb15553a498e325edf14078dab6a1ca596750604aca9b84e82d497295f13d32"
LC172_REPORT = LC172_DIR / "report.json"
LC172_REPORT_SHA256 = "6e0f176d7e0ff328e1cfe0f898525c5d93d6da1fa219d4905065bc90b1c3985c"
LC174_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc174_phase15_failure_state_dagger3_milestone_001/report.json"
LC174_REPORT_SHA256 = "451f838803201494232499018d83e9b901ee935f69be5f03d0e85773072f25b6"
PREREGISTRATION = ROOT / "docs/vq2_lc175_phase15_alignment_return_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc175_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc175_phase15_alignment_return_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def source_measured_action_gap() -> dict[str, Any]:
    records = np.memmap(LC172_FEATURES, dtype=FEATURE_DTYPE, mode="r")
    selected = records[records["agent_index"] < 256]
    student = np.tanh(np.asarray(selected["base_pre_tanh"], dtype=np.float64))
    teacher = np.asarray(selected["teacher_action"], dtype=np.float64)
    difference = teacher - student
    return {
        "records": int(selected.size),
        "action_rmse": float(np.sqrt(np.mean(difference * difference))),
        "per_channel_rms": np.sqrt(np.mean(difference * difference, axis=0)).tolist(),
        "per_channel_mean": difference.mean(axis=0).tolist(),
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC172_FEATURES: LC172_FEATURES_SHA256,
        LC172_REPORT: LC172_REPORT_SHA256,
        LC174_REPORT: LC174_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC175 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(LC172_REPORT.read_text())
    rejected = json.loads(LC174_REPORT.read_text())
    items = rejected.get("items", [])
    gap = source_measured_action_gap()
    if (
        parent.get("schema") != "vq2_lc173_phase15_failure_state_dagger3_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc173_phase15_failure_state_dagger3_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc172_phase15_failure_state_dagger3_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 375_552
        or gap["records"] != 123_136
        or not np.isfinite(gap["per_channel_rms"]).all()
        or rejected.get("schema") != "vq2_lc174_phase15_failure_state_dagger3_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or len(items) != 2
        or any(item.get("target_passes") != 0 for item in items)
        or any(item.get("pre_target_terminals") != 128 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC172/LC173/LC174 do not authorize LC175")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC172_FEATURES, LC172_REPORT, LC174_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
        ROOT / "scripts/collect_vq2_lc172_phase15_failure_state_dagger3.py",
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c", ROOT / "ocean/drone_race/drone_race.h",
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


def training_load_config(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["vec"]["env_seed_group_size"] = SEED_GROUP_SIZE
    config["vec"]["env_seed_index_offset"] = SEED_INDEX_OFFSET
    config.setdefault("env", {}).update(REWARD_OVERRIDES)
    return config, [
        *overrides,
        "--vec.env-seed-group-size", str(SEED_GROUP_SIZE),
        "--vec.env-seed-index-offset", str(SEED_INDEX_OFFSET),
    ]


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> Any:
    return expanded.SplitBatchActor((
        BASE_ACTOR_LOADER(payload, device), BASE_ACTOR_LOADER(payload, device),
    ))


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        frozen = corrected.pop("frozen_non_phase6_state_exact")
        corrected["frozen_non_target_phase_state_exact"] = frozen
        corrected["actor_execution"] = "two independent complete 256-row Puffer actors over 512 exact seed-15 trajectories"
        corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
        corrected["native_seed_group_size"] = SEED_GROUP_SIZE
        corrected["native_seed_index_offset"] = SEED_INDEX_OFFSET
        corrected["training_reward_overrides"] = REWARD_OVERRIDES
        corrected["source_measured_action_gap"] = source_measured_action_gap()
        updates = corrected.get("updates", [])
        nonzero_updates = bool(updates and all(item.get("parameter_delta_l2", 0.0) > 0.0 for item in updates))
        return_variance = bool(updates and all(item.get("queried_phase_return_std", 0.0) > 0.0 for item in updates))
        corrected["nonzero_updates"] = nonzero_updates
        corrected["nonzero_phase_return_variance"] = return_variance
        corrected["training_admitted"] = bool(corrected.get("training_admitted") and nonzero_updates and return_variance)
        corrected["next_authority"] = (
            "Run one teacher-free deterministic LC173-versus-selected-candidate raw-16 screen; no FlightSim authority."
            if corrected["training_admitted"] and corrected.get("candidate_selected_for_screen") is not None
            else "Reject LC175 and implement a phase-local recurrent adapter; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.SEED, base.MAX_STEPS, base.ROLLOUTS,
        base.PPO_EPOCHS, base.LEARNING_RATE,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.EXPLORATION_STD,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT, base.MAXIMUM_UPDATE_DELTA_L2,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once, base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TOTAL_AGENTS, base.SEED, base.MAX_STEPS, base.ROLLOUTS = TOTAL_AGENTS, SEED, MAX_STEPS, ROLLOUTS
    base.PPO_EPOCHS, base.LEARNING_RATE = PPO_EPOCHS, LEARNING_RATE
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.EXPLORATION_STD = EXPLORATION_STD
    base.PHASE_RETURN_ADVANTAGE_WEIGHT = PHASE_RETURN_ADVANTAGE_WEIGHT
    base.MAXIMUM_UPDATE_DELTA_L2 = MAXIMUM_UPDATE_DELTA_L2
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.load_config, base.write_json_once = training_load_config, corrected_writer
    base.milestone.load_actor = split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.SEED, base.MAX_STEPS, base.ROLLOUTS,
        base.PPO_EPOCHS, base.LEARNING_RATE,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.EXPLORATION_STD,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT, base.MAXIMUM_UPDATE_DELTA_L2,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once, base.milestone.load_actor,
    ) = originals


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
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
