#!/usr/bin/env python3
"""Phase-8 PPO at the source-measured oracle correction scale."""

from __future__ import annotations

import argparse
import json
import math
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

TAG = "vq2_lc133_phase8_anisotropic_ppo_001"
SCHEMA = "vq2_lc133_phase8_anisotropic_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc133_phase8_anisotropic_ppo_checkpoint_v1"
TOTAL_AGENTS = 512
ACTOR_BATCH_SIZE = 256
SEED_GROUP_SIZE = 128
ROLLOUTS = 4
PPO_EPOCHS = 2
LEARNING_RATE = 1e-5
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
EXPLORATION_STD = (0.05, 0.05, 0.05, 0.002)
PHASE_RETURN_ADVANTAGE_WEIGHT = 1.0
REWARD_OVERRIDES: dict[str, int | float] = {
    "w_progress": 5.0, "w_gate": 0.0, "w_ordered_gate": 30.0,
    "w_finish": 0.0, "w_time": 0.0, "w_ctrl": 0.0,
    "w_body_rate": 0.0, "w_cross_track": 0.0,
    "w_gate_camera_alignment": 0.0, "w_gate_crossing_error": 0.0,
    "w_gate_exit_lateral_velocity": 0.0, "w_gate_exit_velocity": 0.0,
    "w_forward_speed_excess": 0.0, "w_altitude_floor": 0.0,
    "w_descent_floor": 0.0, "w_action_teacher": 0.0,
    "invalid_penalty": 0.0, "late_invalid_penalty": 0.0,
}
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
LC125_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc125_lc123_phase8_9_rescue_features_001"
)
LC125_FEATURES = LC125_DIR / "features.bin"
LC125_FEATURES_SHA256 = "d505e3977e795aa81585bee69e1b9bad49d463f49c94fc9f2c7f07c2017444b9"
LC125_REPORT = LC125_DIR / "report.json"
LC125_REPORT_SHA256 = "0021aa7d0a91825dd24ce1fe36ef75234170e5cf0a8cc2eb850a4854888fe3e4"
LC132_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc132_phase8_progress_return_ppo_001/report.json"
)
LC132_REPORT_SHA256 = "8468dfafcdd13bb574f15f3176086b4b4d2fabb82da35d70fcd73da23edf9897"
PREREGISTRATION = ROOT / "docs/vq2_lc133_phase8_anisotropic_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc133_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc133_phase8_anisotropic_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def rescue_action_gap() -> dict[str, Any]:
    records = np.memmap(LC125_FEATURES, dtype=FEATURE_DTYPE, mode="r")
    selected = records[
        (records["agent_index"] == 175) & (records["phase_index"] == TARGET_PHASE)
    ]
    student = np.tanh(np.asarray(selected["base_pre_tanh"], dtype=np.float64))
    teacher = np.asarray(selected["teacher_action"], dtype=np.float64)
    difference = teacher - student
    return {
        "agent": 175,
        "records": int(selected.size),
        "action_rmse": float(np.sqrt(np.mean(difference * difference))),
        "per_channel_rms": np.sqrt(np.mean(difference * difference, axis=0)).tolist(),
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC125_FEATURES: LC125_FEATURES_SHA256,
        LC125_REPORT: LC125_REPORT_SHA256,
        LC132_REPORT: LC132_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC133 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rescue = json.loads(LC125_REPORT.read_text())
    rejected = json.loads(LC132_REPORT.read_text())
    gap = rescue_action_gap()
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rescue.get("schema")
        != "vq2_lc125_lc123_phase8_9_rescue_features_report_v1"
        or rescue.get("training_dataset_admitted")
        or rescue.get("query_outcome_success_agents") != [175]
        or gap["records"] != 1248
        or not math.isclose(gap["action_rmse"], 0.04401613728668076, abs_tol=1e-14)
        or rejected.get("schema")
        != "vq2_lc132_phase8_progress_return_ppo_report_v1"
        or rejected.get("candidate_selected_for_screen") is not None
        or len(rejected.get("updates", [])) != 5
        or any(item.get("raw9_or_later") != 0 for item in rejected.get("rollouts", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC125/LC132 do not authorize LC133")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC125_FEATURES, LC125_REPORT,
        LC132_REPORT, ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
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


def repeated_load_config(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["vec"]["env_seed_group_size"] = SEED_GROUP_SIZE
    config.setdefault("env", {}).update(REWARD_OVERRIDES)
    return config, [*overrides, "--vec.env-seed-group-size", str(SEED_GROUP_SIZE)]


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> Any:
    return expanded.SplitBatchActor((
        BASE_ACTOR_LOADER(payload, device), BASE_ACTOR_LOADER(payload, device),
    ))


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        frozen = corrected.pop("frozen_non_phase6_state_exact")
        corrected["frozen_non_target_phase_state_exact"] = frozen
        corrected["actor_execution"] = (
            "two independent complete 256-row Puffer actors over four repeated 128-seed groups"
        )
        corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
        corrected["native_seed_group_size"] = SEED_GROUP_SIZE
        corrected["training_reward_overrides"] = REWARD_OVERRIDES
        corrected["source_measured_rescue_action_gap"] = rescue_action_gap()
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
            else "Reject LC133 and retain LC105 as the safe frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS, base.PPO_EPOCHS, base.LEARNING_RATE,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.EXPLORATION_STD,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.load_config,
        base.write_json_once, base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TOTAL_AGENTS, base.ROLLOUTS = TOTAL_AGENTS, ROLLOUTS
    base.PPO_EPOCHS, base.LEARNING_RATE = PPO_EPOCHS, LEARNING_RATE
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.EXPLORATION_STD = EXPLORATION_STD
    base.PHASE_RETURN_ADVANTAGE_WEIGHT = PHASE_RETURN_ADVANTAGE_WEIGHT
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.load_config, base.write_json_once = repeated_load_config, corrected_writer
    base.milestone.load_actor = split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TOTAL_AGENTS, base.ROLLOUTS, base.PPO_EPOCHS, base.LEARNING_RATE,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.EXPLORATION_STD,
        base.PHASE_RETURN_ADVANTAGE_WEIGHT,
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
