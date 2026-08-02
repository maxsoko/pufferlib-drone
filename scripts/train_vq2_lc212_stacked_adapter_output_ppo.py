#!/usr/bin/env python3
"""Teacher-free closed-loop PPO of LC209's continuation output head."""

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

from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import (
    load_actor as shared_load_actor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus as expanded
import scripts.train_vq2_lc127_phase6_onpolicy_ppo as base


BASE_LOAD_CONFIG = base.load_config
BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc212_stacked_adapter_output_ppo_001"
SCHEMA = "vq2_lc212_stacked_adapter_output_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc212_stacked_adapter_output_ppo_checkpoint_v1"
TOTAL_AGENTS = 512
ACTOR_BATCH_SIZE = 256
FEATURE_SIZE = 64
SEED = 432_190
SEED_GROUP_SIZE = 1
SEED_INDEX_OFFSET = 15
MAX_STEPS = 33_000
ROLLOUTS = 6
PPO_EPOCHS = 2
BATCH_SIZE = 8_192
LEARNING_RATE = 3e-5
TARGET_PHASE = 16
TARGET_RAW_INDEX = 18
EXPLORATION_STD = (0.25, 0.75, 0.25, 0.002)
PHASE_RETURN_ADVANTAGE_WEIGHT = 1.0
MAXIMUM_UPDATE_DELTA_L2 = 0.75
OUTPUT_PARAMETER_NAMES = (
    "continuation_adapter_output.weight",
    "continuation_adapter_output.bias",
)
ROLLOUT_DTYPE = np.dtype([
    ("hidden", "<f2", (FEATURE_SIZE,)),
    ("base_pre_tanh", "<f4", (4,)),
    ("sample_pre_tanh", "<f4", (4,)),
    ("old_log_probability", "<f4"),
    ("agent_index", "<u2"),
])
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
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "50654f1f1e16e919fd74c35e5841f36986d759d36ff101eaf70ddc5843fcdea4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ff8801c14a3e4620a056723bf9ca4566d61b059a4c8e7262d5a1accc666fd41b"
TRUSTED_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
TRUSTED_CHECKPOINT = TRUSTED_DIR / "policy_selected.pt"
TRUSTED_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
LC210_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc210_stacked_adapter_onpolicy2_milestone_001/report.json"
LC210_REPORT_SHA256 = "3096f9bac8406bb8432118616fc2cc24b160015b0851713d8036a78829c4b270"
LC211_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc211_lc209_phase16_cem_001/report.json"
LC211_REPORT_SHA256 = "588996548612c9317d5ba5ed5a69aa70a8312d4c5bfc6b6287818cf8ec21ef0b"
PREREGISTRATION = ROOT / "docs/vq2_lc212_stacked_adapter_output_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc212_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc212_stacked_adapter_output_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        TRUSTED_CHECKPOINT: TRUSTED_CHECKPOINT_SHA256,
        LC210_REPORT: LC210_REPORT_SHA256,
        LC211_REPORT: LC211_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC212 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    trusted = torch.load(TRUSTED_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC210_REPORT.read_text())
    exhausted = json.loads(LC211_REPORT.read_text())
    baseline, candidate = rejected.get("items", [{}, {}])
    generations = exhausted.get("generations", [])
    if (
        parent.get("schema")
        != "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("continuation_phase_min") != 16
        or parent.get("model", {}).get("continuation_phase_max_exclusive") != 18
        or parent.get("model", {}).get("continuation_adapter_size") != FEATURE_SIZE
        or parent_report.get("schema")
        != "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or not parent_report.get("frozen_lc189_state_exact")
        or rejected.get("schema")
        != "vq2_lc210_stacked_adapter_onpolicy2_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or candidate.get("maximum_raw_index_distribution", {}).get("16") != 128
        or exhausted.get("schema") != "vq2_lc211_lc209_phase16_cem_report_v1"
        or exhausted.get("training_admitted")
        or len(generations) != 3
        or any(item.get("target_passes") for item in generations)
        or max(item.get("maximum_raw_index_distribution", {}).get("17", 0) for item in generations) != 1
        or any(
            report.get("safety", {}).get("flight_sim_packets_sent") != 0
            or report.get("safety", {}).get("submission_authorized")
            for report in (parent_report, rejected, exhausted)
        )
    ):
        raise RuntimeError("LC209/LC210/LC211 do not authorize LC212")
    for name, value in trusted["model_state"].items():
        if not torch.equal(value, parent["model_state"][name]):
            raise RuntimeError(f"LC209 changed trusted LC189 tensor {name}")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, TRUSTED_CHECKPOINT,
        LC210_REPORT, LC211_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
        ROOT / "scripts/train_vq2_lc209_stacked_adapter_onpolicy_dagger2.py",
        ROOT / "scripts/train_vq2_lc211_lc209_phase16_cem.py",
        ROOT / "scripts/eval_vq2_lc178_phase15_recurrent_adapter_milestone.py",
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
        shared_load_actor(payload, device), shared_load_actor(payload, device),
    ))


def output_feature_components(
    actor: Any,
    next_recurrent: torch.Tensor,
    current_mean: torch.Tensor,
    selected_index: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    del actor
    feature = next_recurrent[0, :, -FEATURE_SIZE:].index_select(0, selected_index)
    weight = state[OUTPUT_PARAMETER_NAMES[0]].to(feature.device)
    bias = state[OUTPUT_PARAMETER_NAMES[1]].to(feature.device)
    continuation = feature @ weight.T + bias
    return feature, current_mean - continuation


def frozen_output_scope_exact(
    state: dict[str, torch.Tensor], parent_state: dict[str, torch.Tensor]
) -> bool:
    return all(
        name in OUTPUT_PARAMETER_NAMES or torch.equal(value, parent_state[name])
        for name, value in state.items()
    )


def ppo_output_update(
    records: np.ndarray,
    maximum_raw_index: np.ndarray,
    passed: np.ndarray,
    phase_return: np.ndarray,
    state: dict[str, torch.Tensor],
    *, iteration: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if records.size == 0:
        raise RuntimeError("LC212 rollout reached no phase-16 states")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    advantage_by_agent, score_metrics = base.trajectory_advantages(
        maximum_raw_index, passed, agents, phase_return
    )
    counts = np.bincount(agents, minlength=TOTAL_AGENTS)
    row_weights_np = 1.0 / counts[agents].astype(np.float64)
    row_weights_np /= row_weights_np.sum()
    feature = torch.from_numpy(
        np.array(records["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    baseline = torch.from_numpy(
        np.array(records["base_pre_tanh"], dtype=np.float32, copy=True)
    ).to(device)
    sampled = torch.from_numpy(
        np.array(records["sample_pre_tanh"], dtype=np.float32, copy=True)
    ).to(device)
    old_log_probability = torch.from_numpy(
        np.array(records["old_log_probability"], dtype=np.float32, copy=True)
    ).to(device)
    advantages = torch.from_numpy(advantage_by_agent[agents]).to(device)
    row_weights = torch.from_numpy(row_weights_np.astype(np.float32)).to(device)
    parents = tuple(state[name].to(device).detach().clone() for name in OUTPUT_PARAMETER_NAMES)
    parameters = tuple(value.clone().requires_grad_(True) for value in parents)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 20_000 * iteration)
    updates = 0
    maximum_approximate_kl = 0.0
    maximum_clip_fraction = 0.0
    losses: list[float] = []
    for _ in range(PPO_EPOCHS):
        permutation = torch.randperm(records.size, generator=generator)
        for start in range(0, records.size, BATCH_SIZE):
            index = permutation[start : start + BATCH_SIZE].to(device)
            new_mean = (
                baseline.index_select(0, index)
                + feature.index_select(0, index) @ parameters[0].T
                + parameters[1]
            )
            new_log_probability = base.normal_log_probability(
                sampled.index_select(0, index), new_mean, EXPLORATION_STD
            )
            old_log = old_log_probability.index_select(0, index)
            log_ratio = new_log_probability - old_log
            ratio = torch.exp(log_ratio.clamp(-8.0, 8.0))
            advantage = advantages.index_select(0, index)
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - base.CLIP_COEFFICIENT,
                1.0 + base.CLIP_COEFFICIENT,
            ) * advantage
            weight = row_weights.index_select(0, index)
            policy_loss = -(
                weight * torch.minimum(unclipped, clipped)
            ).sum() / weight.sum().clamp_min(1e-12)
            anchor = sum(
                (value - parent).square().mean()
                for value, parent in zip(parameters, parents)
            )
            loss = policy_loss + base.ANCHOR_COEFFICIENT * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRADIENT_NORM)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = float(((ratio - 1.0) - log_ratio).mean())
                clip_fraction = float(
                    ((ratio - 1.0).abs() > base.CLIP_COEFFICIENT).float().mean()
                )
            maximum_approximate_kl = max(maximum_approximate_kl, approximate_kl)
            maximum_clip_fraction = max(maximum_clip_fraction, clip_fraction)
            losses.append(float(policy_loss.detach()))
            updates += 1
    delta_l2 = float(torch.sqrt(sum(
        (value - parent).square().sum()
        for value, parent in zip(parameters, parents)
    )).detach())
    finite = bool(
        all(torch.isfinite(value).all() for value in parameters)
        and math.isfinite(delta_l2)
        and all(math.isfinite(value) for value in losses)
    )
    admitted = finite and 0.0 < delta_l2 <= MAXIMUM_UPDATE_DELTA_L2
    next_state = {name: value.detach().cpu().clone() for name, value in state.items()}
    if admitted:
        for name, value in zip(OUTPUT_PARAMETER_NAMES, parameters):
            next_state[name].copy_(value.detach().cpu().float())
    return next_state, {
        "iteration": iteration,
        "optimizer_updates": updates,
        "policy_loss_mean": float(np.mean(losses)),
        "policy_loss_last": losses[-1],
        "parameter_delta_l2": delta_l2,
        "maximum_approximate_kl": maximum_approximate_kl,
        "maximum_clip_fraction": maximum_clip_fraction,
        "finite": finite,
        "update_admitted": admitted,
        "updated_parameters": list(OUTPUT_PARAMETER_NAMES),
        **score_metrics,
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        frozen = corrected.pop("frozen_non_phase6_state_exact")
        corrected["frozen_non_continuation_output_state_exact"] = frozen
        corrected["frozen_trusted_lc189_state_exact"] = True
        corrected["actor_execution"] = (
            "two independent complete 256-row LC209 stacked-adapter Puffers"
        )
        corrected["optimization"] = (
            "teacher-free closed-loop PPO of only the 64-state continuation "
            "adapter output weight and bias"
        )
        corrected["training_reward_overrides"] = REWARD_OVERRIDES
        corrected["next_authority"] = (
            "Run one exact-context deterministic LC189-versus-selected raw-18 screen; no FlightSim authority."
            if corrected.get("training_admitted")
            and corrected.get("candidate_selected_for_screen") is not None
            else "Reject LC212 and retain LC189; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "TOTAL_AGENTS", "SEED",
        "MAX_STEPS", "ROLLOUTS", "PPO_EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
        "TARGET_PHASE", "TARGET_RAW_INDEX", "EXPLORATION_STD",
        "PHASE_RETURN_ADVANTAGE_WEIGHT", "MAXIMUM_UPDATE_DELTA_L2",
        "PARENT_CHECKPOINT", "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT",
        "PARENT_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "ROLLOUT_DTYPE", "verify_inputs", "source_identity",
        "load_config", "write_json_once", "ppo_update",
        "rollout_feature_components", "frozen_update_scope_exact",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, TOTAL_AGENTS, SEED, MAX_STEPS,
        ROLLOUTS, PPO_EPOCHS, BATCH_SIZE, LEARNING_RATE, TARGET_PHASE,
        TARGET_RAW_INDEX, EXPLORATION_STD, PHASE_RETURN_ADVANTAGE_WEIGHT,
        MAXIMUM_UPDATE_DELTA_L2, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, ROLLOUT_DTYPE, verify_inputs, source_identity,
        training_load_config, corrected_writer, ppo_output_update,
        output_feature_components, frozen_output_scope_exact,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    actor_original = base.milestone.load_actor
    base.milestone.load_actor = split_actor_loader
    return (names, originals), (("actor_loader",), (actor_original,))


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    (names, originals), (_, actor_originals) = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)
    base.milestone.load_actor = actor_originals[0]


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        return base.train(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("training_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
