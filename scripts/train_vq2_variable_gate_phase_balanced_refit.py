#!/usr/bin/env python3
"""Refit VG033 with causal VG039 sequences and phase-balanced row losses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.train_vq2_variable_gate_recurrent_bc import (
    sha256_path,
    source_label,
)
import scripts.train_vq2_variable_gate_nine_source_refit as vg040
import scripts.train_vq2_variable_gate_seven_source_refit as core


TAG = "vq2_vg042_variable_gate_phase_balanced_refit_001"
REPORT_SCHEMA = "vq2_variable_gate_phase_balanced_refit_report_v1"
STATE_SCHEMA = "vq2_variable_gate_phase_balanced_refit_state_v1"
SEED = 429156

PREREGISTRATION = (
    ROOT / "docs/vq2_vg042_phase_balanced_refit_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg042_vast.sh"
REJECTION = (
    ROOT
    / "docs/vq2_vg041_paired_count5_diagnostic_rejection_2026-07-31.json"
)
REJECTION_SHA256 = (
    "a9eb3079c7e59cf52594ce89c07182356724d99a0ab545c7e96eb9f954a55dd9"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)

# Preserve VG033's 0.65 anchor mass. Split its former 0.35 latest-source mass
# between VG032 and the phase-balanced VG039 source.
SOURCE_WEIGHTS = {
    "clean": 0.25,
    "dagger1": 0.03,
    "dagger2": 0.03,
    "dagger3": 0.06,
    "dagger4": 0.08,
    "dagger5": 0.08,
    "dagger6": 0.12,
    "dagger7": 0.15,
    "dagger8": 0.20,
}

# Rows stay in their original causal recurrent sequences. Only their loss
# contribution changes. The validation split has no phase-5 rows, so selection
# is source-locked on phases 0--4 while the 240 training phase-5 rows remain
# bounded at the same weight as phase 4.
PHASE_ROW_WEIGHTS = (
    0.25,
    0.50,
    1.00,
    6.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
    32.00,
)


@dataclass(frozen=True)
class PhaseBalancedConfig(vg040.NineSourceConfig):
    seed: int = SEED
    epochs: int = 4
    learning_rate: float = 2e-6
    clean_objective_weight: float = 0.25
    dagger1_objective_weight: float = 0.03
    dagger2_objective_weight: float = 0.03
    dagger3_objective_weight: float = 0.06
    dagger4_objective_weight: float = 0.08
    dagger5_objective_weight: float = 0.08
    dagger6_objective_weight: float = 0.12
    dagger7_objective_weight: float = 0.15
    dagger8_objective_weight: float = 0.20
    dagger8_phase_row_weights: tuple[float, ...] = PHASE_ROW_WEIGHTS


def source_paths() -> list[Path]:
    return [
        *vg040.source_paths(),
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        REJECTION,
    ]


def current_source_identity() -> tuple[str, dict[str, str]]:
    hashes = {source_label(path): sha256_path(path) for path in source_paths()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, hashes


def verify_inputs() -> None:
    vg040.verify_inputs()
    if sha256_path(REJECTION) != REJECTION_SHA256:
        raise RuntimeError("VG042 rejection evidence hash mismatch")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG042 preregistration is missing")
    rejection = json.loads(REJECTION.read_text())
    if (
        rejection.get("schema")
        != "vq2_vg041_paired_count5_diagnostic_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("qualified_for_count11_diagnostic")
        or rejection.get("candidate", {}).get("actor") != "VG040"
        or rejection.get("parent", {}).get("actor") != "VG033"
        or rejection.get("criteria", {}).get("crash_count_not_regressed")
        or rejection.get("criteria", {}).get("downstream_improves_parent")
    ):
        raise RuntimeError("VG041 rejection does not authorize VG042")


def phase_row_weights(
    observation: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if (
        observation.ndim != 3
        or observation.shape[:-1] != valid.shape
        or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE
    ):
        raise ValueError("phase-balanced rows do not align with actor input")
    scaled = observation[..., -1] * 16.0
    rounded = torch.round(scaled)
    if valid.any() and torch.max(torch.abs(scaled[valid] - rounded[valid])) > 1e-5:
        raise RuntimeError("phase-balanced source contains a non-/16 phase")
    phase = rounded.to(torch.long).clamp_(0, 16)
    lookup = observation.new_tensor(PHASE_ROW_WEIGHTS)
    return lookup[phase]


def evaluate_phase_balanced(
    model: core.VQ2PhaseRecurrentActor,
    dataset: core.VariableGateBCDataset,
    agents: np.ndarray,
    config: PhaseBalancedConfig,
    device: torch.device,
) -> dict[str, Any]:
    result = core.evaluate(model, dataset, agents, config, device)
    action_weights = torch.tensor(
        config.action_weights, dtype=torch.float64, device=device
    )
    lookup = torch.tensor(PHASE_ROW_WEIGHTS, dtype=torch.float64, device=device)
    weighted_error = torch.zeros(4, dtype=torch.float64, device=device)
    weighted_count = torch.zeros((), dtype=torch.float64, device=device)
    phase_error = torch.zeros((17, 4), dtype=torch.float64, device=device)
    phase_count = torch.zeros(17, dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        for batch_agents in core._agent_batches(agents, config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, _ = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output, state = model.forward_sequence(observation, state)
                scaled = observation[..., -1] * 16.0
                rounded = torch.round(scaled)
                if valid.any() and (
                    torch.max(torch.abs(scaled[valid] - rounded[valid])) > 1e-5
                ):
                    raise RuntimeError(
                        "phase-balanced validation found a non-/16 phase"
                    )
                phase = rounded.to(torch.long).clamp_(0, 16)
                error = (output.mean.float() - target.float()).square().double()
                valid_double = valid.to(torch.float64)
                effective = lookup[phase] * valid_double
                weighted_error += (error * effective.unsqueeze(-1)).sum((0, 1))
                weighted_count += effective.sum()
                for index in range(17):
                    mask = valid_double * (phase == index).to(torch.float64)
                    count = mask.sum()
                    if count:
                        phase_error[index] += (
                            error * mask.unsqueeze(-1)
                        ).sum((0, 1))
                        phase_count[index] += count

    channel_mse = weighted_error / weighted_count.clamp_min(1.0)
    phase_balanced_mse = (
        (channel_mse * action_weights).sum() / action_weights.sum()
    )
    per_phase: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index in range(17):
        count = int(phase_count[index].item())
        if count:
            mse = phase_error[index] / phase_count[index]
            value = (mse * action_weights).sum() / action_weights.sum()
            per_phase[str(index)] = float(value.item())
            counts[str(index)] = count
    result["unweighted_weighted_mse"] = float(result["weighted_mse"])
    result["weighted_mse"] = float(phase_balanced_mse.item())
    result["phase_balanced_weighted_mse"] = float(
        phase_balanced_mse.item()
    )
    result["phase_row_weights"] = list(PHASE_ROW_WEIGHTS)
    result["phase_counts"] = counts
    result["per_phase_weighted_mse"] = per_phase
    return result


def numerical_admission_predicate(
    *,
    best_epoch: int,
    best_score: float,
    baseline_validation: dict[str, Any],
    selected_validation: dict[str, Any],
    source_balance_audit: bool,
    config: PhaseBalancedConfig,
) -> bool:
    caps = {
        "clean": config.maximum_clean_validation_weighted_mse,
        "dagger1": config.maximum_dagger1_validation_weighted_mse,
        "dagger2": config.maximum_dagger2_validation_weighted_mse,
        "dagger3": config.maximum_dagger3_validation_weighted_mse,
        "dagger4": config.maximum_dagger4_validation_weighted_mse,
        "dagger5": config.maximum_dagger5_validation_weighted_mse,
        "dagger6": config.maximum_dagger6_validation_weighted_mse,
        "dagger7": config.maximum_dagger7_validation_weighted_mse,
        "dagger8": config.maximum_dagger8_validation_weighted_mse,
    }
    baseline_phase = baseline_validation["dagger8"][
        "per_phase_weighted_mse"
    ]
    selected_phase = selected_validation["dagger8"][
        "per_phase_weighted_mse"
    ]
    return bool(
        best_epoch > 0
        and np.isfinite(best_score)
        and best_score
        < float(baseline_validation["source_balanced_weighted_mse"])
        and selected_validation["dagger8"]["weighted_mse"]
        < baseline_validation["dagger8"]["weighted_mse"]
        and selected_phase["3"] <= baseline_phase["3"]
        and selected_phase["4"] < baseline_phase["4"]
        and selected_validation["dagger8"]["unweighted_weighted_mse"] <= 0.12
        and all(
            selected_validation[name]["weighted_mse"] <= cap
            for name, cap in caps.items()
        )
        and source_balance_audit
    )


def configure_core() -> None:
    vg040.configure_core()
    core.TAG = TAG
    core.REPORT_SCHEMA = REPORT_SCHEMA
    core.STATE_SCHEMA = STATE_SCHEMA
    core.SEED = SEED
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.SOURCE_WEIGHTS = dict(SOURCE_WEIGHTS)
    core.ROW_OBJECTIVE_WEIGHTERS = {"dagger8": phase_row_weights}
    core.DATASET_EVALUATORS = {"dagger8": evaluate_phase_balanced}
    core.EXTRA_TRAINING_IDENTITY = {
        "dagger8_phase_row_weights": list(PHASE_ROW_WEIGHTS),
        "dagger8_row_weighting_changes_actor_input": False,
        "causal_sequences_preserved": True,
        "validation_selection_uses_phase_balanced_dagger8_mse": True,
    }
    core.NUMERICAL_ADMISSION_PREDICATE = numerical_admission_predicate
    core.verify_inputs = verify_inputs
    core.source_paths = source_paths
    core.current_source_identity = current_source_identity


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: PhaseBalancedConfig = PhaseBalancedConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    if tuple(config.dagger8_phase_row_weights) != PHASE_ROW_WEIGHTS:
        raise RuntimeError("VG042 phase weights changed")
    vg040.verify_inputs()
    configure_core()
    return core.train(
        output=output,
        device_name=device_name,
        config=config,
        resume=resume,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
