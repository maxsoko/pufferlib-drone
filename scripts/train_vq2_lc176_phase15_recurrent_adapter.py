#!/usr/bin/env python3
"""Fit a legal phase-local recurrent Puffer adapter on LC172 sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseLocalAdapterActor
from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc176_phase15_recurrent_adapter_001"
SCHEMA = "vq2_lc176_phase15_recurrent_adapter_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc176_phase15_recurrent_adapter_checkpoint_v1"
TARGET_PHASE = 15
ADAPTER_SIZE = 64
EPOCHS = 20
BATCH_AGENTS = 64
LEARNING_RATE = 1e-3
MAX_GRADIENT_NORM = 1.0
MINIMUM_VALIDATION_IMPROVEMENT = 5.0
TRAINABLE_ADAPTER_PREFIXES = ("phase_adapter_cell.", "phase_adapter_output.")
NEXT_AUTHORITY_ADMITTED = (
    "Run one teacher-free LC169-versus-LC176 raw-16 screen; no FlightSim authority."
)
NEXT_AUTHORITY_REJECTED = "Reject LC176; do not screen or run FlightSim."
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc169_phase15_failure_state_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "8ae1a6d9aebd01d584344f006724f5a81699a011a759356283ada1f271ece1b7"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "182e81da9cb41a7f8479f0a2cfa99baef9b408a63cdfd1024130e38349cccf9b"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc172_phase15_failure_state_dagger3_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "ddb15553a498e325edf14078dab6a1ca596750604aca9b84e82d497295f13d32"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "6e0f176d7e0ff328e1cfe0f898525c5d93d6da1fa219d4905065bc90b1c3985c"
LC175_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc175_phase15_alignment_return_ppo_001/report.json"
LC175_REPORT_SHA256 = "87bba2917e8a9e63d6370dbaa5cbc3be3a5aaa046faa8b732c70bb8e19f20e2f"
LC177_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc177_phase15_alignment_ppo_milestone_001/report.json"
LC177_REPORT_SHA256 = "677c0c76e35513fc751cca4d27a647a77eb47b7fd976566a8e7b5b245228126a"
PREREGISTRATION = ROOT / "docs/vq2_lc176_phase15_recurrent_adapter_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc176_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc176_phase15_recurrent_adapter.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC175_REPORT: LC175_REPORT_SHA256,
        LC177_REPORT: LC177_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC176 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    ppo = json.loads(LC175_REPORT.read_text())
    rejected = json.loads(LC177_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc172_phase15_failure_state_dagger3_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 375_552
        or dataset.get("query_outcome_success_agents") != list(range(256, 512))
        or dataset.get("query_outcome_failure_agents") != list(range(256))
        or ppo.get("schema") != "vq2_lc175_phase15_alignment_return_ppo_report_v1"
        or not ppo.get("training_admitted")
        or ppo.get("candidate_selected_for_screen", {}).get("sha256")
        != "d4f59e06dc98a80c01b28097be38f9e80af61d725f31f887a911058acb09d950"
        or ppo.get("selected_rollout_metrics", {}).get("target_passes") != 6
        or rejected.get("schema") != "vq2_lc177_phase15_alignment_ppo_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC169/LC172/LC175/LC177 do not authorize LC176")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        LC175_REPORT, LC177_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc172_phase15_failure_state_dagger3.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def sequence_contract(records: np.ndarray) -> dict[str, Any]:
    lengths = np.bincount(
        np.asarray(records["agent_index"], dtype=np.int64), minlength=512
    )
    starts = np.full(512, np.iinfo(np.uint16).max, dtype=np.uint16)
    ends = np.zeros(512, dtype=np.uint16)
    np.minimum.at(starts, records["agent_index"], records["step"])
    np.maximum.at(ends, records["agent_index"], records["step"])
    return {
        "agents": int((lengths > 0).sum()),
        "control_length_min": int(lengths[:256].min()),
        "control_length_max": int(lengths[:256].max()),
        "rescue_length_min": int(lengths[256:].min()),
        "rescue_length_max": int(lengths[256:].max()),
        "start_step_min": int(starts.min()),
        "start_step_max": int(starts.max()),
        "end_step_control": int(ends[:256].max()),
        "end_step_rescue": int(ends[256:].max()),
        "contiguous": bool(np.array_equal(lengths, ends.astype(np.int64) - starts.astype(np.int64) + 1)),
    }


def materialize_sequences(records: np.ndarray) -> tuple[np.ndarray, ...]:
    contract = sequence_contract(records)
    if contract != {
        "agents": 512,
        "control_length_min": 481,
        "control_length_max": 481,
        "rescue_length_min": 986,
        "rescue_length_max": 986,
        "start_step_min": 19_920,
        "start_step_max": 19_920,
        "end_step_control": 20_400,
        "end_step_rescue": 20_905,
        "contiguous": True,
    }:
        raise RuntimeError(f"LC176 sequence contract changed: {contract}")
    maximum = 986
    hidden = np.zeros((512, maximum, 256), dtype=np.float16)
    base = np.zeros((512, maximum, 4), dtype=np.float32)
    teacher = np.zeros((512, maximum, 4), dtype=np.float32)
    mask = np.zeros((512, maximum), dtype=bool)
    lengths = np.zeros(512, dtype=np.int64)
    agent_field = np.asarray(records["agent_index"])
    for agent in range(512):
        selected = np.flatnonzero(agent_field == agent)
        sequence = records[selected]
        order = np.argsort(sequence["step"], kind="stable")
        sequence = sequence[order]
        length = len(sequence)
        hidden[agent, :length] = sequence["hidden"]
        base[agent, :length] = sequence["base_pre_tanh"]
        teacher[agent, :length] = sequence["teacher_action"]
        mask[agent, :length] = True
        lengths[agent] = length
    return hidden, base, teacher, mask, lengths


def evaluate(
    gru: nn.GRU,
    head: nn.Linear,
    arrays: tuple[np.ndarray, ...],
    agents: np.ndarray,
    *,
    device: torch.device,
) -> float:
    hidden, base, teacher, mask, lengths = arrays
    weighted_error = 0.0
    agent_count = 0
    gru.eval()
    head.eval()
    with torch.no_grad():
        for start in range(0, len(agents), BATCH_AGENTS):
            selected = agents[start : start + BATCH_AGENTS]
            size = int(lengths[selected].max())
            x = torch.from_numpy(hidden[selected, :size].astype(np.float32)).to(device)
            base_tensor = torch.from_numpy(base[selected, :size]).to(device)
            target = torch.from_numpy(teacher[selected, :size]).to(device)
            valid = torch.from_numpy(mask[selected, :size]).to(device)
            recurrent_output, _ = gru(x)
            action = torch.tanh(base_tensor + head(recurrent_output))
            row_error = (action - target).square().mean(dim=-1)
            weights = valid.float() / torch.from_numpy(lengths[selected]).to(device)[:, None]
            weighted_error += float((row_error * weights).sum())
            agent_count += len(selected)
    return weighted_error / agent_count


def parent_error(arrays: tuple[np.ndarray, ...], agents: np.ndarray) -> float:
    _, base, teacher, mask, lengths = arrays
    action = np.tanh(base[agents].astype(np.float64))
    row_error = np.mean((action - teacher[agents]) ** 2, axis=-1)
    weights = mask[agents] / lengths[agents, None]
    return float((row_error * weights).sum() / len(agents))


def build_actor(parent: dict[str, Any]) -> VQ2PhaseLocalAdapterActor:
    contract = parent["model"]
    actor = VQ2PhaseLocalAdapterActor(
        target_phase=TARGET_PHASE,
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        adapter_size=ADAPTER_SIZE,
        initial_std=float(contract["initial_std"]),
    )
    actor.load_base_state(parent["model_state"])
    return actor


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    parent = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC176 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC176 does not resume a partial fit")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    arrays = materialize_sequences(records)
    validation_agents = np.arange(0, 512, 4, dtype=np.int64)
    training_agents = np.setdiff1d(np.arange(512, dtype=np.int64), validation_agents)
    if len(training_agents) != 384 or len(validation_agents) != 128:
        raise RuntimeError("LC176 agent split changed")

    contract = parent["model"]
    actor = build_actor(parent)
    base_state_hash = state_sha256(parent["model_state"])
    device = torch.device(device_name)
    gru = nn.GRU(int(contract["hidden_size"]), ADAPTER_SIZE, batch_first=True).to(device)
    head = nn.Linear(ADAPTER_SIZE, 4).to(device)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(actor.phase_adapter_cell.weight_ih)
        gru.weight_hh_l0.copy_(actor.phase_adapter_cell.weight_hh)
        gru.bias_ih_l0.copy_(actor.phase_adapter_cell.bias_ih)
        gru.bias_hh_l0.copy_(actor.phase_adapter_cell.bias_hh)
        head.weight.copy_(actor.phase_adapter_output.weight)
        head.bias.copy_(actor.phase_adapter_output.bias)
    initial_trainable = [value.detach().cpu().clone() for value in (*gru.parameters(), *head.parameters())]
    optimizer = torch.optim.Adam((*gru.parameters(), *head.parameters()), lr=LEARNING_RATE)
    generator = np.random.default_rng(432_260)
    parent_validation_mse = evaluate(
        gru, head, arrays, validation_agents, device=device
    )
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        gru.train()
        head.train()
        shuffled = generator.permutation(training_agents)
        losses: list[float] = []
        for start in range(0, len(shuffled), BATCH_AGENTS):
            selected = shuffled[start : start + BATCH_AGENTS]
            hidden, base, teacher, mask, lengths = arrays
            size = int(lengths[selected].max())
            x = torch.from_numpy(hidden[selected, :size].astype(np.float32)).to(device)
            base_tensor = torch.from_numpy(base[selected, :size]).to(device)
            target = torch.from_numpy(teacher[selected, :size]).to(device)
            valid = torch.from_numpy(mask[selected, :size]).to(device)
            recurrent_output, _ = gru(x)
            action = torch.tanh(base_tensor + head(recurrent_output))
            row_error = (action - target).square().mean(dim=-1)
            weights = valid.float() / torch.from_numpy(lengths[selected]).to(device)[:, None]
            loss = (row_error * weights).sum() / len(selected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((*gru.parameters(), *head.parameters()), MAX_GRADIENT_NORM)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_mse = evaluate(gru, head, arrays, validation_agents, device=device)
        training_mse = evaluate(gru, head, arrays, training_agents, device=device)
        history.append({
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(losses)),
            "training_teacher_action_mse": training_mse,
            "validation_teacher_action_mse": validation_mse,
            "validation_improvement_factor": parent_validation_mse / validation_mse,
        })
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                **{f"gru.{name}": value.detach().cpu().clone() for name, value in gru.state_dict().items()},
                **{f"head.{name}": value.detach().cpu().clone() for name, value in head.state_dict().items()},
            }
    if best_state is None:
        raise RuntimeError("LC176 selected no finite epoch")
    gru.load_state_dict({name[4:]: value for name, value in best_state.items() if name.startswith("gru.")})
    head.load_state_dict({name[5:]: value for name, value in best_state.items() if name.startswith("head.")})
    with torch.no_grad():
        actor.phase_adapter_cell.weight_ih.copy_(gru.weight_ih_l0.cpu())
        actor.phase_adapter_cell.weight_hh.copy_(gru.weight_hh_l0.cpu())
        actor.phase_adapter_cell.bias_ih.copy_(gru.bias_ih_l0.cpu())
        actor.phase_adapter_cell.bias_hh.copy_(gru.bias_hh_l0.cpu())
        actor.phase_adapter_output.weight.copy_(head.weight.cpu())
        actor.phase_adapter_output.bias.copy_(head.bias.cpu())
    model_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    frozen_base_exact = all(
        torch.equal(model_state[name], value)
        for name, value in parent["model_state"].items()
        if not name.startswith(TRAINABLE_ADAPTER_PREFIXES)
    )
    final_trainable = [value.detach().cpu() for value in (*gru.parameters(), *head.parameters())]
    parameter_delta_l2 = math.sqrt(sum(float((a - b).square().sum()) for a, b in zip(final_trainable, initial_trainable)))
    improvement = parent_validation_mse / best_validation
    finite = bool(math.isfinite(best_validation) and math.isfinite(parameter_delta_l2))
    admitted = bool(finite and frozen_base_exact and improvement >= MINIMUM_VALIDATION_IMPROVEMENT)
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model": {**contract, "adapter_size": ADAPTER_SIZE, "adapter_target_phase": TARGET_PHASE},
        "model_state": model_state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "numerically_admitted": admitted,
        "deployment_candidate": False,
        "phase15_recurrent_adapter": {
            "best_epoch": best_epoch,
            "validation_teacher_action_mse": best_validation,
            "validation_improvement_factor": improvement,
        },
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "candidate_state_sha256": state_sha256(model_state),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_sha256": base_state_hash,
        "frozen_base_state_exact": frozen_base_exact,
        "sequence_contract": sequence_contract(records),
        "training_agents": training_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "parent_validation_teacher_action_mse": parent_validation_mse,
        "best_epoch": best_epoch,
        "best_validation_teacher_action_mse": best_validation,
        "validation_improvement_factor": improvement,
        "adapter_parameter_delta_l2": parameter_delta_l2,
        "history": history,
        "configuration": {
            "target_phase": TARGET_PHASE, "adapter_size": ADAPTER_SIZE,
            "epochs": EPOCHS, "batch_agents": BATCH_AGENTS,
            "learning_rate": LEARNING_RATE,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
            "minimum_validation_improvement": MINIMUM_VALIDATION_IMPROVEMENT,
        },
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0,
            "training_only_teacher_targets": True,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            NEXT_AUTHORITY_ADMITTED if admitted else NEXT_AUTHORITY_REJECTED
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
