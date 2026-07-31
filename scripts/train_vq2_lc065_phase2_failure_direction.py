#!/usr/bin/env python3
"""Fit a legal-feature phase-2 failure direction for closed-loop surgery."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import FEATURE_DTYPE
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc065_phase2_failure_direction_001"
SCHEMA = "vq2_lc065_phase2_failure_direction_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc065_phase2_failure_direction_checkpoint_v1"
SEED = 431650
TARGET_PHASE = 2
FEATURE_CHUNK = 65_536
MAX_EPOCHS = 400
LEARNING_RATE = 0.05
L2 = 1e-4
PATIENCE = 60
MINIMUM_VALIDATION_AGENT_AUC = 0.65
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc062_phase2_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "06381161445f5f207a3b12f7c97b82c884f91a71fdb3e34e05189255cc58d04a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5bf0fc1f95fec795dcf0c8212f7838a0f18a683baaece679228e87e667085af5"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc028_index2_student_dagger_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "4a74f8178a7f3abd33bba8f1ec2009245c53b5dcb06bf03b26267f96f7650701"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "ec8d760edeca4f07c4238f6edd76fb4db7ebc955bf0506f619020615b0ba1a1e"
LC064_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc064_phase2_residual_direction_001/report.json"
)
LC064_REPORT_SHA256 = "f7349cb80d503f6bf8450bd6070303eda722a7374e7f2f5a21592f29cf6b06d8"
PREREGISTRATION = ROOT / "docs/vq2_lc065_phase2_failure_direction_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc065_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc065_phase2_failure_direction.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def terminal_outcome_by_agent(
    agents: np.ndarray, steps: np.ndarray, terminals: np.ndarray, *, capacity: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    agents = np.asarray(agents, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    terminals = np.asarray(terminals, dtype=np.uint8)
    if agents.shape != steps.shape or agents.shape != terminals.shape or agents.ndim != 1:
        raise ValueError("LC065 outcome columns do not align")
    if agents.size == 0 or agents.min() < 0 or agents.max() >= capacity:
        raise ValueError("LC065 agent indices are invalid")
    last_step = np.full(capacity, -1, dtype=np.int64)
    np.maximum.at(last_step, agents, steps)
    last_mask = steps == last_step[agents]
    present = np.flatnonzero(last_step >= 0)
    if int(last_mask.sum()) != int(present.size):
        raise RuntimeError("LC065 expected one final phase-2 record per queried agent")
    outcome = np.full(capacity, -1, dtype=np.int8)
    outcome[agents[last_mask]] = (terminals[last_mask] == 0).astype(np.int8)
    return present, outcome


def stratified_agent_split(
    present: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        ids = present[outcome[present] == label].copy()
        rng.shuffle(ids)
        cut = max(1, min(ids.size - 1, int(round(0.8 * ids.size))))
        train.extend(ids[:cut].tolist())
        validation.extend(ids[cut:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def weighted_standardization(
    features: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = weights / weights.sum()
    mean = (features * normalized[:, None]).sum(dim=0)
    variance = ((features - mean).square() * normalized[:, None]).sum(dim=0)
    return mean, variance.clamp_min(1e-8).sqrt()


def agent_means(values: np.ndarray, agents: np.ndarray, present: np.ndarray) -> np.ndarray:
    sums = np.bincount(agents, weights=values, minlength=256)
    counts = np.bincount(agents, minlength=256)
    return sums[present] / counts[present]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("LC065 AUC requires both outcome classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normalize_failure_direction(
    weight: torch.Tensor, bias: torch.Tensor, features: torch.Tensor,
    agents: np.ndarray, present: np.ndarray, outcome: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    scores = (features @ weight + bias).detach().cpu().numpy()
    means = agent_means(scores, agents, present)
    labels = outcome[present]
    success_mean = float(means[labels == 1].mean())
    failure_mean = float(means[labels == 0].mean())
    gap = failure_mean - success_mean
    if not np.isfinite(gap) or gap <= 1e-6:
        raise RuntimeError("LC065 classifier does not orient toward failure")
    return weight / gap, (bias - success_mean) / gap, success_mean, failure_mean


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC064_REPORT: LC064_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC065 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC064_REPORT.read_text())
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or payload.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc028_index2_student_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 431_450
        or dataset.get("teacher_plant_actions_executed") != 0
        or rejected.get("schema") != "vq2_lc064_phase2_residual_direction_report_v1"
        or rejected.get("causal_screen_selected") is not None
        or not rejected.get("diagnostic_valid")
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062/LC064 and LC028 do not authorize LC065")
    return payload


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC064_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in paths
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    payload = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC065 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC065 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    started = time.perf_counter()

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    steps = np.asarray(records["step"], dtype=np.int64)
    terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = terminal_outcome_by_agent(agents, steps, terminals)
    if present.size != 248 or int((outcome[present] == 1).sum()) != 57:
        raise RuntimeError("LC065 source-locked phase-2 outcome counts changed")
    train_agents, validation_agents = stratified_agent_split(present, outcome)
    train_agent_mask = np.zeros(256, dtype=bool)
    validation_agent_mask = np.zeros(256, dtype=bool)
    train_agent_mask[train_agents] = True
    validation_agent_mask[validation_agents] = True
    train_mask_np = train_agent_mask[agents]
    validation_mask_np = validation_agent_mask[agents]

    parent_state = payload["model_state"]
    input_weight = parent_state["indexed_phase_residual_input"][TARGET_PHASE].to(device)
    input_bias = parent_state["indexed_phase_residual_input_bias"][TARGET_PHASE].to(device)
    feature_values = np.empty((records.size, input_weight.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, records.size, FEATURE_CHUNK):
            stop = min(records.size, start + FEATURE_CHUNK)
            hidden = torch.from_numpy(
                np.asarray(records["hidden"][start:stop], dtype=np.float32)
            ).to(device)
            transformed = torch.tanh(hidden @ input_weight.T + input_bias)
            feature_values[start:stop] = transformed.cpu().numpy()

    features = torch.from_numpy(feature_values).to(device)
    labels_np = (outcome[agents] == 0).astype(np.float32)  # failure is positive
    labels = torch.from_numpy(labels_np).to(device)
    counts = np.bincount(agents, minlength=256).astype(np.float64)
    record_weights_np = 1.0 / counts[agents]
    for mask, ids in ((train_mask_np, train_agents), (validation_mask_np, validation_agents)):
        for label in (0, 1):
            class_ids = ids[outcome[ids] == label]
            class_records = mask & np.isin(agents, class_ids)
            record_weights_np[class_records] /= max(1, class_ids.size)
    train_index = torch.from_numpy(np.flatnonzero(train_mask_np)).to(device)
    validation_index = torch.from_numpy(np.flatnonzero(validation_mask_np)).to(device)
    train_weights = torch.from_numpy(record_weights_np[train_mask_np].astype(np.float32)).to(device)
    validation_weights = torch.from_numpy(record_weights_np[validation_mask_np].astype(np.float32)).to(device)
    train_features = features.index_select(0, train_index)
    validation_features = features.index_select(0, validation_index)
    train_labels = labels.index_select(0, train_index)
    validation_labels = labels.index_select(0, validation_index)
    mean, scale = weighted_standardization(train_features, train_weights)
    train_standard = (train_features - mean) / scale
    validation_standard = (validation_features - mean) / scale

    torch.manual_seed(SEED)
    weight = torch.zeros(train_standard.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((weight, bias), lr=LEARNING_RATE)
    best = (float("inf"), -1, None, None)
    history: list[dict[str, float | int]] = []
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = train_standard @ weight + bias
        loss = (
            F.binary_cross_entropy_with_logits(logits, train_labels, reduction="none")
            * (train_weights / train_weights.sum())
        ).sum() + L2 * weight.square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_logits = validation_standard @ weight + bias
            validation_loss = float((
                F.binary_cross_entropy_with_logits(
                    validation_logits, validation_labels, reduction="none"
                ) * (validation_weights / validation_weights.sum())
            ).sum())
        if validation_loss < best[0] - 1e-7:
            best = (
                validation_loss, epoch,
                weight.detach().cpu().clone(), bias.detach().cpu().clone(),
            )
        if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
            history.append({
                "epoch": epoch, "train_loss": float(loss.detach()),
                "validation_loss": validation_loss,
            })
        if epoch - int(best[1]) >= PATIENCE:
            break
    if best[2] is None or best[3] is None:
        raise RuntimeError("LC065 optimizer produced no finite classifier")
    standardized_weight = best[2].to(device)
    standardized_bias = best[3].to(device)
    raw_weight = standardized_weight / scale
    raw_bias = standardized_bias - (standardized_weight * mean / scale).sum()
    raw_weight, raw_bias, old_success_mean, old_failure_mean = normalize_failure_direction(
        raw_weight, raw_bias, features, agents, present, outcome
    )
    all_scores = (features @ raw_weight + raw_bias).detach().cpu().numpy()
    per_agent = agent_means(all_scores, agents, present)
    per_agent_labels = (outcome[present] == 0).astype(np.int8)
    train_present_mask = np.isin(present, train_agents)
    validation_present_mask = np.isin(present, validation_agents)
    train_auc = auc(per_agent_labels[train_present_mask], per_agent[train_present_mask])
    validation_auc = auc(
        per_agent_labels[validation_present_mask], per_agent[validation_present_mask]
    )
    success_scores = per_agent[outcome[present] == 1]
    failure_scores = per_agent[outcome[present] == 0]
    admitted = bool(
        np.isfinite(per_agent).all()
        and validation_auc >= MINIMUM_VALIDATION_AGENT_AUC
        and abs(float(success_scores.mean())) <= 1e-4
        and abs(float(failure_scores.mean()) - 1.0) <= 1e-4
    )
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "numerically_admitted": admitted,
        "feature_weight": raw_weight.detach().cpu(),
        "feature_bias": raw_bias.detach().cpu(),
        "feature_size": int(raw_weight.numel()),
        "target_phase": TARGET_PHASE,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "normalization": {
            "source_success_mean": old_success_mean,
            "source_failure_mean": old_failure_mean,
            "normalized_success_mean": float(success_scores.mean()),
            "normalized_failure_mean": float(failure_scores.mean()),
        },
    }
    checkpoint_path = output / "failure_direction.pt"
    atomic_torch_save(checkpoint_path, checkpoint_payload)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": "failure_direction.pt",
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "wall_time_seconds": time.perf_counter() - started,
        "feature_records": int(records.size), "query_agents": int(present.size),
        "success_agents": int((outcome[present] == 1).sum()),
        "failure_agents": int((outcome[present] == 0).sum()),
        "train_agents": int(train_agents.size),
        "validation_agents": int(validation_agents.size),
        "best_epoch": int(best[1]), "validation_loss": float(best[0]),
        "train_agent_auc": train_auc, "validation_agent_auc": validation_auc,
        "minimum_validation_agent_auc": MINIMUM_VALIDATION_AGENT_AUC,
        "normalized_success_score_mean": float(success_scores.mean()),
        "normalized_failure_score_mean": float(failure_scores.mean()),
        "normalized_success_score_std": float(success_scores.std()),
        "normalized_failure_score_std": float(failure_scores.std()),
        "direction_weight_l2": float(raw_weight.double().norm()),
        "direction_bias": float(raw_bias), "history": history,
        "source_identity": identity,
        "safety": {
            "training_only_outcome_labels": True,
            "runtime_privileged_values": 0,
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Merge scaled copies of the legal-feature direction into phase-2 Puffer output weights and run one fast paired milestone screen."
            if admitted else "Reject the outcome direction and retain LC062."
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
