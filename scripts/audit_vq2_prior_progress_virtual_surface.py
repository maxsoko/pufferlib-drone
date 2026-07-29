#!/usr/bin/env python3
"""Read-only virtual first-step surface for N716 prior progress repair."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _sampling_schedule,
)
from scripts.continue_vq2_prior_progress_offline import (
    MATERIALIZATION_SHA256,
    N716_CHECKPOINT_SHA256,
    N716_REPORT_SHA256,
    N717_REPORT_SHA256,
    TRAIN_SHA256,
    _losses,
    _metrics_to_float,
    _ridge_tensors,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    _fit_from_auxiliary,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_prior_progress_virtual_step_surface_v1"
N718_PREREG_SHA256 = (
    "f6381c143e73ee7f35165e017c9a21589fae693a1105b521570cc0b4f027e1d4"
)
OBJECTIVES = {
    "level_only": (0.0, 1.0, 0.0),
    "level_plus_delta": (0.0, 1.0, 1.0),
    "kl_plus_level": (1.0, 1.0, 0.0),
    "kl_plus_level_plus_delta": (1.0, 1.0, 1.0),
}
LEARNING_RATES = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6)


@torch.no_grad()
def _virtual_adam_step(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    source: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    learning_rate: float,
    *,
    epsilon: float = 1e-8,
) -> None:
    if learning_rate <= 0.0 or epsilon <= 0.0:
        raise ValueError("virtual Adam scale must be positive")
    for name, parameter in named_parameters:
        gradient = gradients[name]
        parameter.copy_(
            source[name].to(parameter.device)
            - learning_rate * gradient / (gradient.abs() + epsilon)
        )


@torch.no_grad()
def _restore(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    source: dict[str, torch.Tensor],
) -> None:
    for name, parameter in named_parameters:
        parameter.copy_(source[name].to(parameter.device))


def _objective_value(
    metrics: dict[str, float], weights: tuple[float, float, float]
) -> float:
    return (
        weights[0] * metrics["kl"]
        + weights[1] * metrics["level"]
        + weights[2] * metrics["delta"]
    )


def _selection_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(row["after"]["progress_mae"]),
        -float(row["after"]["progress_rmse"]),
        -float(row["after"]["kl"]),
        -float(row["learning_rate"]),
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n717_report": _sha256(args.n717_report),
        "n718_preregistration": _sha256(args.n718_preregistration),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N716_CHECKPOINT_SHA256,
        "n716_report": N716_REPORT_SHA256,
        "n717_report": N717_REPORT_SHA256,
        "n718_preregistration": N718_PREREG_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("virtual surface source hash mismatch")
    n716 = json.loads(args.n716_report.read_text(encoding="utf-8"))
    n717 = json.loads(args.n717_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n716.get("process_passed") is not True
        or n717.get("process_passed") is not True
        or n717.get("prior_admitted") is not False
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("virtual surface predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("virtual surface fixed ridge mismatch")
    ridge = _ridge_tensors(auxiliary, device)
    model = _model_from_payload(payload, device)
    model_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_prior = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith("rssm.prior.")
    ]
    for _name, parameter in named_prior:
        parameter.requires_grad_(True)
    reference = _posterior_reference(
        model, arrays, crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    batch = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=719,
        batch_size=16, updates=1,
    )[0]
    event, start = batch[:, 0], batch[:, 1]
    legal, action = _legal_action_batch(
        arrays, event, start, sequence_length=128, device=device
    )
    initial = _pair_initial_state(arrays, reference, event, start, device=device)
    with torch.no_grad():
        _base_total, base_metrics_t = _losses(
            model, legal, action, initial, ridge, burn_in=32
        )
    base_metrics = _metrics_to_float(base_metrics_t)
    rows: list[dict[str, object]] = []
    for objective, weights in OBJECTIVES.items():
        _restore(named_prior, model_before)
        model.zero_grad(set_to_none=True)
        _total, metrics_t = _losses(model, legal, action, initial, ridge, burn_in=32)
        objective_loss = (
            weights[0] * metrics_t["kl"]
            + weights[1] * metrics_t["level"]
            + weights[2] * metrics_t["delta"]
        )
        objective_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _name, parameter in named_prior], 1.0
        )
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in named_prior
        }
        for learning_rate in LEARNING_RATES:
            _virtual_adam_step(
                named_prior, model_before, gradients, learning_rate
            )
            with torch.no_grad():
                _after_total, after_metrics_t = _losses(
                    model, legal, action, initial, ridge, burn_in=32
                )
            after = _metrics_to_float(after_metrics_t)
            before_objective = _objective_value(base_metrics, weights)
            after_objective = _objective_value(after, weights)
            eligible = bool(
                after["progress_mae"] < base_metrics["progress_mae"]
                and after["kl"] <= base_metrics["kl"] + 0.001
                and after_objective < before_objective
            )
            rows.append(
                {
                    "objective": objective,
                    "weights": {"kl": weights[0], "level": weights[1], "delta": weights[2]},
                    "learning_rate": learning_rate,
                    "gradient_norm_preclip": float(gradient_norm.detach().cpu()),
                    "before": base_metrics,
                    "after": after,
                    "before_objective": before_objective,
                    "after_objective": after_objective,
                    "eligible": eligible,
                }
            )
            _restore(named_prior, model_before)
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=_selection_key) if eligible_rows else None
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n717_report": _sha256(args.n717_report),
        "n718_preregistration": _sha256(args.n718_preregistration),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_restored_exact": changed == [],
        "fixed_surface_exact": tuple(LEARNING_RATES) == (1e-8, 3e-8, 1e-7, 3e-7, 1e-6) and len(OBJECTIVES) == 4,
        "twenty_virtual_trials": len(rows) == 20,
        "finite_surface": all(
            math.isfinite(float(value))
            for row in rows
            for section in ("before", "after")
            for value in row[section].values()
        ),
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "surface": rows,
        "eligible_candidates": len(eligible_rows),
        "selected_candidate": selected,
        "selection_informative": selected is not None,
        "model_tensor_changes_after_restore": changed,
        "virtual_parameter_trials": len(rows),
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "actor_updates": 0,
        "prior_updates": 0,
        "reward_updates": 0,
        "validation_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if not report["process_passed"]:
        raise RuntimeError("virtual surface process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n716-report", type=Path, required=True)
    parser.add_argument("--n717-report", type=Path, required=True)
    parser.add_argument("--n718-preregistration", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("virtual surface refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
