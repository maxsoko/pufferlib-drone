#!/usr/bin/env python3
"""Diagnose full-train multi-horizon prior-gradient conflict at N720."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.audit_vq2_prior_multihorizon_aggregate_surface import _mean_rows
from scripts.audit_vq2_prior_multihorizon_surface import (
    HORIZONS,
    _float_rows,
    _observe,
    _surface_metrics,
)
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _sampling_schedule,
    _sampling_summary,
)
from scripts.continue_vq2_prior_progress_offline import (
    MATERIALIZATION_SHA256,
    TRAIN_SHA256,
    _ridge_tensors,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_prior_full_train_horizon_gradient_conflict_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N727_REPORT_SHA256 = "13dfbbf1cf1b2835d1cf9e3c42ce73f2dcedbb8f3126a47db341f19d2ad4ff78"
RANDOM_WEIGHT_CANDIDATES = 8192


def _cosine_matrix(gradients: torch.Tensor) -> list[list[float]]:
    norms = gradients.norm(dim=1).clamp_min(1e-30)
    cosine = gradients @ gradients.T / (norms[:, None] * norms[None, :])
    return cosine.detach().cpu().double().tolist()


def _direction_row(
    gradients: torch.Tensor,
    losses: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, object]:
    weights = weights / weights.sum()
    combined = weights @ gradients
    norm = combined.norm()
    clipped = combined / torch.maximum(norm, torch.ones_like(norm))
    adam_direction = -clipped / (clipped.abs() + epsilon)
    derivatives = gradients @ adam_direction
    relative = derivatives / losses
    return {
        "weights": [float(value) for value in weights.detach().cpu()],
        "combined_gradient_norm_preclip": float(norm.detach().cpu()),
        "directional_derivative": {
            str(horizon): float(derivatives[index].detach().cpu())
            for index, horizon in enumerate(HORIZONS)
        },
        "relative_directional_derivative": {
            str(horizon): float(relative[index].detach().cpu())
            for index, horizon in enumerate(HORIZONS)
        },
        "common_adam_descent": bool(torch.all(derivatives < 0).item()),
        "worst_relative_derivative": float(relative.max().detach().cpu()),
        "mean_relative_derivative": float(relative.mean().detach().cpu()),
    }


def _weight_candidates() -> tuple[np.ndarray, int]:
    canonical: list[list[float]] = []
    for h8_weight in (1.0, 2.0, 4.0, 8.0):
        for h16_weight in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            canonical.append([1.0, 1.0, 1.0, h8_weight, h16_weight])
    rng = np.random.default_rng(728)
    random = rng.dirichlet(np.full(len(HORIZONS), 0.5), RANDOM_WEIGHT_CANDIDATES)
    return np.concatenate((np.asarray(canonical, dtype=np.float64), random), axis=0), len(canonical)


def _selection_key(row: dict[str, object]) -> tuple[float, float]:
    return (
        float(row["worst_relative_derivative"]),
        float(row["mean_relative_derivative"]),
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n727_report": _sha256(args.n727_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n727_report": N727_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("horizon-gradient source hash mismatch")
    n727 = json.loads(args.n727_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n727.get("process_passed") is not True
        or n727.get("selection_informative") is not False
        or n727.get("eligible_candidates") != 0
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("horizon-gradient predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("horizon-gradient fixed ridge mismatch")
    model = _model_from_payload(payload, device)
    model_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_prior = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("rssm.prior.")
    ]
    prior_parameters = [parameter for _name, parameter in named_prior]
    for parameter in prior_parameters:
        parameter.requires_grad_(True)
    ridge = _ridge_tensors(auxiliary, device)
    reference = _posterior_reference(
        model,
        arrays,
        crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]),
        FIXED_CROP_STARTS,
        seed=728,
        batch_size=16,
        updates=119,
    )
    sampling = _sampling_summary(schedule, len(arrays["mask"]))
    gradient_buffers = [
        [torch.zeros_like(parameter) for parameter in prior_parameters]
        for _horizon in HORIZONS
    ]
    metric_rows: list[dict[str, dict[str, float]]] = []
    for batch_index, batch in enumerate(schedule, 1):
        event, start = batch[:, 0], batch[:, 1]
        legal, action = _legal_action_batch(
            arrays, event, start, sequence_length=128, device=device
        )
        initial = _pair_initial_state(arrays, reference, event, start, device=device)
        posterior = _observe(model, legal, action, initial)
        _objective, metrics = _surface_metrics(
            model, posterior, action, ridge, burn_in=32, stride=8
        )
        for horizon_index, horizon in enumerate(HORIZONS):
            batch_gradients = torch.autograd.grad(
                metrics[str(horizon)]["loss"] / len(schedule),
                prior_parameters,
                retain_graph=horizon_index + 1 < len(HORIZONS),
            )
            for accumulator, gradient in zip(
                gradient_buffers[horizon_index], batch_gradients, strict=True
            ):
                accumulator.add_(gradient)
        metric_rows.append(_float_rows(metrics))
        if args.progress_every and batch_index % args.progress_every == 0:
            print(
                json.dumps(
                    {"phase": "gradients", "batch": batch_index, "batches": len(schedule)},
                    sort_keys=True,
                ),
                flush=True,
            )
    baseline = _mean_rows(metric_rows)
    gradients = torch.stack(
        [
            torch.cat([value.reshape(-1) for value in horizon_gradients])
            for horizon_gradients in gradient_buffers
        ]
    ).double()
    losses = torch.tensor(
        [baseline[str(horizon)]["loss"] for horizon in HORIZONS],
        device=device,
        dtype=torch.float64,
    )
    weight_candidates, canonical_count = _weight_candidates()
    weight_sha = hashlib.sha256(weight_candidates.astype("<f8").tobytes()).hexdigest()
    rows = [
        _direction_row(
            gradients,
            losses,
            torch.as_tensor(weights, device=device, dtype=torch.float64),
        )
        for weights in weight_candidates
    ]
    common = [row for row in rows if row["common_adam_descent"]]
    selected = min(common, key=_selection_key) if common else None
    canonical_rows = rows[:canonical_count]
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    gradient_norms = {
        str(horizon): float(gradients[index].norm().detach().cpu())
        for index, horizon in enumerate(HORIZONS)
    }
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n727_report": _sha256(args.n727_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_tensors_exact": changed == [],
        "full_train_schedule_exact": (
            sampling["draws"] == 1904
            and sampling["unseen_pairs"] == 0
            and sampling["pair_count_minimum"] == 1
            and sampling["pair_count_maximum"] == 1
            and sampling["event_count_minimum"] == 7
            and sampling["event_count_maximum"] == 7
        ),
        "fixed_horizons_exact": HORIZONS == (1, 2, 4, 8, 16),
        "weight_search_exact": (
            len(rows) == canonical_count + RANDOM_WEIGHT_CANDIDATES
            and canonical_count == 28
        ),
        "finite_diagnostics": all(
            math.isfinite(value)
            for value in [
                *gradient_norms.values(),
                *(value for row in rows for value in row["relative_directional_derivative"].values()),
            ]
        ),
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "sampling": sampling,
        "baseline": baseline,
        "gradient_norms": gradient_norms,
        "gradient_cosine_matrix_horizon_order": list(HORIZONS),
        "gradient_cosine_matrix": _cosine_matrix(gradients),
        "weight_candidate_seed": 728,
        "weight_candidate_dirichlet_alpha": 0.5,
        "weight_candidate_count": len(rows),
        "weight_candidate_sha256": weight_sha,
        "canonical_weight_rows": canonical_rows,
        "common_adam_descent_candidates": len(common),
        "selected_candidate": selected,
        "selection_informative": selected is not None,
        "model_tensor_changes": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "prior_updates": 0,
        "actor_updates": 0,
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
        raise RuntimeError("horizon-gradient process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n727-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("horizon-gradient audit refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
