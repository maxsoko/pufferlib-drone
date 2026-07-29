#!/usr/bin/env python3
"""Recompute a raw common-descent surface from the retained N731 prior."""

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
from scripts.audit_vq2_prior_multihorizon_aggregate_surface import (
    _evaluate_schedule,
    _mean_rows,
)
from scripts.audit_vq2_prior_multihorizon_surface import (
    HORIZONS,
    _eligible,
    _float_rows,
    _observe,
    _restore,
    _selection_key,
    _surface_metrics,
)
from scripts.audit_vq2_prior_raw_common_descent import _minimum_norm_weights
from scripts.audit_vq2_prior_raw_common_descent_surface import (
    _apply_unit_step,
    _baseline_parity,
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


AUDIT_SCHEMA = "vq2_prior_raw_recomputed_continuation_surface_v1"
N731_CHECKPOINT_SHA256 = (
    "ebc6f0ef1caee3cd53ba489d7440158da243d50ecfc9d353685c1b1dafe9cb1c"
)
N731_REPORT_SHA256 = "9d5841ec6420f15276a49925713a84ddd36037c24f5e497e32eebc03cf818043"
N732_REPORT_SHA256 = "2b06b237c960be7e207b7dc623fb5b254d29504959ce6032ce81b344e54ef9cc"
LEARNING_RATES = (1e-3, 3e-3, 6e-3, 1e-2, 1.3e-2)


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n731_report": _sha256(args.n731_report),
        "n732_report": _sha256(args.n732_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N731_CHECKPOINT_SHA256,
        "n731_report": N731_REPORT_SHA256,
        "n732_report": N732_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("raw continuation surface source hash mismatch")
    n731 = json.loads(args.n731_report.read_text(encoding="utf-8"))
    n732 = json.loads(args.n732_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n731.get("process_passed") is not True
        or n732.get("process_passed") is not True
        or n732.get("improved_all_horizons_vs_n720") is not True
        or n732.get("prior_admitted") is not False
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("raw continuation surface predecessor mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("raw continuation fixed ridge mismatch")
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
        seed=733,
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
            gradients = torch.autograd.grad(
                metrics[str(horizon)]["loss"] / len(schedule),
                prior_parameters,
                retain_graph=horizon_index + 1 < len(HORIZONS),
            )
            for accumulator, gradient in zip(
                gradient_buffers[horizon_index], gradients, strict=True
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
    before = _mean_rows(metric_rows)
    horizon_gradients = torch.stack(
        [
            torch.cat([value.reshape(-1) for value in gradients])
            for gradients in gradient_buffers
        ]
    ).double()
    gram = (horizon_gradients @ horizon_gradients.T).detach().cpu().numpy()
    weight_array, minimum_norm_iterations, final_gap = _minimum_norm_weights(gram)
    weights = torch.as_tensor(weight_array, device=device, dtype=torch.float64)
    combined = weights @ horizon_gradients
    unit_direction = combined / combined.norm()
    raw_derivatives = horizon_gradients @ (-unit_direction)
    common_descent = bool(torch.all(raw_derivatives < 0).item())
    rows: list[dict[str, object]] = []
    for rate_index, learning_rate in enumerate(LEARNING_RATES, 1):
        _apply_unit_step(named_prior, model_before, unit_direction, learning_rate)
        after = _evaluate_schedule(
            model,
            arrays,
            reference,
            schedule,
            ridge,
            device=device,
            progress_every=args.progress_every,
            phase=f"rate_{learning_rate:g}",
        )
        maximum_delta = max(
            float((parameter.detach().cpu() - model_before[name]).abs().max())
            for name, parameter in named_prior
        )
        eligible = common_descent and _eligible(before, after) and maximum_delta <= 0.001
        rows.append(
            {
                "learning_rate": learning_rate,
                "before": before,
                "after": after,
                "maximum_parameter_delta": maximum_delta,
                "eligible": eligible,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "rate_complete",
                    "rate_index": rate_index,
                    "rates": len(LEARNING_RATES),
                    "learning_rate": learning_rate,
                    "eligible": eligible,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _restore(named_prior, model_before)
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=_selection_key) if eligible_rows else None
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    baseline_parity = _baseline_parity(before, n731["final"])
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n731_report": _sha256(args.n731_report),
        "n732_report": _sha256(args.n732_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    gradient_norms = {
        str(horizon): float(horizon_gradients[index].norm().detach().cpu())
        for index, horizon in enumerate(HORIZONS)
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_restored_exact": changed == [],
        "full_train_schedule_exact": (
            sampling["draws"] == 1904
            and sampling["unseen_pairs"] == 0
            and sampling["pair_count_minimum"] == 1
            and sampling["pair_count_maximum"] == 1
        ),
        "n731_baseline_reproduced": baseline_parity <= 1e-6,
        "minimum_norm_weights_valid": (
            bool(np.all(weight_array >= 0.0))
            and abs(float(weight_array.sum()) - 1.0) <= 1e-12
        ),
        "five_virtual_rates": len(rows) == 5 and LEARNING_RATES == (
            1e-3,
            3e-3,
            6e-3,
            1e-2,
            1.3e-2,
        ),
        "finite_surface": all(
            math.isfinite(float(value))
            for row in rows
            for section in ("before", "after")
            for horizon in row[section].values()
            for value in horizon.values()
        ),
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "sampling": sampling,
        "n731_baseline_parity_max_abs_error": baseline_parity,
        "gradient_norms": gradient_norms,
        "gradient_cosine_matrix": (
            gram
            / np.maximum(
                np.sqrt(np.diag(gram))[:, None] * np.sqrt(np.diag(gram))[None, :],
                1e-30,
            )
        ).tolist(),
        "minimum_norm_iterations": minimum_norm_iterations,
        "minimum_norm_final_frank_wolfe_gap": final_gap,
        "horizon_weights": weight_array.tolist(),
        "combined_gradient_norm": float(combined.norm().detach().cpu()),
        "unit_raw_directional_derivative": {
            str(horizon): float(raw_derivatives[index].detach().cpu())
            for index, horizon in enumerate(HORIZONS)
        },
        "common_raw_descent": common_descent,
        "surface": rows,
        "eligible_candidates": len(eligible_rows),
        "selected_candidate": selected,
        "selection_informative": selected is not None,
        "model_tensor_changes_after_restore": changed,
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
        raise RuntimeError("raw continuation surface process gates failed")
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
    parser.add_argument("--n731-report", type=Path, required=True)
    parser.add_argument("--n732-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("raw continuation surface refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
