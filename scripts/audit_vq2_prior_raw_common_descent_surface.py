#!/usr/bin/env python3
"""Finite-rate full-train virtual surface for N729's raw common descent."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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
from scripts.audit_vq2_prior_raw_common_descent import AUDIT_SCHEMA as N729_SCHEMA
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


AUDIT_SCHEMA = "vq2_prior_raw_common_descent_full_train_surface_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N728_REPORT_SHA256 = "c9af8630ea20ec87fa21d0216957700eb27ef26cbb7378639d58013f01fba376"
N729_REPORT_SHA256 = "62d2267a1970744b19d7c34a585930e2aa379249c4b7434703b91e752f335ee0"
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@torch.no_grad()
def _apply_unit_step(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    source: dict[str, torch.Tensor],
    unit_direction: torch.Tensor,
    learning_rate: float,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError("raw-SGD learning rate must be positive")
    offset = 0
    for name, parameter in named_parameters:
        count = parameter.numel()
        direction = unit_direction[offset : offset + count].reshape(parameter.shape)
        parameter.copy_(
            source[name].to(parameter.device)
            - learning_rate * direction.to(device=parameter.device, dtype=parameter.dtype)
        )
        offset += count
    if offset != unit_direction.numel():
        raise RuntimeError("raw-SGD direction size mismatch")


def _baseline_parity(
    actual: dict[str, dict[str, float]], expected: dict[str, object]
) -> float:
    return max(
        abs(float(actual[str(horizon)][name]) - float(expected[str(horizon)][name]))
        for horizon in HORIZONS
        for name in actual[str(horizon)]
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n728_report": _sha256(args.n728_report),
        "n729_report": _sha256(args.n729_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n728_report": N728_REPORT_SHA256,
        "n729_report": N729_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("raw common-descent surface source hash mismatch")
    n728 = json.loads(args.n728_report.read_text(encoding="utf-8"))
    n729 = json.loads(args.n729_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    selected = n729.get("minimum_norm_candidate")
    if (
        n728.get("process_passed") is not True
        or n729.get("contract") != N729_SCHEMA
        or n729.get("process_passed") is not True
        or n729.get("selection_informative") is not True
        or not isinstance(selected, dict)
        or selected.get("common_raw_descent") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("raw common-descent surface predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("raw common-descent fixed ridge mismatch")
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
    weights = torch.as_tensor(
        selected["weights"], device=device, dtype=torch.float64
    )
    weights = weights / weights.sum()
    combined = weights @ horizon_gradients
    unit_direction = combined / combined.norm()
    raw_derivatives = horizon_gradients @ (-unit_direction)
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
        eligible = _eligible(before, after) and maximum_delta <= 0.001
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
    selected_rate = max(eligible_rows, key=_selection_key) if eligible_rows else None
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    baseline_parity = _baseline_parity(before, n728["baseline"])
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n728_report": _sha256(args.n728_report),
        "n729_report": _sha256(args.n729_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
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
            and sampling["event_count_minimum"] == 7
            and sampling["event_count_maximum"] == 7
        ),
        "n728_baseline_reproduced": baseline_parity <= 1e-6,
        "common_raw_direction_reproduced": bool(torch.all(raw_derivatives < 0).item()),
        "seven_virtual_rates": len(rows) == 7 and LEARNING_RATES == (
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
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
        "n728_baseline_parity_max_abs_error": baseline_parity,
        "horizon_weights": [float(value) for value in weights.detach().cpu()],
        "combined_gradient_norm": float(combined.norm().detach().cpu()),
        "unit_raw_directional_derivative": {
            str(horizon): float(raw_derivatives[index].detach().cpu())
            for index, horizon in enumerate(HORIZONS)
        },
        "unit_direction_max_abs_component": float(unit_direction.abs().max().detach().cpu()),
        "surface": rows,
        "eligible_candidates": len(eligible_rows),
        "selected_candidate": selected_rate,
        "selection_informative": selected_rate is not None,
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
        raise RuntimeError("raw common-descent surface process gates failed")
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
    parser.add_argument("--n728-report", type=Path, required=True)
    parser.add_argument("--n729-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("raw common-descent surface refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
