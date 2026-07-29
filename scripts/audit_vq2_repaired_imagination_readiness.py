#!/usr/bin/env python3
"""Validate N716's repaired prior without updating or saving a model."""

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
from scripts.audit_vq2_imagination_readiness import (
    FIXED_HORIZONS,
    MATERIALIZATION_SHA256,
    VALIDATION_SHA256,
    _audit_model,
    _load_validation,
)
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import CONTINUATION_SCHEMA
from scripts.continue_vq2_success_prefix_representation_offline import _model_from_payload
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    PACKAGE_SCHEMA,
    _fit_from_auxiliary,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_repaired_imagination_readiness_audit_v1"
N716_CHECKPOINT_SHA256 = (
    "0a0a3af0fe9846c1a05001c1d8fabed8ec9b4f82237be2519aca47307ab4208f"
)
N716_REPORT_SHA256 = (
    "e00494ee63bb6fdd6f0842b3a94e202e4a327e7265c0f13da31cc4a13215f492"
)
N714_REPORT_SHA256 = (
    "647bf221f375d44c9c70b706db6603a94c913d935389a53737ac1db9e97fc4a0"
)


def _prior_admission(metrics: dict[str, object]) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for horizon in FIXED_HORIZONS:
        row = metrics[str(horizon)]["pre_event"]
        prefix = f"h{horizon}"
        gates[f"{prefix}_kl_at_most_0p01"] = (
            float(row["posterior_to_open_prior_kl"]["mean"]) <= 0.01
        )
        gates[f"{prefix}_agreement_at_least_0p80"] = (
            float(row["categorical_argmax_agreement"]["mean"]) >= 0.80
        )
        gates[f"{prefix}_action_rmse_at_most_0p001"] = (
            float(row["actor_action_drift"]["rmse"]) <= 0.001
        )
        gates[f"{prefix}_action_max_at_most_0p01"] = (
            float(row["actor_action_drift"]["max_abs"]) <= 0.01
        )
        gates[f"{prefix}_progress_mae_within_0p10"] = (
            float(row["open_prior_progress"]["mae"])
            <= float(row["posterior_progress"]["mae"]) + 0.10
        )
        gates[f"{prefix}_open_vs_posterior_progress_mae_at_most_0p10"] = (
            float(row["open_vs_posterior_progress"]["mae"]) <= 0.10
        )
    return gates


def _posterior_parity_error(
    repaired: dict[str, object], baseline: dict[str, object]
) -> float:
    errors: list[float] = []
    for horizon in FIXED_HORIZONS:
        current = repaired[str(horizon)]["pre_event"]
        old = baseline[str(horizon)]["pre_event"]
        for section, keys in (
            ("posterior_progress", ("mae", "rmse", "bias", "correlation")),
            ("posterior_reward", ("mae", "rmse", "bias", "correlation")),
            ("posterior_continuation", ("bce", "mae", "accuracy", "probability_mean")),
        ):
            errors.extend(abs(float(current[section][key]) - float(old[section][key])) for key in keys)
    return max(errors)


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n714_report": _sha256(args.n714_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N716_CHECKPOINT_SHA256,
        "n716_report": N716_REPORT_SHA256,
        "n714_report": N714_REPORT_SHA256,
        "validation_dataset": VALIDATION_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("N717 source hash mismatch")
    n716 = json.loads(args.n716_report.read_text(encoding="utf-8"))
    n714 = json.loads(args.n714_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n716.get("contract") != CONTINUATION_SCHEMA
        or n716.get("process_passed") is not True
        or n716.get("updates") != 357
        or n714.get("process_passed") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("N717 predecessor contract mismatch")
    arrays = _load_validation(args.validation_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("N716 fixed ridge auxiliary is missing")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N716 fixed ridge hash mismatch")
    model = _model_from_payload(payload, device)
    model_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    metrics = _audit_model(
        model, arrays, horizons=FIXED_HORIZONS, burn_in=32, stride=4,
        microbatch_size=args.microbatch_size, device=device, fit=fit,
    )
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    baseline = n714.get("n711")
    if not isinstance(baseline, dict):
        raise RuntimeError("N714 lacks N711 baseline metrics")
    parity_error = _posterior_parity_error(metrics, baseline)
    admission_gates = _prior_admission(metrics)
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n714_report": _sha256(args.n714_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_tensors_exact": changed == [],
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "posterior_metrics_exact": parity_error <= 1e-9,
        "fixed_horizons_exact": FIXED_HORIZONS == (1, 2, 4, 8, 16),
        "no_train_or_test_path": True,
    }
    event_reward_rmse = float(metrics["1"]["event_endpoint"]["open_prior_reward"]["rmse"])
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "prior_admission_gates": admission_gates,
        "prior_admitted": all(process_gates.values()) and all(admission_gates.values()),
        "posterior_metric_parity_max_abs_error": parity_error,
        "reward_event_recalibration_required": event_reward_rmse > 1.0,
        "horizon1_event_open_prior_reward_rmse": event_reward_rmse,
        "metrics": metrics,
        "model_tensor_changes": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "prior_updates": 0,
        "reward_updates": 0,
        "train_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if not report["process_passed"]:
        raise RuntimeError("N717 process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n716-report", type=Path, required=True)
    parser.add_argument("--n714-report", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("N717 refuses to overwrite its report")
    if args.microbatch_size <= 0:
        parser.error("microbatch size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
