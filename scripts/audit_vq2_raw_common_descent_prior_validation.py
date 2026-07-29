#!/usr/bin/env python3
"""Held-out validation of the N731 raw common-descent prior child."""

from __future__ import annotations

import argparse
import json
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
from scripts.audit_vq2_multihorizon_prior_validation import _improvement_gates
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.audit_vq2_progress_prior_validation import _progress_deltas
from scripts.audit_vq2_repaired_imagination_readiness import (
    _posterior_parity_error,
    _prior_admission,
)
from scripts.continue_vq2_prior_raw_common_descent_offline import CONTINUATION_SCHEMA
from scripts.continue_vq2_success_prefix_representation_offline import _model_from_payload
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_raw_common_descent_prior_heldout_validation_v1"
N731_CHECKPOINT_SHA256 = (
    "ebc6f0ef1caee3cd53ba489d7440158da243d50ecfc9d353685c1b1dafe9cb1c"
)
N731_REPORT_SHA256 = "9d5841ec6420f15276a49925713a84ddd36037c24f5e497e32eebc03cf818043"
N721_REPORT_SHA256 = "b9ed4b751f4b14e9e1c6092200ecd9d05600b6344aa8d03036469b6d10049c29"
N717_REPORT_SHA256 = "f7605851aeb7accd790f0fbd3fab5fe006acfa8b8d1710cc6b24dbf660787be8"


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n731_report": _sha256(args.n731_report),
        "n721_report": _sha256(args.n721_report),
        "n717_report": _sha256(args.n717_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N731_CHECKPOINT_SHA256,
        "n731_report": N731_REPORT_SHA256,
        "n721_report": N721_REPORT_SHA256,
        "n717_report": N717_REPORT_SHA256,
        "validation_dataset": VALIDATION_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("N732 source hash mismatch")
    n731 = json.loads(args.n731_report.read_text(encoding="utf-8"))
    n721 = json.loads(args.n721_report.read_text(encoding="utf-8"))
    n717 = json.loads(args.n717_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n731.get("contract") != CONTINUATION_SCHEMA
        or n731.get("process_passed") is not True
        or n731.get("prior_updates") != 1
        or n731.get("output_checkpoint_sha256") != N731_CHECKPOINT_SHA256
        or n721.get("process_passed") is not True
        or n717.get("process_passed") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("N732 predecessor contract mismatch")
    baseline_n720 = n721.get("metrics")
    baseline_n716 = n717.get("metrics")
    if not isinstance(baseline_n720, dict) or not isinstance(baseline_n716, dict):
        raise RuntimeError("N732 baseline metrics are missing")
    arrays = _load_validation(args.validation_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict):
        raise RuntimeError("N732 fixed ridge auxiliary is missing")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N732 fixed ridge hash mismatch")
    model = _model_from_payload(payload, device)
    model_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    metrics = _audit_model(
        model,
        arrays,
        horizons=FIXED_HORIZONS,
        burn_in=32,
        stride=4,
        microbatch_size=args.microbatch_size,
        device=device,
        fit=fit,
    )
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    posterior_parity_n720 = _posterior_parity_error(metrics, baseline_n720)
    posterior_parity_n716 = _posterior_parity_error(metrics, baseline_n716)
    deltas_n720 = _progress_deltas(metrics, baseline_n720)
    deltas_n716 = _progress_deltas(metrics, baseline_n716)
    improvement_n720 = _improvement_gates(deltas_n720, suffix="vs_n720")
    improvement_n716 = _improvement_gates(deltas_n716, suffix="vs_n716")
    admission_gates = _prior_admission(metrics)
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n731_report": _sha256(args.n731_report),
        "n721_report": _sha256(args.n721_report),
        "n717_report": _sha256(args.n717_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_tensors_exact": changed == [],
        "posterior_metrics_exact_vs_n720": posterior_parity_n720 <= 1e-9,
        "posterior_metrics_exact_vs_n716": posterior_parity_n716 <= 1e-9,
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "no_train_or_test_path": True,
    }
    process_passed = all(process_gates.values())
    improved_all_n720 = all(improvement_n720.values())
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "improvement_gates_vs_n720": improvement_n720,
        "improvement_gates_vs_n716": improvement_n716,
        "improved_all_horizons_vs_n720": improved_all_n720,
        "improved_all_horizons_vs_n716": all(improvement_n716.values()),
        "prior_admission_gates": admission_gates,
        "prior_admitted": process_passed and improved_all_n720 and all(admission_gates.values()),
        "progress_deltas_vs_n720": deltas_n720,
        "progress_deltas_vs_n716": deltas_n716,
        "posterior_metric_parity_vs_n720_max_abs_error": posterior_parity_n720,
        "posterior_metric_parity_vs_n716_max_abs_error": posterior_parity_n716,
        "metrics": metrics,
        "model_tensor_changes": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "train_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if not process_passed:
        failed = [name for name, passed in process_gates.items() if not passed]
        raise RuntimeError(f"N732 process gates failed: {failed}")
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
    parser.add_argument("--n721-report", type=Path, required=True)
    parser.add_argument("--n717-report", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("N732 refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
