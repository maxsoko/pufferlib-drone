#!/usr/bin/env python3
"""Held-out validation of the N734 recomputed raw prior child."""

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
from scripts.continue_vq2_prior_raw_recomputed_offline import CONTINUATION_SCHEMA
from scripts.continue_vq2_success_prefix_representation_offline import _model_from_payload
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_raw_recomputed_prior_heldout_validation_v1"
N734_CHECKPOINT_SHA256 = (
    "406229b85458d79478b95d3389982683ae9ede9a893e1be94a965b709fdae721"
)
N734_REPORT_SHA256 = "6a83430ba5299a93dfe008f626ca7fbaaf44482c805a47a3d03e2bb061b8c5fc"
N732_REPORT_SHA256 = "2b06b237c960be7e207b7dc623fb5b254d29504959ce6032ce81b344e54ef9cc"
N721_REPORT_SHA256 = "b9ed4b751f4b14e9e1c6092200ecd9d05600b6344aa8d03036469b6d10049c29"


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n734_report": _sha256(args.n734_report),
        "n732_report": _sha256(args.n732_report),
        "n721_report": _sha256(args.n721_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N734_CHECKPOINT_SHA256,
        "n734_report": N734_REPORT_SHA256,
        "n732_report": N732_REPORT_SHA256,
        "n721_report": N721_REPORT_SHA256,
        "validation_dataset": VALIDATION_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("N735 source hash mismatch")
    n734 = json.loads(args.n734_report.read_text(encoding="utf-8"))
    n732 = json.loads(args.n732_report.read_text(encoding="utf-8"))
    n721 = json.loads(args.n721_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n734.get("contract") != CONTINUATION_SCHEMA
        or n734.get("process_passed") is not True
        or n734.get("prior_updates") != 1
        or n734.get("output_checkpoint_sha256") != N734_CHECKPOINT_SHA256
        or n732.get("process_passed") is not True
        or n732.get("improved_all_horizons_vs_n720") is not True
        or n721.get("process_passed") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("N735 predecessor contract mismatch")
    baseline_n731 = n732.get("metrics")
    baseline_n720 = n721.get("metrics")
    if not isinstance(baseline_n731, dict) or not isinstance(baseline_n720, dict):
        raise RuntimeError("N735 baseline metrics are missing")
    arrays = _load_validation(args.validation_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict):
        raise RuntimeError("N735 fixed ridge auxiliary is missing")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N735 fixed ridge hash mismatch")
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
    posterior_parity_n731 = _posterior_parity_error(metrics, baseline_n731)
    posterior_parity_n720 = _posterior_parity_error(metrics, baseline_n720)
    deltas_n731 = _progress_deltas(metrics, baseline_n731)
    deltas_n720 = _progress_deltas(metrics, baseline_n720)
    improvement_n731 = _improvement_gates(deltas_n731, suffix="vs_n731")
    improvement_n720 = _improvement_gates(deltas_n720, suffix="vs_n720")
    admission_gates = _prior_admission(metrics)
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n734_report": _sha256(args.n734_report),
        "n732_report": _sha256(args.n732_report),
        "n721_report": _sha256(args.n721_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_tensors_exact": changed == [],
        "posterior_metrics_exact_vs_n731": posterior_parity_n731 <= 1e-9,
        "posterior_metrics_exact_vs_n720": posterior_parity_n720 <= 1e-9,
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "no_train_or_test_path": True,
    }
    process_passed = all(process_gates.values())
    improved_all_n731 = all(improvement_n731.values())
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "improvement_gates_vs_n731": improvement_n731,
        "improvement_gates_vs_n720": improvement_n720,
        "improved_all_horizons_vs_n731": improved_all_n731,
        "improved_all_horizons_vs_n720": all(improvement_n720.values()),
        "prior_admission_gates": admission_gates,
        "prior_admitted": process_passed and improved_all_n731 and all(admission_gates.values()),
        "progress_deltas_vs_n731": deltas_n731,
        "progress_deltas_vs_n720": deltas_n720,
        "posterior_metric_parity_vs_n731_max_abs_error": posterior_parity_n731,
        "posterior_metric_parity_vs_n720_max_abs_error": posterior_parity_n720,
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
        raise RuntimeError(f"N735 process gates failed: {failed}")
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
    parser.add_argument("--n734-report", type=Path, required=True)
    parser.add_argument("--n732-report", type=Path, required=True)
    parser.add_argument("--n721-report", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("N735 refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
