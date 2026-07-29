#!/usr/bin/env python3
"""Read-only held-out validation of N720's one-step progress-prior child."""

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
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.audit_vq2_repaired_imagination_readiness import (
    _posterior_parity_error,
    _prior_admission,
)
from scripts.continue_vq2_prior_progress_offline import CONTINUATION_SCHEMA
from scripts.continue_vq2_success_prefix_representation_offline import _model_from_payload
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    PACKAGE_SCHEMA,
    _fit_from_auxiliary,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_progress_prior_heldout_validation_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N720_REPORT_SHA256 = (
    "8cae6c61eaa96cc4e109d4fc60b500fc075ada33c2b896336f37acbd1f9b8fa8"
)
N717_REPORT_SHA256 = (
    "f7605851aeb7accd790f0fbd3fab5fe006acfa8b8d1710cc6b24dbf660787be8"
)


def _progress_deltas(
    current: dict[str, object], baseline: dict[str, object]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for horizon in FIXED_HORIZONS:
        now = current[str(horizon)]["pre_event"]
        old = baseline[str(horizon)]["pre_event"]
        result[str(horizon)] = {
            "open_vs_posterior_mae_delta": float(now["open_vs_posterior_progress"]["mae"])
            - float(old["open_vs_posterior_progress"]["mae"]),
            "open_physical_mae_delta": float(now["open_prior_progress"]["mae"])
            - float(old["open_prior_progress"]["mae"]),
            "kl_delta": float(now["posterior_to_open_prior_kl"]["mean"])
            - float(old["posterior_to_open_prior_kl"]["mean"]),
            "action_rmse_delta": float(now["actor_action_drift"]["rmse"])
            - float(old["actor_action_drift"]["rmse"]),
        }
    return result


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n720_report": _sha256(args.n720_report),
        "n717_report": _sha256(args.n717_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n720_report": N720_REPORT_SHA256,
        "n717_report": N717_REPORT_SHA256,
        "validation_dataset": VALIDATION_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("N721 source hash mismatch")
    n720 = json.loads(args.n720_report.read_text(encoding="utf-8"))
    n717 = json.loads(args.n717_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n720.get("contract") != CONTINUATION_SCHEMA
        or n720.get("process_passed") is not True
        or n720.get("updates") != 1
        or n717.get("process_passed") is not True
        or n717.get("prior_admitted") is not False
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("N721 predecessor contract mismatch")
    baseline = n717.get("metrics")
    if not isinstance(baseline, dict):
        raise RuntimeError("N717 baseline metrics are missing")
    arrays = _load_validation(args.validation_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("N720 fixed ridge auxiliary is missing")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N720 fixed ridge hash mismatch")
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
    posterior_parity = _posterior_parity_error(metrics, baseline)
    deltas = _progress_deltas(metrics, baseline)
    admission_gates = _prior_admission(metrics)
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n720_report": _sha256(args.n720_report),
        "n717_report": _sha256(args.n717_report),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_tensors_exact": changed == [],
        "posterior_metrics_exact": posterior_parity <= 1e-9,
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "no_train_or_test_path": True,
    }
    improvement_gates = {
        f"h{horizon}_open_vs_posterior_mae_improved": (
            deltas[str(horizon)]["open_vs_posterior_mae_delta"] < 0.0
        )
        for horizon in FIXED_HORIZONS
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "improvement_gates": improvement_gates,
        "improved_all_horizons": all(improvement_gates.values()),
        "prior_admission_gates": admission_gates,
        "prior_admitted": all(process_gates.values()) and all(admission_gates.values()),
        "progress_deltas_vs_n717": deltas,
        "posterior_metric_parity_max_abs_error": posterior_parity,
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
    if not report["process_passed"]:
        raise RuntimeError("N721 process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n720-report", type=Path, required=True)
    parser.add_argument("--n717-report", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("N721 refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
