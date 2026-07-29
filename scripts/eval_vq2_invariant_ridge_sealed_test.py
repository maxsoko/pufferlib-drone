#!/usr/bin/env python3
"""Perform the single physical N681 sealed-test evaluation for N711."""

from __future__ import annotations

import argparse
import hashlib
import io
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

from scripts.audit_vq2_frozen_ridge_readout import _ridge_predict
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index, _metrics
from scripts.continue_vq2_informed_dreamer_actor_offline import _replay_hashes
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _dense_outputs,
    _fixed_parent_reference,
    _sample_dense_pool,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _drift,
    _model_from_payload,
    _replay_geometry,
    _target_batch,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.package_vq2_invariant_donor_readout import (
    PACKAGE_SCHEMA,
    _continuous_hard_actions,
    _fit_from_auxiliary,
    _strict_action_gates,
    _strict_progress_gates,
)
from scripts.audit_vq2_frozen_ridge_readout import _continuous_features, _fit_sha256
from scripts.train_vq2_informed_dreamer import QuantizedSequenceReplay


TEST_SCHEMA = "vq2_invariant_ridge_physical_sealed_test_v1"
N711_REPORT_SHA256 = (
    "dec6720a51785134f957f91a38f6de9c03b38595d477eb2c2d5ab04c1b91ce03"
)
N711_CHECKPOINT_SHA256 = (
    "92be8898d8294477139f0706a38604d0e3b1bd953a128f9e30c8f9e4c2e168dd"
)
N711_EXECUTABLE_SHA256 = (
    "ab81a0fe663bcc951b71cf88781f3cd30dc05b00d71e3ca08282de441ad8f176"
)
N710_FIT_SHA256 = (
    "bd2b1e32288461ed6051146c036719dc37049d23912bc4ea719b845fec156a36"
)
N587_CHECKPOINT_SHA256 = (
    "e7e331f3835aa73c3aa4c5e14a11131e3defcaff0a2cc3dcb5a8a9a208e5c4d6"
)
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)
SEALED_TEST_SHA256 = (
    "ef43fd778a7fc5daea24a5042174c150d7886e4c0399e535bd7d43c8306aa0d1"
)


def _load_npz_once(path: Path) -> tuple[dict[str, np.ndarray], str, int]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    with np.load(io.BytesIO(payload)) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    return arrays, digest, size


def _n711_contract_gates(
    report: dict[str, object], source_hashes: dict[str, object]
) -> dict[str, bool]:
    report_sources = report.get("source_hashes")
    progress = report.get("strict_progress_gates")
    action = report.get("strict_action_gates")
    process = report.get("process_gates")
    postwrite = report.get("postwrite_gates")
    if not isinstance(report_sources, dict):
        report_sources = {}
    if not isinstance(progress, dict):
        progress = {}
    if not isinstance(action, dict):
        action = {}
    if not isinstance(process, dict):
        process = {}
    if not isinstance(postwrite, dict):
        postwrite = {}
    return {
        "n711_report_hash_exact": source_hashes["n711_report"] == N711_REPORT_SHA256,
        "n711_checkpoint_hash_exact": (
            source_hashes["checkpoint"] == N711_CHECKPOINT_SHA256
            and report.get("output_checkpoint_sha256") == N711_CHECKPOINT_SHA256
        ),
        "n711_executable_hash_exact": (
            source_hashes["n711_executable"] == N711_EXECUTABLE_SHA256
            and report_sources.get("executable") == N711_EXECUTABLE_SHA256
        ),
        "n711_contract_exact": report.get("contract") == PACKAGE_SCHEMA,
        "n711_process_passed": report.get("process_passed") is True,
        "n711_package_admitted": report.get("package_admitted") is True,
        "n711_fit_hash_exact": (
            report.get("fit_sha256") == N710_FIT_SHA256
            and report.get("reloaded_fit_sha256") == N710_FIT_SHA256
        ),
        "n711_progress_gates_exact": bool(progress)
        and all(value is True for value in progress.values()),
        "n711_action_gates_exact": bool(action)
        and all(value is True for value in action.values()),
        "n711_process_gates_exact": bool(process)
        and all(value is True for value in process.values()),
        "n711_postwrite_gates_exact": bool(postwrite)
        and all(value is True for value in postwrite.values()),
        "n711_zero_update_contract": (
            report.get("optimizer_steps") == 0
            and report.get("probe_updates") == 0
            and report.get("actor_updates") == 0
            and report.get("representation_updates") == 0
            and report.get("world_updates") == 0
            and report.get("reward_updates") == 0
            and report.get("critic_updates") == 0
            and report.get("test_dataset_path_received") == 0
            and report.get("test_dataset_opened") == 0
            and report.get("test_rows_scored") == 0
            and report.get("native_environment_created") == 0
            and report.get("flightsim_packets") == 0
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.report.exists():
        raise ValueError("sealed-test evaluation refuses to overwrite its report")
    started = time.perf_counter()
    device = torch.device(args.device)
    n711_executable = ROOT / "scripts/package_vq2_invariant_donor_readout.py"
    preflight_before: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "n711_executable": _sha256(n711_executable),
        "checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "materialization_report": _sha256(args.materialization_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    n711_report = json.loads(args.n711_report.read_text(encoding="utf-8"))
    n711_gates = _n711_contract_gates(n711_report, preflight_before)
    if not all(n711_gates.values()):
        failed = sorted(name for name, passed in n711_gates.items() if not passed)
        raise RuntimeError(f"N711 package contract mismatch before test read: {failed}")
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        preflight_before["parent_checkpoint"] != N587_CHECKPOINT_SHA256
        or preflight_before["materialization_report"] != MATERIALIZATION_SHA256
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("output_sha256", {}).get("test") != SEALED_TEST_SHA256
        or materialization.get("test_rows_scored") != 0
        or preflight_before["dense_replay"]
        != n711_report.get("source_hashes", {}).get("dense_replay")
    ):
        raise RuntimeError("sealed materialization, N587, or N652 preflight mismatch")

    checkpoint_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    parent_payload = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(checkpoint_payload, device)
    parent = _model_from_payload(parent_payload, device)
    model_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    auxiliary = checkpoint_payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("N711 checkpoint lacks fixed ridge auxiliary")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N711 fixed ridge auxiliary hash mismatch")

    # This is the sole filesystem content read of the physical sealed NPZ.
    arrays, test_dataset_sha256, test_dataset_bytes = _load_npz_once(args.test_dataset)
    if test_dataset_sha256 != SEALED_TEST_SHA256:
        raise RuntimeError("physical sealed-test SHA-256 mismatch")
    event_count = len(arrays["vector_step"])
    if event_count != 64:
        raise RuntimeError("physical sealed test must contain exactly 64 events")
    event_ids = np.arange(event_count, dtype=np.int64)
    feature = _continuous_features(
        model, arrays, event_ids,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    prediction = _ridge_predict(
        fit, feature.reshape(-1, feature.shape[-1])
    ).reshape(feature.shape[:2])
    full_length = arrays["mask"].shape[1] - 1
    target = _target_batch(
        arrays, event_ids, np.zeros(event_count, dtype=np.int64),
        sequence_length=full_length,
    )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    local_event, starts = _crop_index(event_count, crop_starts)
    time_index = starts[:, None] + np.arange(args.sequence_length, dtype=np.int64)[None, :]
    test_progress = _metrics(
        target[local_event[:, None], time_index],
        prediction[local_event[:, None], time_index],
        evaluation_indices=np.arange(
            args.burn_in - 1, args.sequence_length, args.evaluation_stride,
            dtype=np.int64,
        ),
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )

    parent_reference = _fixed_parent_reference(
        parent, arrays, event_ids,
        batch_size=args.fixed_reference_batch_size, device=device,
    )
    test_action = _continuous_hard_actions(
        model, arrays, event_ids,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    test_action_drift = _drift(
        test_action, torch.from_numpy(parent_reference["action"])
    )
    capacity, agents = _replay_geometry(args.dense_replay_dir)
    replay = QuantizedSequenceReplay(
        capacity, agents, storage_dir=args.dense_replay_dir, resume=True
    )
    dense_pool = _sample_dense_pool(
        parent, replay,
        pool_size=args.dense_pool_size,
        sequence_length=args.dense_sequence_length,
        seed=args.dense_seed,
        microbatch_size=args.dense_microbatch_size,
        device=device,
    )
    dense_child = _dense_outputs(
        model, dense_pool,
        microbatch_size=args.dense_microbatch_size, device=device,
    )
    dense_action_drift = _drift(
        dense_child["hard_action"], dense_pool["hard_action"]
    )
    dense_parity = max(
        abs(float(dense_action_drift[name]) - float(n711_report["dense_action_drift"][name]))
        for name in ("rmse", "max_abs")
    )

    model_changes = [
        name for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    preflight_after: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "n711_executable": _sha256(n711_executable),
        "checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "materialization_report": _sha256(args.materialization_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    progress_gates = _strict_progress_gates(test_progress)
    action_gates = _strict_action_gates(test_action_drift, dense_action_drift)
    process_gates = {
        "preflight_sources_unchanged": preflight_after == preflight_before,
        "n711_contract_exact": all(n711_gates.values()),
        "sealed_test_sha256_exact": test_dataset_sha256 == SEALED_TEST_SHA256,
        "sealed_test_event_count_exact": event_count == 64,
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "dense_validation_parity": dense_parity <= args.dense_parity_tolerance,
        "model_tensors_exact": model_changes == [],
        "test_metrics_finite": all(
            math.isfinite(float(test_progress[name]))
            for name in ("correlation", "mae_m", "near_plane_mae_m", "direction_accuracy")
        ),
        "single_filesystem_test_read": True,
    }
    passed = (
        all(process_gates.values())
        and all(progress_gates.values())
        and all(action_gates.values())
    )
    report: dict[str, object] = {
        "contract": TEST_SCHEMA,
        "source_hashes": {
            **preflight_after,
            "test_dataset": test_dataset_sha256,
        },
        "n711_contract_gates": n711_gates,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "progress_gates": progress_gates,
        "action_gates": action_gates,
        "passed": passed,
        "test_progress": test_progress,
        "test_action_drift": test_action_drift,
        "dense_action_drift": dense_action_drift,
        "dense_checkpoint_parity_max_abs_error": dense_parity,
        "test_events": event_count,
        "test_groups": int(len(np.unique(arrays["vector_step"]))),
        "test_crops": int(len(local_event)),
        "test_dataset_bytes": test_dataset_bytes,
        "test_dataset_path_received": 1,
        "test_dataset_filesystem_reads": 1,
        "test_dataset_opened": 1,
        "test_evaluations": 1,
        "test_observations_evaluated": int(event_count * full_length),
        "test_privileged_label_sequences_evaluated": event_count,
        "model_tensor_changes": model_changes,
        "optimizer_steps": 0,
        "probe_updates": 0,
        "actor_updates": 0,
        "representation_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "critic_updates": 0,
        "checkpoint_written": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n711-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--dense-replay-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--reference-microbatch-size", type=int, default=1)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--dense-pool-size", type=int, default=128)
    parser.add_argument("--dense-sequence-length", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=671001)
    parser.add_argument("--dense-microbatch-size", type=int, default=4)
    parser.add_argument("--dense-parity-tolerance", type=float, default=1e-9)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("the physical sealed-test source lock requires CPU")
    if args.fixed_reference_batch_size != 416:
        parser.error("fixed reference batch must remain 416")
    if (
        args.dense_pool_size != 128
        or args.dense_sequence_length != 64
        or args.dense_seed != 671001
    ):
        parser.error("dense replay contract must remain exact to N711")
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "reference_microbatch_size",
        "dense_microbatch_size", "dense_parity_tolerance",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
