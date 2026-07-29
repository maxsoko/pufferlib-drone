#!/usr/bin/env python3
"""One-time N668 test evaluation for a validation-admitted calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index, _metrics
from scripts.calibrate_vq2_frozen_representation_actor import (
    CALIBRATION_SCHEMA,
    _actor_drift_from_numpy,
    _actor_drift_from_tensor,
    _dense_feature_corpus,
    _event_feature_corpus,
    _model_from_payload,
    _progress_gates,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _replay_hashes, _sha256
from scripts.continue_vq2_success_prefix_representation_offline import (
    SoftPlaneProbe,
    _crop_full,
    _replay_geometry,
    _target_batch,
)
from scripts.train_vq2_informed_dreamer import QuantizedSequenceReplay


TEST_SCHEMA = "vq2_frozen_representation_n668_test_v1"


def _action_gates(test: dict[str, float], dense: dict[str, float], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "test_rmse_bounded": test["rmse"] <= args.maximum_action_rmse,
        "test_max_bounded": test["max_abs"] <= args.maximum_action_max_abs,
        "dense_rmse_bounded": dense["rmse"] <= args.maximum_action_rmse,
        "dense_max_bounded": dense["max_abs"] <= args.maximum_action_max_abs,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.report.exists():
        raise ValueError("test evaluation refuses to overwrite its report")
    device = torch.device(args.device)
    source_before: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "validation_report": _sha256(args.validation_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if (
        validation_report.get("contract") != CALIBRATION_SCHEMA
        or not validation_report.get("process_passed")
        or not validation_report.get("validation_admitted")
        or validation_report.get("output_checkpoint_sha256") != source_before["checkpoint"]
        or validation_report.get("test_evaluations") != 0
    ):
        raise RuntimeError("N675 validation admission contract mismatch")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    parent_payload = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    parent = _model_from_payload(parent_payload, device)
    auxiliary = payload.get("frozen_representation_calibration_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != CALIBRATION_SCHEMA:
        raise RuntimeError("checkpoint lacks frozen calibration auxiliary")
    probe = SoftPlaneProbe(model.rssm.feature_size, args.probe_hidden_size).to(device)
    probe.load_state_dict(auxiliary["probe_state"])
    probe.eval()
    target_mean = float(auxiliary["target_mean"])
    target_std = float(auxiliary["target_std"])
    with np.load(args.dataset) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    phase_after = np.rint(arrays["phase_after_event"].astype(np.float64) * 6.0).astype(np.int64)
    event_split = _stratified_group_split(
        arrays["vector_step"].astype(np.int64),
        phase_after,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    test_events = np.flatnonzero(event_split == 2)
    feature = _event_feature_corpus(
        model,
        arrays,
        test_events,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    local_event, crop_start = _crop_index(len(test_events), crop_starts)
    offsets = np.arange(args.sequence_length, dtype=np.int64)
    time_index = crop_start[:, None] + offsets[None, :]
    crop_feature = feature["soft"][local_event[:, None], time_index]
    prediction_rows: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(crop_feature), args.evaluation_batch_size):
            prediction = probe(
                torch.from_numpy(crop_feature[offset : offset + args.evaluation_batch_size]).to(device)
            )
            prediction_rows.append(prediction.cpu().numpy() * target_std + target_mean)
    test_original_event = test_events[local_event]
    target = _target_batch(
        arrays,
        test_original_event,
        crop_start,
        sequence_length=args.sequence_length,
    )
    evaluation_indices = np.arange(
        args.burn_in - 1, args.sequence_length, args.evaluation_stride, dtype=np.int64
    )
    progress = _metrics(
        target,
        np.concatenate(prediction_rows),
        evaluation_indices=evaluation_indices,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    test_action_drift = _actor_drift_from_numpy(
        model,
        feature["hard"],
        arrays["action"][test_events, :-1].astype(np.float32),
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    replay_capacity, replay_agents = _replay_geometry(args.dense_replay_dir)
    replay = QuantizedSequenceReplay(
        replay_capacity, replay_agents, storage_dir=args.dense_replay_dir, resume=True
    )
    dense = _dense_feature_corpus(
        model,
        parent,
        replay,
        count=args.dense_train_count + args.dense_validation_count,
        sequence_length=args.dense_sequence_length,
        seed=args.dense_seed,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    dense_validation = slice(args.dense_train_count, None)
    dense_action_drift = _actor_drift_from_tensor(
        model,
        dense["feature"][dense_validation],
        dense["target_action"][dense_validation],
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    dense_parity_error = max(
        abs(
            dense_action_drift[key]
            - validation_report["selected_dense_action_drift"][key]
        )
        for key in dense_action_drift
    )
    progress_gates = _progress_gates(progress, args)
    action_gates = _action_gates(test_action_drift, dense_action_drift, args)
    source_after: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "validation_report": _sha256(args.validation_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "test_partition_nonempty": len(test_events) > 0,
        "dense_checkpoint_parity": dense_parity_error <= args.dense_parity_tolerance,
        "finite_progress": all(
            math.isfinite(float(progress[key]))
            for key in ("correlation", "mae_m", "near_plane_mae_m", "direction_accuracy")
        ),
        "checkpoint_unchanged": _sha256(args.checkpoint) == source_before["checkpoint"],
    }
    passed = all(process_gates.values()) and all(progress_gates.values()) and all(action_gates.values())
    report: dict[str, object] = {
        "contract": TEST_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "test_events": int(len(test_events)),
        "test_groups": int(len(np.unique(arrays["vector_step"][test_events]))),
        "test_crops": int(len(local_event)),
        "test_evaluations": 1,
        "test_observations_evaluated": int(len(test_events) * (arrays["mask"].shape[1] - 1)),
        "test_privileged_label_sequences_evaluated": int(len(test_events)),
        "test_progress": progress,
        "test_action_drift": test_action_drift,
        "dense_action_drift": dense_action_drift,
        "dense_checkpoint_parity_max_abs_error": dense_parity_error,
        "progress_gates": progress_gates,
        "action_gates": action_gates,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "passed": passed,
        "model_updates": 0,
        "probe_updates": 0,
        "actor_updates": 0,
        "representation_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--dense-replay-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=668)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--probe-hidden-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--dense-train-count", type=int, default=256)
    parser.add_argument("--dense-validation-count", type=int, default=128)
    parser.add_argument("--dense-sequence-length", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=675001)
    parser.add_argument("--minimum-validation-correlation", type=float, default=0.8)
    parser.add_argument("--maximum-validation-mae-m", type=float, default=0.5)
    parser.add_argument("--maximum-validation-near-plane-mae-m", type=float, default=0.5)
    parser.add_argument("--minimum-validation-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--maximum-action-rmse", type=float, default=0.001)
    parser.add_argument("--maximum-action-max-abs", type=float, default=0.01)
    parser.add_argument("--dense-parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
