#!/usr/bin/env python3
"""Package N707 with its fixed ridge while leaving every model tensor exact."""

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

from scripts.audit_vq2_event_partition_support import _crop_exposure_counts
from scripts.audit_vq2_frozen_ridge_readout import (
    _continuous_features,
    _fit_sha256,
    _ridge_predict,
    _weighted_standardized_ridge,
)
from scripts.audit_vq2_invariant_donor_ridge import (
    AUDIT_SCHEMA as N710_SCHEMA,
    N707_CHECKPOINT_SHA256,
    N707_REPORT_SHA256,
    _n707_contract_gates,
)
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
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _replay_geometry,
    _target_batch,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import QuantizedSequenceReplay, _save_checkpoint


PACKAGE_SCHEMA = "vq2_invariant_ridge_zero_update_package_v1"
N710_REPORT_SHA256 = (
    "e7e3492fd2cbb8fd9c73658932d1dad081a62fa14dae33dff2a89a50de6d41e9"
)
N710_FIT_SHA256 = (
    "bd2b1e32288461ed6051146c036719dc37049d23912bc4ea719b845fec156a36"
)
N710_EXECUTABLE_SHA256 = (
    "4bee8933597b9765508c5ced63f168a160d9694bba3ed47685451d637c386fdf"
)
N587_CHECKPOINT_SHA256 = (
    "e7e331f3835aa73c3aa4c5e14a11131e3defcaff0a2cc3dcb5a8a9a208e5c4d6"
)
TRAIN_SHA256 = "6bb788c0ae27264fa8a68b75500228d9be3269e80bfa454c8d01f8bb654d91ec"
VALIDATION_SHA256 = (
    "c8ded2bd9191a5c548b3bfc1f0a7142b66e0d75558ff80c883819075a5f48ad1"
)
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)
MINIMUM_CORRELATION = 0.80
MAXIMUM_MAE_M = 0.50
MAXIMUM_NEAR_PLANE_MAE_M = 0.50
MINIMUM_DIRECTION_ACCURACY = 0.75
MAXIMUM_ACTION_RMSE = 0.001
MAXIMUM_ACTION_MAX_ABS = 0.01


def _strict_progress_gates(metrics: dict[str, object]) -> dict[str, bool]:
    return {
        "correlation_at_least_0p80": (
            float(metrics.get("correlation", -math.inf)) >= MINIMUM_CORRELATION
        ),
        "mae_at_most_0p50_m": float(metrics.get("mae_m", math.inf)) <= MAXIMUM_MAE_M,
        "near_plane_mae_at_most_0p50_m": (
            float(metrics.get("near_plane_mae_m", math.inf))
            <= MAXIMUM_NEAR_PLANE_MAE_M
        ),
        "direction_accuracy_at_least_0p75": (
            float(metrics.get("direction_accuracy", -math.inf))
            >= MINIMUM_DIRECTION_ACCURACY
        ),
    }


def _strict_action_gates(
    validation: dict[str, float], dense: dict[str, float]
) -> dict[str, bool]:
    return {
        "n681_validation_rmse_at_most_0p001": validation["rmse"] <= MAXIMUM_ACTION_RMSE,
        "n681_validation_max_at_most_0p01": validation["max_abs"] <= MAXIMUM_ACTION_MAX_ABS,
        "n652_dense_rmse_at_most_0p001": dense["rmse"] <= MAXIMUM_ACTION_RMSE,
        "n652_dense_max_at_most_0p01": dense["max_abs"] <= MAXIMUM_ACTION_MAX_ABS,
    }


def _metric_parity_error(
    actual: dict[str, object], expected: dict[str, object]
) -> float:
    keys = (
        "correlation",
        "mae_m",
        "near_plane_mae_m",
        "direction_accuracy",
        "rmse_m",
        "near_plane_correlation",
        "strictly_decreasing_crop_fraction",
        "final_is_minimum_crop_fraction",
    )
    return max(abs(float(actual[key]) - float(expected[key])) for key in keys)


def _fit_auxiliary(
    fit: dict[str, np.ndarray | float],
    *,
    fit_sha256: str,
    covered: np.ndarray,
    source_hashes: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": PACKAGE_SCHEMA,
        "feature_mean": torch.from_numpy(np.asarray(fit["feature_mean"]).copy()),
        "feature_std": torch.from_numpy(np.asarray(fit["feature_std"]).copy()),
        "active": torch.from_numpy(np.asarray(fit["active"], dtype=np.bool_).copy()),
        "target_mean": float(fit["target_mean"]),
        "target_std": float(fit["target_std"]),
        "coefficient": torch.from_numpy(np.asarray(fit["coefficient"]).copy()),
        "fit_sha256": fit_sha256,
        "ridge": 0.001,
        "minimum_feature_std": 1e-6,
        "near_plane_multiplier": 2.0,
        "covered_history_indices": torch.from_numpy(covered.astype(np.int64, copy=True)),
        "model_updates": 0,
        "source_hashes": source_hashes,
    }


def _fit_from_auxiliary(auxiliary: dict[str, object]) -> dict[str, np.ndarray | float]:
    return {
        "feature_mean": auxiliary["feature_mean"].cpu().numpy(),
        "feature_std": auxiliary["feature_std"].cpu().numpy(),
        "active": auxiliary["active"].cpu().numpy(),
        "target_mean": float(auxiliary["target_mean"]),
        "target_std": float(auxiliary["target_std"]),
        "coefficient": auxiliary["coefficient"].cpu().numpy(),
    }


@torch.no_grad()
def _continuous_hard_actions(
    model,
    arrays: dict[str, np.ndarray],
    event_ids: np.ndarray,
    *,
    microbatch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    full_length = arrays["mask"].shape[1] - 1
    model.eval()
    for offset in range(0, len(event_ids), microbatch_size):
        chosen = event_ids[offset : offset + microbatch_size]
        legal, action = _legal_action_batch(
            arrays,
            chosen,
            np.zeros(len(chosen), dtype=np.int64),
            sequence_length=full_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model,
            legal,
            action,
            _initial_state(arrays, chosen, device=device),
        )
        rows.append(
            model.deterministic_actor_action(
                model.actor_distribution(state.features)
            ).float().cpu()
        )
    return torch.cat(rows)


def _n710_contract_gates(
    report: dict[str, object], source_hashes: dict[str, object]
) -> dict[str, bool]:
    report_sources = report.get("source_hashes")
    admission = report.get("admission_gates")
    process = report.get("process_gates")
    if not isinstance(report_sources, dict):
        report_sources = {}
    if not isinstance(admission, dict):
        admission = {}
    if not isinstance(process, dict):
        process = {}
    return {
        "n710_report_hash_exact": source_hashes["n710_report"] == N710_REPORT_SHA256,
        "n710_executable_hash_exact": (
            source_hashes["n710_executable"] == N710_EXECUTABLE_SHA256
            and report_sources.get("executable") == N710_EXECUTABLE_SHA256
        ),
        "n710_contract_exact": report.get("contract") == N710_SCHEMA,
        "n710_process_passed": report.get("process_passed") is True,
        "n710_readout_admitted": report.get("readout_admitted") is True,
        "n710_fit_hash_exact": report.get("fit_sha256") == N710_FIT_SHA256,
        "n710_admission_gates_exact": bool(admission)
        and all(value is True for value in admission.values()),
        "n710_process_gates_exact": bool(process)
        and all(value is True for value in process.values()),
        "n710_donor_sources_exact": (
            report_sources.get("checkpoint") == N707_CHECKPOINT_SHA256
            and report_sources.get("n707_report") == N707_REPORT_SHA256
            and report_sources.get("train_dataset") == TRAIN_SHA256
            and report_sources.get("validation_dataset") == VALIDATION_SHA256
            and report_sources.get("materialization_report") == MATERIALIZATION_SHA256
        ),
        "n710_isolation_exact": (
            report.get("optimizer_steps") == 0
            and report.get("checkpoint_written") == 0
            and report.get("test_dataset_path_received") == 0
            and report.get("test_dataset_opened") == 0
            and report.get("test_rows_scored") == 0
            and report.get("native_environment_created") == 0
            and report.get("flightsim_packets") == 0
        ),
    }


def package(args: argparse.Namespace) -> dict[str, object]:
    if args.output_checkpoint.exists() or args.report.exists():
        raise ValueError("N711 packaging refuses to overwrite outputs")
    if args.output_checkpoint.resolve() == args.donor_checkpoint.resolve():
        raise ValueError("N711 cannot overwrite N707")
    started = time.perf_counter()
    device = torch.device(args.device)
    n710_executable = ROOT / "scripts/audit_vq2_invariant_donor_ridge.py"
    source_before: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "n710_executable": _sha256(n710_executable),
        "donor_checkpoint": _sha256(args.donor_checkpoint),
        "donor_report": _sha256(args.donor_report),
        "n710_report": _sha256(args.n710_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    donor_report = json.loads(args.donor_report.read_text(encoding="utf-8"))
    n707_gates = _n707_contract_gates(
        donor_report,
        {
            "n707_report": str(source_before["donor_report"]),
            "checkpoint": str(source_before["donor_checkpoint"]),
        },
    )
    if not all(n707_gates.values()):
        failed = sorted(name for name, passed in n707_gates.items() if not passed)
        raise RuntimeError(f"N707 donor contract mismatch: {failed}")
    n710_report = json.loads(args.n710_report.read_text(encoding="utf-8"))
    n710_gates = _n710_contract_gates(n710_report, source_before)
    if not all(n710_gates.values()):
        failed = sorted(name for name, passed in n710_gates.items() if not passed)
        raise RuntimeError(f"N710 readout contract mismatch: {failed}")
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        source_before["parent_checkpoint"] != N587_CHECKPOINT_SHA256
        or source_before["train_dataset"] != TRAIN_SHA256
        or source_before["validation_dataset"] != VALIDATION_SHA256
        or source_before["materialization_report"] != MATERIALIZATION_SHA256
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("output_sha256", {}).get("train") != TRAIN_SHA256
        or materialization.get("output_sha256", {}).get("validation")
        != VALIDATION_SHA256
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("N681 materialization or N587 contract mismatch")
    expected_replay = donor_report.get("source_hashes", {}).get("dense_replay")
    if source_before["dense_replay"] != expected_replay:
        raise RuntimeError("N652 replay contract mismatch")

    with np.load(args.train_dataset) as loaded:
        train = {name: loaded[name].copy() for name in loaded.files}
    with np.load(args.validation_dataset) as loaded:
        validation = {name: loaded[name].copy() for name in loaded.files}
    if set(train) != set(validation):
        raise RuntimeError("N681 train and validation schemas differ")
    arrays = {name: np.concatenate((train[name], validation[name]), 0) for name in train}
    train_events = np.arange(len(train["vector_step"]), dtype=np.int64)
    validation_events = np.arange(len(train_events), len(arrays["vector_step"]), dtype=np.int64)
    allowed_events = np.arange(len(arrays["vector_step"]), dtype=np.int64)

    donor_payload = torch.load(args.donor_checkpoint, map_location=device, weights_only=False)
    parent_payload = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    donor = _model_from_payload(donor_payload, device)
    parent = _model_from_payload(parent_payload, device)
    donor_state_before = {
        name: value.detach().cpu().clone() for name, value in donor.state_dict().items()
    }

    train_feature = _continuous_features(
        donor, arrays, train_events,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    validation_feature = _continuous_features(
        donor, arrays, validation_events,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    full_length = arrays["mask"].shape[1] - 1
    train_target = _target_batch(
        arrays, train_events, np.zeros(len(train_events), dtype=np.int64),
        sequence_length=full_length,
    )
    validation_target = _target_batch(
        arrays, validation_events, np.zeros(len(validation_events), dtype=np.int64),
        sequence_length=full_length,
    )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    exposure = _crop_exposure_counts(
        arrays["mask"].shape[1], crop_starts,
        sequence_length=args.sequence_length, burn_in=args.burn_in,
    )
    covered = np.flatnonzero(exposure > 0)
    covered = covered[covered < full_length]
    fit_feature = train_feature[:, covered].reshape(-1, train_feature.shape[-1])
    fit_target = train_target[:, covered].reshape(-1)
    fit_weight = np.where(
        np.abs(fit_target) <= args.near_plane_m, args.near_plane_multiplier, 1.0
    ).astype(np.float64)
    fit_one, fit_summary = _weighted_standardized_ridge(
        fit_feature, fit_target, fit_weight,
        ridge=args.ridge, minimum_feature_std=args.minimum_feature_std,
    )
    fit_two, _ = _weighted_standardized_ridge(
        fit_feature, fit_target, fit_weight,
        ridge=args.ridge, minimum_feature_std=args.minimum_feature_std,
    )
    fit_hash_one = _fit_sha256(fit_one)
    fit_hash_two = _fit_sha256(fit_two)
    validation_prediction = _ridge_predict(
        fit_one, validation_feature.reshape(-1, validation_feature.shape[-1])
    ).reshape(validation_feature.shape[:2])
    local_event, starts = _crop_index(len(validation_events), crop_starts)
    time_index = starts[:, None] + np.arange(args.sequence_length, dtype=np.int64)[None, :]
    validation_metrics = _metrics(
        validation_target[local_event[:, None], time_index],
        validation_prediction[local_event[:, None], time_index],
        evaluation_indices=np.arange(
            args.burn_in - 1, args.sequence_length, args.evaluation_stride,
            dtype=np.int64,
        ),
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    validation_metric_parity = _metric_parity_error(
        validation_metrics, n710_report["validation_progress"]
    )

    parent_reference = _fixed_parent_reference(
        parent, arrays, allowed_events,
        batch_size=args.fixed_reference_batch_size, device=device,
    )
    validation_action = _continuous_hard_actions(
        donor, arrays, validation_events,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    validation_action_drift = _drift(
        validation_action,
        torch.from_numpy(parent_reference["action"][validation_events]),
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
        donor, dense_pool,
        microbatch_size=args.dense_microbatch_size, device=device,
    )
    dense_action_drift = _drift(dense_child["hard_action"], dense_pool["hard_action"])

    source_after: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "n710_executable": _sha256(n710_executable),
        "donor_checkpoint": _sha256(args.donor_checkpoint),
        "donor_report": _sha256(args.donor_report),
        "n710_report": _sha256(args.n710_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    model_changes_before_write = [
        name for name, value in donor.state_dict().items()
        if not torch.equal(value.detach().cpu(), donor_state_before[name])
    ]
    strict_progress = _strict_progress_gates(validation_metrics)
    strict_action = _strict_action_gates(validation_action_drift, dense_action_drift)
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "n707_contract_exact": all(n707_gates.values()),
        "n710_contract_exact": all(n710_gates.values()),
        "fit_hash_exact": fit_hash_one == N710_FIT_SHA256,
        "fit_deterministic": fit_hash_one == fit_hash_two,
        "n710_metric_parity": validation_metric_parity <= args.metric_parity_tolerance,
        "all_512_features_active": fit_summary["active_features"] == 512,
        "exact_train_fit_rows": fit_summary["rows"] == 78336,
        "covered_history_exact": bool(
            len(covered) == 288 and int(covered[0]) == 32 and int(covered[-1]) == 319
        ),
        "model_tensors_exact_before_write": model_changes_before_write == [],
        "no_test_path": True,
    }
    package_admitted_before_write = (
        all(process_gates.values())
        and all(strict_progress.values())
        and all(strict_action.values())
    )
    output_model_changes: list[str] = []
    output_model_missing: list[str] = []
    output_checkpoint_sha256: str | None = None
    reloaded_fit_hash: str | None = None
    checkpoint_written = 0
    if package_admitted_before_write:
        output_payload = dict(donor_payload)
        auxiliary_sources = {
            "donor_checkpoint": source_after["donor_checkpoint"],
            "donor_report": source_after["donor_report"],
            "n710_report": source_after["n710_report"],
            "train_dataset": source_after["train_dataset"],
            "validation_dataset": source_after["validation_dataset"],
            "materialization_report": source_after["materialization_report"],
            "dense_replay": source_after["dense_replay"],
        }
        output_payload["frozen_invariant_ridge_aux"] = _fit_auxiliary(
            fit_one,
            fit_sha256=fit_hash_one,
            covered=covered,
            source_hashes=auxiliary_sources,
        )
        _save_checkpoint(args.output_checkpoint, output_payload)
        checkpoint_written = 1
        output_checkpoint_sha256 = _sha256(args.output_checkpoint)
        reloaded = torch.load(args.output_checkpoint, map_location="cpu", weights_only=False)
        donor_model_payload = donor_payload.get("model")
        output_model_payload = reloaded.get("model")
        if not isinstance(donor_model_payload, dict) or not isinstance(output_model_payload, dict):
            raise RuntimeError("packaged checkpoint model payload is invalid")
        output_model_changes = [
            name for name, value in output_model_payload.items()
            if name not in donor_model_payload
            or not torch.equal(value.cpu(), donor_model_payload[name].cpu())
        ]
        output_model_missing = sorted(set(donor_model_payload) - set(output_model_payload))
        reloaded_aux = reloaded.get("frozen_invariant_ridge_aux")
        if not isinstance(reloaded_aux, dict):
            raise RuntimeError("packaged checkpoint lacks ridge auxiliary")
        reloaded_fit_hash = _fit_sha256(_fit_from_auxiliary(reloaded_aux))
        postwrite_gates = {
            "output_model_tensors_exact": (
                output_model_changes == [] and output_model_missing == []
            ),
            "output_actor_tensors_exact": not any(
                name.startswith("actor.") for name in output_model_changes
            ),
            "output_ridge_hash_exact": reloaded_fit_hash == N710_FIT_SHA256,
            "output_aux_schema_exact": reloaded_aux.get("schema") == PACKAGE_SCHEMA,
        }
    else:
        postwrite_gates = {
            "output_skipped_after_failed_prewrite_admission": (
                not args.output_checkpoint.exists()
            )
        }
    package_admitted = (
        package_admitted_before_write
        and checkpoint_written == 1
        and all(postwrite_gates.values())
    )
    report: dict[str, object] = {
        "contract": PACKAGE_SCHEMA,
        "source_hashes": source_after,
        "n707_contract_gates": n707_gates,
        "n710_contract_gates": n710_gates,
        "process_gates": process_gates,
        "postwrite_gates": postwrite_gates,
        "process_passed": all(process_gates.values()) and all(postwrite_gates.values()),
        "strict_progress_gates": strict_progress,
        "strict_action_gates": strict_action,
        "package_admitted": package_admitted,
        "fit": fit_summary,
        "fit_sha256": fit_hash_one,
        "reloaded_fit_sha256": reloaded_fit_hash,
        "covered_history_indices": covered.tolist(),
        "validation_progress": validation_metrics,
        "n710_validation_metric_parity_max_abs_error": validation_metric_parity,
        "validation_action_drift": validation_action_drift,
        "dense_action_drift": dense_action_drift,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": output_checkpoint_sha256,
        "model_tensor_changes_before_write": model_changes_before_write,
        "output_model_tensor_changes": output_model_changes,
        "output_model_tensor_missing": output_model_missing,
        "optimizer_steps": 0,
        "probe_updates": 0,
        "actor_updates": 0,
        "representation_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "critic_updates": 0,
        "checkpoint_written": checkpoint_written,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
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
    parser.add_argument("--donor-checkpoint", type=Path, required=True)
    parser.add_argument("--donor-report", type=Path, required=True)
    parser.add_argument("--n710-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--dense-replay-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--minimum-feature-std", type=float, default=1e-6)
    parser.add_argument("--reference-microbatch-size", type=int, default=1)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--dense-pool-size", type=int, default=128)
    parser.add_argument("--dense-sequence-length", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=671001)
    parser.add_argument("--dense-microbatch-size", type=int, default=4)
    parser.add_argument("--metric-parity-tolerance", type=float, default=1e-9)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("N711 source lock requires CPU")
    if args.ridge != 0.001 or args.near_plane_multiplier != 2.0:
        parser.error("N711 ridge and near-plane multiplier are fixed")
    if args.fixed_reference_batch_size != 416:
        parser.error("N711 fixed reference batch must remain 416")
    if (
        args.dense_pool_size != 128
        or args.dense_sequence_length != 64
        or args.dense_seed != 671001
    ):
        parser.error("N711 dense pool contract is fixed to N707")
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "reference_microbatch_size",
        "dense_microbatch_size", "metric_parity_tolerance",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    return args


if __name__ == "__main__":
    print(json.dumps(package(parse_args()), indent=2, sort_keys=True))
