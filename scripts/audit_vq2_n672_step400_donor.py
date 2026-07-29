#!/usr/bin/env python3
"""Audit an export-only deterministic reconstruction of N672 step 400."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.continue_vq2_success_prefix_representation_offline import (
    _changed_model_keys,
)
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    JOINT_SCHEMA,
)


AUDIT_SCHEMA = "vq2_n672_step400_donor_export_audit_v1"


def _history_numeric_differences(
    reference: list[dict], reconstructed: list[dict]
) -> dict[str, float | int]:
    if len(reference) != len(reconstructed):
        raise RuntimeError("validation history length mismatch")
    maximum = 0.0
    compared = 0

    def compare(left, right, path: str) -> None:
        nonlocal maximum, compared
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                raise RuntimeError(f"validation history structure mismatch at {path}")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, bool):
            if right is not left:
                raise RuntimeError(f"validation history boolean mismatch at {path}")
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isfinite(float(left)) or not math.isfinite(float(right)):
                raise RuntimeError(f"non-finite validation value at {path}")
            maximum = max(maximum, abs(float(left) - float(right)))
            compared += 1
            return
        if left != right:
            raise RuntimeError(f"validation history value mismatch at {path}")

    for index, (left, right) in enumerate(zip(reference, reconstructed, strict=True)):
        compare(left, right, f"history[{index}]")
    return {"maximum_absolute_error": maximum, "numeric_values_compared": compared}


def audit(args: argparse.Namespace) -> dict[str, object]:
    if args.report.exists():
        raise ValueError("donor audit refuses to overwrite its report")
    reference_sha = _sha256(args.reference_report)
    training_sha = _sha256(args.training_report)
    donor_sha = _sha256(args.donor_checkpoint)
    parent_sha = _sha256(args.parent_checkpoint)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    if reference.get("contract") != JOINT_SCHEMA:
        raise RuntimeError("reference report contract mismatch")
    if training.get("contract") != JOINT_SCHEMA:
        raise RuntimeError("reconstruction report contract mismatch")
    trajectory_fields = (
        "step",
        "validation_progress",
        "validation_action_drift",
        "dense_drift",
    )
    reference_trajectory = [
        {field: row[field] for field in trajectory_fields}
        for row in reference.get("validation_history", [])
    ]
    training_trajectory = [
        {field: row[field] for field in trajectory_fields}
        for row in training.get("validation_history", [])
    ]
    history_difference = _history_numeric_differences(
        reference_trajectory,
        training_trajectory,
    )
    parent_payload = torch.load(
        args.parent_checkpoint, map_location="cpu", weights_only=False
    )
    donor_payload = torch.load(
        args.donor_checkpoint, map_location="cpu", weights_only=False
    )
    changed, forbidden = _changed_model_keys(
        parent_payload["model"], donor_payload["model"]
    )
    auxiliary = donor_payload.get("success_prefix_joint_representation_aux")
    if not isinstance(auxiliary, dict):
        raise RuntimeError("donor lacks joint representation auxiliary")
    final_reference = reference["validation_history"][-1]
    selected_training = training["selected_validation_progress"]
    final_metric_error = max(
        abs(
            float(final_reference["validation_progress"][key])
            - float(selected_training[key])
        )
        for key in final_reference["validation_progress"]
        if isinstance(final_reference["validation_progress"][key], (int, float))
    )
    gates = {
        "reference_is_rejected_n672": (
            reference.get("selected_step") == 150
            and reference.get("validation_admitted") is False
            and reference.get("process_passed") is True
        ),
        "source_hashes_reproduce": (
            training.get("source_hashes") == reference.get("source_hashes")
        ),
        "all_eight_checkpoints_reproduce": (
            len(reference.get("validation_history", [])) == 8
            and len(training.get("validation_history", [])) == 8
            and history_difference["maximum_absolute_error"]
            <= args.numeric_tolerance
        ),
        "training_selects_step400": training.get("selected_step") == 400,
        "auxiliary_identifies_step400": (
            auxiliary.get("schema") == JOINT_SCHEMA
            and auxiliary.get("selected_step") == 400
            and auxiliary.get("representation_updates") == 400
        ),
        "selected_metrics_are_final_metrics": (
            final_metric_error <= args.numeric_tolerance
        ),
        "donor_hash_matches_training_report": (
            training.get("output_checkpoint_sha256") == donor_sha
        ),
        "training_process_passed": training.get("process_passed") is True,
        "donor_remains_nonadmitted": training.get("validation_admitted") is False,
        "test_not_evaluated": (
            training.get("test_evaluations") == 0
            and training.get("test_observations_evaluated") == 0
            and training.get("test_privileged_labels_evaluated") == 0
        ),
        "no_native_or_live_activity": (
            training.get("native_environment_created") == 0
            and training.get("collected_transitions") == 0
            and training.get("flightsim_packets") == 0
        ),
        "only_allowed_model_tensors_change": bool(changed) and not forbidden,
        "actor_weights_exact": not any(name.startswith("actor.") for name in changed),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": {
            "executable": _sha256(Path(__file__)),
            "reference_report": reference_sha,
            "training_report": training_sha,
            "parent_checkpoint": parent_sha,
            "donor_checkpoint": donor_sha,
            "training_executable": training.get("source_hashes", {}).get("executable"),
        },
        "reference_selected_step": reference.get("selected_step"),
        "donor_selected_step": training.get("selected_step"),
        "history_difference": history_difference,
        "selected_final_metric_max_abs_error": final_metric_error,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "gates": gates,
        "passed": all(gates.values()),
        "optimizer_steps": 0,
        "model_updates": 0,
        "test_evaluations": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    if not report["passed"]:
        raise RuntimeError("N672 donor export audit failed")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--donor-checkpoint", type=Path, required=True)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-7)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.numeric_tolerance < 0.0:
        parser.error("numeric tolerance cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
