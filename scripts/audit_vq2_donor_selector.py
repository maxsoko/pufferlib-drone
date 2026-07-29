#!/usr/bin/env python3
"""Apply the progress-gated donor selector to an immutable validation history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    JOINT_SCHEMA,
    _candidate_key,
)


SELECTOR_AUDIT_SCHEMA = "vq2_joint_donor_progress_selector_audit_v1"


def _select_candidate(history: list[dict[str, object]], args: argparse.Namespace) -> dict:
    if not history:
        raise ValueError("validation history is empty")
    return max(history, key=lambda candidate: _candidate_key(candidate, args))


def audit(args: argparse.Namespace) -> dict[str, object]:
    source_before = _sha256(args.source_report)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if (
        source.get("contract") != JOINT_SCHEMA
        or not source.get("process_passed")
        or source.get("validation_admitted") is not False
        or source.get("test_evaluations") != 0
        or source.get("sealed_test_dataset_opened") != 0
    ):
        raise RuntimeError("source donor rejection contract mismatch")
    history = source.get("validation_history")
    if not isinstance(history, list):
        raise RuntimeError("source donor lacks validation history")
    selected = _select_candidate(history, args)
    metrics = selected["validation_progress"]
    preservation = selected["preservation_gates"]
    progress_gates = {
        "correlation_at_least_minimum": (
            metrics["correlation"] >= args.minimum_validation_correlation
        ),
        "mae_at_most_maximum": (
            metrics["mae_m"] <= args.maximum_validation_mae_m
        ),
        "near_plane_mae_at_most_maximum": (
            metrics["near_plane_mae_m"]
            <= args.maximum_validation_near_plane_mae_m
        ),
        "direction_at_least_minimum": (
            metrics["direction_accuracy"]
            >= args.minimum_validation_direction_accuracy
        ),
    }
    source_after = _sha256(args.source_report)
    report: dict[str, object] = {
        "contract": SELECTOR_AUDIT_SCHEMA,
        "source_report_sha256": source_after,
        "source_unchanged": source_after == source_before,
        "source_selected_step": source.get("selected_step"),
        "corrected_selected_step": selected["step"],
        "corrected_selected_validation_progress": metrics,
        "corrected_selected_preservation_gates": preservation,
        "corrected_progress_gates": progress_gates,
        "history_steps": [candidate["step"] for candidate in history],
        "selection_prioritizes_complete_progress_pass": True,
        "process_passed": bool(
            source_after == source_before
            and all(preservation.values())
            and all(progress_gates.values())
            and selected["step"] != source.get("selected_step")
        ),
        "model_loaded": 0,
        "model_tensor_changes": [],
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "optimizer_steps": 0,
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
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-validation-correlation", type=float, default=0.8)
    parser.add_argument("--maximum-validation-mae-m", type=float, default=0.5)
    parser.add_argument(
        "--maximum-validation-near-plane-mae-m", type=float, default=0.6
    )
    parser.add_argument(
        "--minimum-validation-direction-accuracy", type=float, default=0.75
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("selector audit refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))

