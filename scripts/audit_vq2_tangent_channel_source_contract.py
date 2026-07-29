#!/usr/bin/env python3
"""Correctly audit the immutable N698/N699 tangent-channel report contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


CHANNEL_AUDIT_SCHEMA = "vq2_actor_low_sensitivity_progress_channel_audit_v1"
SOURCE_CONTRACT_AUDIT_SCHEMA = "vq2_tangent_channel_source_contract_audit_v1"
N698_REPORT_SHA256 = (
    "73d3e05305ea40e2dbaa4574af074313174848f0343bdd2b7bfa155218640608"
)
N699_REPORT_SHA256 = (
    "d3377af2c694ae78696fbe757432eb1c96f3447b8cdb781069461396aa7d4fa5"
)
N699_EXECUTABLE_SHA256 = (
    "f4dfd0e160913e88949d79f6b14c759812da8aada221a0c1112e5cfe0c333229"
)


def _zero_side_effects(report: dict[str, object]) -> bool:
    return bool(
        report.get("model_tensor_changes") == []
        and report.get("actor_updates") == 0
        and report.get("representation_updates") == 0
        and report.get("model_optimizer_steps") == 0
        and report.get("checkpoint_written") == 0
        and report.get("validation_dataset_path_received") == 0
        and report.get("validation_dataset_opened") == 0
        and report.get("test_dataset_path_received") == 0
        and report.get("test_dataset_opened") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


def _n698_legacy_rejection_exact(report: dict[str, object]) -> bool:
    """Accept N698's original schema, where direction_mode was not emitted."""
    process_gates = report.get("process_gates")
    channel_gates = report.get("channel_gates")
    gradients = report.get("direct_gradient_norms")
    direction_mode = report.get("direction_mode")
    return bool(
        report.get("contract") == CHANNEL_AUDIT_SCHEMA
        and report.get("process_passed") is True
        and report.get("channel_admitted") is False
        and direction_mode in (None, "deterministic_low_sensitivity")
        and isinstance(process_gates, dict)
        and process_gates
        and all(value is True for value in process_gates.values())
        and isinstance(channel_gates, dict)
        and channel_gates.get("direct_gradients_finite_nonzero") is False
        and all(
            value is True
            for key, value in channel_gates.items()
            if key != "direct_gradients_finite_nonzero"
        )
        and isinstance(gradients, dict)
        and gradients.get("encoder") == 0.0
        and gradients.get("posterior") == 0.0
        and isinstance(gradients.get("sequence"), (int, float))
        and math.isfinite(float(gradients["sequence"]))
        and float(gradients["sequence"]) > 0.0
        and report.get("selected_scale") == 0.4
        and report.get("sources_unchanged") is True
        and _zero_side_effects(report)
    )


def _n699_failure_is_only_legacy_field(report: dict[str, object]) -> bool:
    expected_process_gates = {
        "all_scale_metrics_finite": True,
        "direction_is_unit_norm": True,
        "direction_mode_contract": True,
        "feature_cache_finite_float32": True,
        "fixed_reference_boundary_passes": True,
        "model_tensors_exact": True,
        "n696_rejection_exact": True,
        "n698_rejection_exact": False,
        "no_validation_or_test_path": True,
        "source_contracts_exact": True,
        "sources_unchanged": True,
        "train_geometry_exact": True,
    }
    expected_channel_gates = {
        "actor_null_residual_at_most_maximum": True,
        "direct_gradients_finite_nonzero": True,
        "selected_scale_at_least_minimum": True,
        "selected_scale_exists": True,
        "simplex_tangent_residual_at_most_maximum": True,
        "stochastic_direction_nonzero": True,
    }
    source_hashes = report.get("source_hashes")
    return bool(
        report.get("contract") == CHANNEL_AUDIT_SCHEMA
        and report.get("process_passed") is False
        and report.get("channel_admitted") is False
        and report.get("direction_mode") == "actor_tangent_null"
        and report.get("process_gates") == expected_process_gates
        and report.get("channel_gates") == expected_channel_gates
        and isinstance(source_hashes, dict)
        and source_hashes.get("n698_report") == N698_REPORT_SHA256
        and source_hashes.get("executable") == N699_EXECUTABLE_SHA256
        and report.get("sources_unchanged") is True
        and _zero_side_effects(report)
    )


def _n699_channel_evidence_exact(report: dict[str, object]) -> bool:
    gradients = report.get("direct_gradient_norms")
    geometry = report.get("geometry")
    scales = report.get("scale_results")
    selected_scale = report.get("selected_scale")
    if not (
        isinstance(gradients, dict)
        and isinstance(geometry, dict)
        and isinstance(scales, list)
        and selected_scale == 0.05
    ):
        return False
    gradient_values = [gradients.get(key) for key in ("encoder", "posterior", "sequence")]
    if not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in gradient_values
    ):
        return False
    selected = next(
        (entry for entry in scales if entry.get("scale") == selected_scale), None
    )
    return bool(
        geometry.get("actor_first_layer_residual_max_abs", math.inf) <= 1e-8
        and geometry.get("simplex_tangent_residual_max_abs", math.inf) <= 1e-8
        and geometry.get("stochastic_direction_norm", 0.0) > 0.0
        and geometry.get("direction_norm") == 1.0
        and isinstance(selected, dict)
        and selected.get("finite") is True
        and selected.get("negative_stochastic_probability_fraction", math.inf)
        <= 0.005
        and selected.get("simplex_sum_max_abs_error", math.inf) <= 1e-5
        and selected.get("absolute_correlation_improvement", -math.inf) >= 0.10
        and selected.get("residual_readout_rmse", math.inf) <= 1e-5
        and selected.get("raw_actor_drift", {}).get("rmse", math.inf) <= 0.001
        and selected.get("raw_actor_drift", {}).get("max_abs", math.inf) <= 0.01
        and selected.get("deterministic_action_drift", {}).get("rmse", math.inf)
        <= 0.0001
        and selected.get("deterministic_action_drift", {}).get("max_abs", math.inf)
        <= 0.001
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    n698_before = _sha256(args.n698_report)
    n699_before = _sha256(args.n699_report)
    n698 = json.loads(args.n698_report.read_text(encoding="utf-8"))
    n699 = json.loads(args.n699_report.read_text(encoding="utf-8"))

    gates = {
        "n698_report_sha256_exact": n698_before == N698_REPORT_SHA256,
        "n699_report_sha256_exact": n699_before == N699_REPORT_SHA256,
        "n698_legacy_rejection_exact": _n698_legacy_rejection_exact(n698),
        "n698_legacy_direction_field_absent": "direction_mode" not in n698,
        "n699_failure_is_only_legacy_field": _n699_failure_is_only_legacy_field(n699),
        "n699_channel_evidence_exact": _n699_channel_evidence_exact(n699),
    }
    corrected_process_gates = dict(n699["process_gates"])
    corrected_process_gates["n698_rejection_exact"] = gates[
        "n698_legacy_rejection_exact"
    ]

    n698_after = _sha256(args.n698_report)
    n699_after = _sha256(args.n699_report)
    gates["source_reports_unchanged"] = bool(
        n698_after == n698_before and n699_after == n699_before
    )
    corrected_process_passed = bool(
        all(gates.values()) and all(corrected_process_gates.values())
    )
    corrected_channel_admitted = bool(
        corrected_process_passed
        and all(value is True for value in n699["channel_gates"].values())
    )
    report: dict[str, object] = {
        "contract": SOURCE_CONTRACT_AUDIT_SCHEMA,
        "source_report_hashes": {
            "n698": n698_after,
            "n699": n699_after,
        },
        "audit_gates": gates,
        "n699_original_process_gates": n699["process_gates"],
        "corrected_process_gates": corrected_process_gates,
        "n699_original_process_passed": n699["process_passed"],
        "n699_original_channel_admitted": n699["channel_admitted"],
        "corrected_process_passed": corrected_process_passed,
        "corrected_channel_admitted": corrected_channel_admitted,
        "implementation_smoke_authorized": corrected_channel_admitted,
        "correction_scope": "legacy_n698_report_field_only",
        "model_loaded": 0,
        "model_tensor_changes": [],
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "train_dataset_path_received": 0,
        "validation_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "native_environment_created": 0,
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
    parser.add_argument("--n698-report", type=Path, required=True)
    parser.add_argument("--n699-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("source-contract audit refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
