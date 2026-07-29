#!/usr/bin/env python3
"""Verify the promoted legacy prefix and its complete recurrent input bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstructed_legacy_confidence(observation: list[float]) -> float:
    """Recover the old image-size confidence from the current apparent size."""

    if len(observation) < 18 or float(observation[10]) <= 0.5:
        return 0.0
    return max(0.0, min(1.0, 0.5 * float(observation[16])))


def exact_hybrid_roundtrip(
    observation: list[float], gate: int, previous_yaw_action: float
) -> list[float]:
    """Embed archived fields plus the unlogged recurrent yaw state in the new ABI."""

    if len(observation) != 23:
        raise ValueError("expected 23-value legacy observation")
    values = np.zeros(32, dtype=np.float32)
    values[:23] = np.asarray(observation, dtype=np.float32)
    values[17] = np.float32(0.73125)  # current centering quality is independent
    values[18] = np.float32(0.125)  # current run duration is independent
    values[22] = np.float32(previous_yaw_action)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = np.float32(1.0)
    values[30] = np.float32(observation[17])
    values[31] = np.float32(observation[18])
    return hybrid._legacy_prefix_observation(values).tolist()


def legacy_callable_yaw_contract(path: Path) -> dict:
    """Prove how the historical callable populated recurrent field 22."""

    source = path.read_text(encoding="utf-8")
    required_fragments = [
        "_LAST_YAW_ACTION = 0.0",
        "recurrent_observation = values.copy()",
        "recurrent_observation[22] = np.float32(_LAST_YAW_ACTION)",
        "base_action = model.infer(recurrent_observation)",
        "_LAST_YAW_ACTION = float(action[3])",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": not missing,
        "missing_fragments": missing,
        "model_field_22": "previous normalized yaw action",
        "outer_report_field_22": "official active gate index / 3",
    }


def inferred_elapsed_denominator(samples: list[dict]) -> float | None:
    denominators = []
    for sample in samples:
        observation = sample.get("observation") or []
        elapsed_s = float(sample.get("elapsed_s") or 0.0)
        if len(observation) <= 18 or float(observation[18]) <= 0.01:
            continue
        denominators.append(elapsed_s / float(observation[18]))
    return None if not denominators else float(statistics.median(denominators))


def verify(
    reports: list[Path],
    checkpoint: Path,
    policy_callable: Path,
    legacy_policy_callable: Path,
    *,
    confidence_atol: float = 0.01,
    failed_candidate: Path | None = None,
    wrong_phase_candidate: Path | None = None,
) -> dict:
    sources: list[dict] = []
    confidence_samples = 0
    exact_bridge_max_error = 0.0
    max_confidence_error = 0.0
    max_confidence_error_record = None
    gate1_gate2_passes = 0
    gate3_passes = 0
    blockers: list[str] = []
    reference_samples: list[dict] = []
    reference_fields: list[str] | None = None
    yaw_contract = legacy_callable_yaw_contract(legacy_policy_callable)
    if not yaw_contract["passed"]:
        blockers.append("legacy_callable_yaw_contract")
    current_source = policy_callable.read_text(encoding="utf-8")
    if "legacy[22]" in current_source:
        blockers.append("current_callable_overwrites_previous_yaw")
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        official_index = int(report.get("official_active_gate_index") or 0)
        gate1_gate2_passes += int(official_index >= 2)
        gate3_passes += int(official_index >= 3)
        fields = (report.get("policy_trace") or {}).get("observation_fields") or []
        if reference_fields is None:
            reference_fields = list(fields)
        reference_samples.extend((report.get("policy_trace") or {}).get("samples") or [])
        source = {
            "path": str(path),
            "sha256": sha256_file(path),
            "official_active_gate_index": official_index,
            "acceptance_passed": bool(report.get("acceptance_passed")),
            "legacy_confidence_field": (
                fields[17] if len(fields) > 17 else None
            ),
        }
        sources.append(source)
        if source["legacy_confidence_field"] != "gate_pose_confidence":
            blockers.append(f"legacy_confidence_contract:{path}")
        for sample_index, sample in enumerate(
            (report.get("policy_trace") or {}).get("samples") or []
        ):
            gate = int(sample.get("official_active_gate_index", -1))
            if gate < 0 or gate > 2:
                continue
            observation = sample.get("observation") or []
            if len(observation) != 23:
                blockers.append(f"legacy_observation_size:{path}:{sample_index}")
                continue
            expected = float(observation[17])
            reconstructed = reconstructed_legacy_confidence(observation)
            error = abs(reconstructed - expected)
            confidence_samples += 1
            # The archived trace stores the outer phase value at field 22, not
            # the recurrent value injected by the historical callable. Use a
            # sentinel to prove the new runner's explicit previous-yaw value is
            # preserved while every archived numeric field is reconstructed.
            previous_yaw_sentinel = -0.375
            roundtrip = exact_hybrid_roundtrip(
                observation, gate, previous_yaw_sentinel
            )
            expected_float32 = np.asarray(observation, dtype=np.float32)
            expected_float32[22] = np.float32(previous_yaw_sentinel)
            exact_bridge_max_error = max(
                exact_bridge_max_error,
                max(
                    abs(float(a) - float(b))
                    for a, b in zip(roundtrip, expected_float32.tolist())
                ),
            )
            if error > max_confidence_error:
                max_confidence_error = error
                max_confidence_error_record = {
                    "path": str(path),
                    "sample": sample_index,
                    "elapsed_s": sample.get("elapsed_s"),
                    "gate": gate,
                    "gate_visible": float(observation[10]) > 0.5,
                    "apparent_size": float(observation[16]),
                    "expected_confidence": expected,
                    "reconstructed_confidence": reconstructed,
                    "absolute_error": error,
                }
    if len(reports) != 10:
        blockers.append(f"reliability_source_count:{len(reports)}!=10")
    if gate1_gate2_passes != len(reports):
        blockers.append(
            f"gate1_gate2_reliability:{gate1_gate2_passes}!={len(reports)}"
        )
    if gate3_passes < 8:
        blockers.append(f"gate3_reliability:{gate3_passes}<8")
    if confidence_samples <= 0:
        blockers.append("no_prefix_confidence_samples")
    if exact_bridge_max_error != 0.0:
        blockers.append(f"reserved_confidence_bridge_error:{exact_bridge_max_error}")
    failed_candidate_diagnosis = None
    if failed_candidate is not None:
        failed_report = json.loads(failed_candidate.read_text(encoding="utf-8"))
        failed_trace = failed_report.get("policy_trace") or {}
        failed_fields = list(failed_trace.get("observation_fields") or [])[:23]
        field_mismatches = [
            {
                "index": index,
                "legacy": legacy_name,
                "failed_candidate": failed_fields[index],
            }
            for index, legacy_name in enumerate(reference_fields or [])
            if index >= len(failed_fields) or failed_fields[index] != legacy_name
        ]
        reference_denominator = inferred_elapsed_denominator(reference_samples)
        failed_denominator = inferred_elapsed_denominator(failed_trace.get("samples") or [])
        failed_candidate_diagnosis = {
            "path": str(failed_candidate),
            "sha256": sha256_file(failed_candidate),
            "official_active_gate_index": int(
                failed_report.get("official_active_gate_index") or 0
            ),
            "collisions": int(
                ((failed_report.get("sitl") or {}).get("telemetry") or {}).get(
                    "collisions", 0
                )
            ),
            "field_mismatches": field_mismatches,
            "legacy_elapsed_denominator_s": reference_denominator,
            "failed_elapsed_denominator_s": failed_denominator,
            "original_bridge_restored_indices": [17],
            "missing_bridge_indices": [18],
            "semantic_field_22": (
                "current last-yaw field already matches the historical "
                "callable's recurrent input"
            ),
            "corrected_bridge": {
                "17": "reserved slot 30 legacy gate-pose confidence",
                "18": "reserved slot 31 elapsed/10.5",
                "22": "preserve current previous normalized yaw action",
            },
        }
        if [item["index"] for item in field_mismatches] != [17, 22]:
            blockers.append(f"unexpected_failed_field_mismatches:{field_mismatches}")
        if reference_denominator is None or abs(reference_denominator - 10.5) > 1e-3:
            blockers.append(f"legacy_elapsed_denominator:{reference_denominator}")
        if failed_denominator is None or abs(failed_denominator - 30.0) > 1e-3:
            blockers.append(f"failed_elapsed_denominator:{failed_denominator}")
    wrong_phase_candidate_diagnosis = None
    if wrong_phase_candidate is not None:
        wrong_report = json.loads(wrong_phase_candidate.read_text(encoding="utf-8"))
        wrong_phase_candidate_diagnosis = {
            "path": str(wrong_phase_candidate),
            "sha256": sha256_file(wrong_phase_candidate),
            "official_active_gate_index": int(
                wrong_report.get("official_active_gate_index") or 0
            ),
            "collisions": int(
                ((wrong_report.get("sitl") or {}).get("telemetry") or {}).get(
                    "collisions", 0
                )
            ),
            "rejected_adapter": "N147 overwrote recurrent field 22 with gate/3",
            "correct_adapter": "preserve current previous normalized yaw action",
        }
    return {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "expected_reports": 10,
            "required_gate1_gate2_passes": 10,
            "required_gate3_passes": 8,
            "confidence_bridge": "exact legacy confidence in reserved slot 30",
            "elapsed_bridge": "exact legacy 10.5-second fraction in reserved slot 31",
            "recurrent_yaw_bridge": (
                "preserve current field 22 previous normalized yaw action"
            ),
            "phase_selection": "use current six-way gate one-hot outside the model",
            "rejected_approximation": "0.5 * gate_apparent_size_norm",
            "rejected_approximation_atol": confidence_atol,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_callable": str(policy_callable),
        "policy_callable_sha256": sha256_file(policy_callable),
        "legacy_callable_yaw_contract": yaw_contract,
        "gate1_gate2_passes": gate1_gate2_passes,
        "gate3_passes": gate3_passes,
        "confidence_samples": confidence_samples,
        "exact_bridge_max_error": exact_bridge_max_error,
        "max_confidence_error": max_confidence_error,
        "max_confidence_error_record": max_confidence_error_record,
        "failed_candidate_diagnosis": failed_candidate_diagnosis,
        "wrong_phase_candidate_diagnosis": wrong_phase_candidate_diagnosis,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-callable", type=Path, required=True)
    parser.add_argument("--legacy-policy-callable", type=Path, required=True)
    parser.add_argument("--confidence-atol", type=float, default=0.01)
    parser.add_argument("--failed-candidate", type=Path)
    parser.add_argument("--wrong-phase-candidate", type=Path)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify(
        args.reports,
        args.checkpoint,
        args.policy_callable,
        args.legacy_policy_callable,
        confidence_atol=args.confidence_atol,
        failed_candidate=args.failed_candidate,
        wrong_phase_candidate=args.wrong_phase_candidate,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
