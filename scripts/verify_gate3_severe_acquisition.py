#!/usr/bin/env python3
"""Verify the N188 Gate-3 severe acquisition guard on official traces."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_severe_acquisition as policy


EXPECTED_CHECKPOINT_SHA256 = (
    "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
)
EXPECTED_PARAMETERS = {
    "trigger_min_forward_m": 15.0,
    "trigger_max_forward_m": 35.0,
    "trigger_min_abs_right_m": 8.0,
    "release_max_abs_right_m": 4.5,
    "release_max_abs_yaw_error_rad": 0.12,
    "acquisition_pitch_norm": -0.24,
    "acquisition_roll_norm": 0.0,
    "acquisition_thrust_norm": 0.0,
    "max_yaw_step_rad": 0.35,
    "max_dropout_ticks": 30,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(path: Path) -> str:
    bounded = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if bounded:
        return f"candidate{bounded.group(1)}"
    reliability = re.search(r"reliability_129_attempt_(\d{3})", path.name)
    if reliability:
        return f"reliability129_attempt{reliability.group(1)}"
    return path.stem


def _current_observation(sample: dict) -> np.ndarray | None:
    raw = sample.get("observation") or []
    if len(raw) == 32:
        return np.asarray(raw, dtype=np.float32)
    if len(raw) != 23:
        return None
    gate = int(sample.get("official_active_gate_index", -1))
    values = np.zeros(32, dtype=np.float32)
    values[:23] = np.asarray(raw, dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = np.float32(1.0)
    values[30] = np.float32(raw[17])
    values[31] = np.float32(raw[18])
    return values


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = policy.Gate3SevereAcquisition()
    rows = []
    max_unchanged_error = 0.0
    invalid_outputs = 0
    changed_non_gate3 = 0
    for sample_index, sample in enumerate(
        sorted(
            (payload.get("policy_trace") or {}).get("samples") or [],
            key=lambda item: float(item.get("elapsed_s") or 0.0),
        )
    ):
        values = _current_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        governed = np.asarray(controller.apply(values, recorded), dtype=float)
        changed = bool(np.max(np.abs(governed - recorded)) > 1e-7)
        gate = int(sample.get("official_active_gate_index", -1))
        if changed and gate != 2:
            changed_non_gate3 += 1
        if not changed:
            max_unchanged_error = max(
                max_unchanged_error,
                float(np.max(np.abs(governed - recorded))),
            )
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1
        if changed:
            forward_m = policy.tail._inverse_tanh_norm(float(values[11]), 10.0)
            right_m = policy.tail._inverse_tanh_norm(float(values[12]), 5.0)
            yaw_error_rad = policy.tail._clamp(float(values[14])) * (
                math.pi / 4.0
            )
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                    "gate": gate,
                    "visible": float(values[10]) >= 0.5,
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "yaw_error_rad": yaw_error_rad,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                    "controller": controller.snapshot(),
                }
            )
    snapshot = controller.snapshot()
    return {
        "name": _name(path),
        "path": str(path),
        "sha256": _sha256(path),
        "passed_gate3": int(payload.get("official_active_gate_index") or 0) >= 3,
        "activations": snapshot["activations"],
        "releases": snapshot["releases"],
        "dropout_deactivations": snapshot["dropout_deactivations"],
        "active_steps": snapshot["steps_active"],
        "changed_samples": len(rows),
        "changed_non_gate3": changed_non_gate3,
        "max_unchanged_action_error": max_unchanged_error,
        "invalid_outputs": invalid_outputs,
        "rows": rows,
    }


def verify(
    paths: list[Path], checkpoint: Path, expected_policy_sha256: str
) -> dict:
    reports = [analyze(path) for path in paths]
    blockers: list[str] = []
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("prefix_checkpoint_hash_mismatch")
    policy_path = Path(policy.__file__).resolve()
    policy_sha = _sha256(policy_path)
    if policy_sha != expected_policy_sha256.lower():
        blockers.append("policy_callable_hash_mismatch")
    if dataclasses.asdict(policy.PARAMETERS) != EXPECTED_PARAMETERS:
        blockers.append("controller_parameter_vector_mismatch")

    candidate046 = next(
        (report for report in reports if report["name"] == "candidate046"), None
    )
    if candidate046 is None:
        blockers.append("candidate046_missing")
    elif candidate046["activations"] != 2:
        blockers.append(
            f"candidate046_activation_count:{candidate046['activations']}!=2"
        )
    elif not candidate046["rows"]:
        blockers.append("candidate046_no_changed_samples")
    else:
        first = candidate046["rows"][0]
        if not (
            first["visible"]
            and policy.PARAMETERS.trigger_min_forward_m
            <= first["forward_m"]
            <= policy.PARAMETERS.trigger_max_forward_m
            and abs(first["right_m"])
            >= policy.PARAMETERS.trigger_min_abs_right_m
        ):
            blockers.append("candidate046_first_change_outside_trigger")

    protected = [
        report for report in reports if report["name"] != "candidate046"
    ]
    for report in protected:
        if report["activations"]:
            blockers.append(f"protected_activation:{report['name']}")
        if report["changed_samples"]:
            blockers.append(f"protected_action_change:{report['name']}")
    for report in reports:
        if report["changed_non_gate3"]:
            blockers.append(f"changed_non_gate3:{report['name']}")
        if report["max_unchanged_action_error"] != 0.0:
            blockers.append(f"unchanged_action_error:{report['name']}")
        if report["invalid_outputs"]:
            blockers.append(f"invalid_output:{report['name']}")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": (
                "trace invariants can authorize only a hash-pinned passive "
                "shadow; the severe acquisition response is not a "
                "counterfactual flight proof"
            ),
        },
        "contract": {
            "mechanism": "severe far-offset hover/yaw acquisition guard",
            "scope": "official Gate 3 only",
            "base_policy": "exact N184 six-gate hybrid",
            "parameters": dataclasses.asdict(policy.PARAMETERS),
            "protected_sources": len(protected),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": policy_sha,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reports, args.checkpoint, args.expected_policy_sha256)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
