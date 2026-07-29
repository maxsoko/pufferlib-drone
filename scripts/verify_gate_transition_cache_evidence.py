#!/usr/bin/env python3
"""Verify the Candidate-013 diagnosis and Gate-3+ scoped cache correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inverse_tanh(value: float, scale: float) -> float:
    clipped = max(-0.999999, min(0.999999, float(value)))
    return math.atanh(clipped) * scale


def build_report(candidate_path: Path, runner_path: Path) -> dict:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    samples = candidate.get("policy_trace", {}).get("samples", [])
    transition_index = None
    for index in range(1, len(samples)):
        if (
            int(samples[index - 1].get("official_active_gate_index", -1)) == 1
            and int(samples[index].get("official_active_gate_index", -1)) == 2
        ):
            transition_index = index
            break

    blockers: list[str] = []
    if transition_index is None:
        blockers.append("missing_gate2_to_gate3_transition")
        transition = {}
        observed_body = None
        visible_body = None
        mismatch_m = None
        accepted_growth = None
        rejected_growth = None
    else:
        transition = samples[transition_index]
        observation = transition.get("observation", [])
        pose = transition.get("gate_pose") or {}
        visible_body = pose.get("body_vector_ned_m")
        if len(observation) < 14 or not visible_body:
            blockers.append("transition_missing_observation_or_visible_pose")
            observed_body = None
            mismatch_m = None
        else:
            observed_body = [
                _inverse_tanh(observation[11], 10.0),
                _inverse_tanh(observation[12], 5.0),
                _inverse_tanh(observation[13], 5.0),
            ]
            mismatch_m = math.dist(observed_body, visible_body)
            if mismatch_m < 15.0:
                blockers.append("transition_pose_mismatch_not_large")

        gate3_samples = [
            sample
            for sample in samples[transition_index:]
            if int(sample.get("official_active_gate_index", -1)) == 2
        ]
        first_motion = transition.get("gate_motion") or {}
        last_visible_motion = next(
            (
                sample.get("gate_motion") or {}
                for sample in reversed(gate3_samples)
                if sample.get("gate_pose") is not None
            ),
            first_motion,
        )
        accepted_growth = int(last_visible_motion.get("accepted_samples", 0)) - int(
            first_motion.get("accepted_samples", 0)
        )
        rejected_growth = int(last_visible_motion.get("rejected_samples", 0)) - int(
            first_motion.get("rejected_samples", 0)
        )
        if accepted_growth != 0:
            blockers.append("gate3_alias_was_unexpectedly_accepted")
        if rejected_growth <= 0:
            blockers.append("gate3_real_detections_were_not_rejected")

    runner_source = runner_path.read_text(encoding="utf-8")
    required_source = {
        "transition_epoch": "official_gate_observation_epoch.observe(official_gate_index)",
        "gate3plus_scope": "official_gate_index >= 2",
        "gate_pose_clear": "gate_pose = None",
        "control_pose_clear": "control_gate_pose = None",
        "control_timestamp_clear": "last_control_detection_s = None",
        "motion_filter_reset": "policy_gate_motion_state.reset()",
    }
    source_matches = {
        name: token in runner_source for name, token in required_source.items()
    }
    blockers.extend(
        f"runner_missing_{name}" for name, matched in source_matches.items() if not matched
    )

    reset = candidate.get("control_inputs", {}).get("official_reset_start", {})
    if reset.get("reset_detected") is not True:
        blockers.append("candidate_reset_not_detected")
    if candidate.get("official_active_gate_index") != 2:
        blockers.append("candidate_did_not_end_at_gate3")
    if candidate.get("crash_detected") is not False:
        blockers.append("candidate_collision_confounds_diagnosis")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "candidate_report": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "runner": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "candidate_reset": reset,
        "transition_sample_index": transition_index,
        "transition_elapsed_s": transition.get("elapsed_s"),
        "visible_new_gate_body_ned_m": visible_body,
        "policy_observed_gate_body_ned_m": observed_body,
        "visible_vs_observed_mismatch_m": mismatch_m,
        "accepted_sample_growth_during_visible_gate3": accepted_growth,
        "rejected_sample_growth_during_visible_gate3": rejected_growth,
        "runner_source_matches": source_matches,
        "transition_cache_scope": {
            "first_invalidated_official_gate_index": 2,
            "preserves_gate1_to_gate2_handoff": source_matches["gate3plus_scope"],
        },
        "candidate_runtime": {
            "official_active_gate_index": candidate.get("official_active_gate_index"),
            "crash_detected": candidate.get("crash_detected"),
            "effective_command_hz": candidate.get("sitl", {}).get(
                "effective_command_hz"
            ),
            "command_rate_violations": candidate.get("sitl", {}).get(
                "command_rate_violations"
            ),
            "collisions": candidate.get("sitl", {})
            .get("telemetry", {})
            .get("collisions"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument(
        "--runner", type=Path, default=Path("scripts/drone_sitl_competition_smoke.py")
    )
    parser.add_argument("--json-path", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.candidate_report, args.runner)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
