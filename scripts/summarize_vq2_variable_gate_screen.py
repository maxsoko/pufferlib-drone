#!/usr/bin/env python3
"""Create deterministic evidence from a completed variable-gate policy screen."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once


SCHEMA = "vq2_variable_gate_teacher_free_screen_evidence_v1"
COUNTS = (5, 8, 11, 12)
ENGINE_GATE_CAP = 16
MAX_EXECUTED_ACTION_ERROR = 5e-5


def _integer_rate(metrics: dict[str, Any], name: str, episodes: int) -> int:
    value = float(metrics.get(name, math.nan))
    count = round(value * episodes)
    if not math.isfinite(value) or abs(value * episodes - count) > 1e-6:
        raise RuntimeError(f"screen metric {name} is not an integer rate")
    return int(count)


def summarize_count(report: dict[str, Any]) -> dict[str, Any]:
    if (
        not report.get("completed")
        or report.get("num_gates") not in COUNTS
        or report.get("agents") != report.get("episodes")
    ):
        raise RuntimeError("count report is not a completed one-episode screen")
    episodes = int(report["episodes"])
    if episodes <= 0:
        raise RuntimeError("count report has no episodes")
    distribution = {
        int(index): int(count)
        for index, count in report.get(
            "maximum_held_public_index_distribution", {}
        ).items()
    }
    if (
        any(index not in range(ENGINE_GATE_CAP + 1) for index in distribution)
        or any(count < 0 for count in distribution.values())
        or sum(distribution.values()) != episodes
    ):
        raise RuntimeError("held public-index distribution is invalid")
    gate_reach = {
        str(gate): sum(
            count for index, count in distribution.items() if index >= gate
        )
        for gate in range(1, ENGINE_GATE_CAP + 1)
    }
    metrics = report["metrics"]
    crashes = _integer_rate(metrics, "env/crash", episodes)
    crashes_low = _integer_rate(metrics, "env/crash_low", episodes)
    crashes_xy = _integer_rate(metrics, "env/crash_xy", episodes)
    crashes_high = _integer_rate(metrics, "env/crash_high", episodes)
    if crashes != crashes_low + crashes_xy + crashes_high:
        raise RuntimeError("crash classes do not sum to the crash count")
    transport = {
        "executed_action_max_error": float(
            report.get("executed_action_max_error", math.inf)
        ),
        "phase_changes_off_tick": int(report.get("phase_changes_off_tick", -1)),
        "phase_decreases": int(report.get("phase_decreases", -1)),
        "phase_skips": int(report.get("phase_skips", -1)),
        "raw_phase_encoding_max_error": float(
            report.get("raw_phase_encoding_max_error", math.inf)
        ),
        "nonfinite_action": bool(report.get("nonfinite_action", True)),
        "action_envelope_violations": int(
            report.get("action_envelope_violations", -1)
        ),
        "wire_rate_envelope_violations": _integer_rate(
            metrics, "env/wire_rate_envelope_violation", episodes
        ),
        "thrust_envelope_violations": _integer_rate(
            metrics, "env/thrust_envelope_violation", episodes
        ),
        "out_of_order": _integer_rate(metrics, "env/out_of_order", episodes),
    }
    hard_transport_pass = bool(
        report.get("teacher_action_blend") == 0.0
        and transport["executed_action_max_error"] <= MAX_EXECUTED_ACTION_ERROR
        and transport["phase_changes_off_tick"] == 0
        and transport["phase_decreases"] == 0
        and transport["phase_skips"] == 0
        and transport["raw_phase_encoding_max_error"] <= 1e-6
        and not transport["nonfinite_action"]
        and transport["action_envelope_violations"] == 0
        and transport["wire_rate_envelope_violations"] == 0
        and transport["thrust_envelope_violations"] == 0
        and transport["out_of_order"] == 0
        and report.get("safety", {}).get("teacher_labels_written") == 0
        and report.get("safety", {}).get("student_updates") == 0
        and report.get("safety", {}).get("flight_sim_packets_sent") == 0
        and report.get("safety", {}).get("sealed_test_accesses") == 0
        and not report.get("safety", {}).get("submission_authorized")
    )
    return {
        "num_gates": int(report["num_gates"]),
        "episodes": episodes,
        "seed": int(report["seed"]),
        "successes": _integer_rate(metrics, "env/success_rate", episodes),
        "crashes": crashes,
        "crashes_low": crashes_low,
        "crashes_xy": crashes_xy,
        "crashes_high": crashes_high,
        "misses": _integer_rate(metrics, "env/missed_gate", episodes),
        "timeouts": _integer_rate(metrics, "env/timeout", episodes),
        "mean_gates_passed": float(metrics["env/gates_passed"]),
        "gate_reach": gate_reach,
        "crossing_margin_violations": _integer_rate(
            metrics, "env/crossing_margin_violation", episodes
        ),
        "vector_steps": int(report["vector_steps"]),
        "wall_time_seconds": float(report["wall_time_seconds"]),
        "hard_transport_pass": hard_transport_pass,
        "transport": transport,
    }


def build_evidence(
    *,
    screen: Path,
    candidate_admission: Path,
) -> dict[str, Any]:
    aggregate_path = screen / "report.json"
    state_path = screen / "state.json"
    aggregate = json.loads(aggregate_path.read_text())
    state = json.loads(state_path.read_text())
    admission = json.loads(candidate_admission.read_text())
    if (
        aggregate.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or not aggregate.get("completed")
        or aggregate.get("counts") != list(COUNTS)
        or aggregate.get("total_episodes")
        != aggregate.get("episodes_per_count") * len(COUNTS)
        or state.get("status") not in {"admitted", "rejected"}
        or state.get("completed_counts") != list(COUNTS)
        or state.get("report_sha256") != sha256_path(aggregate_path)
    ):
        raise RuntimeError("screen aggregate/state is not terminal and complete")
    checkpoint_sha256 = aggregate.get("checkpoint_sha256")
    if (
        admission.get("artifact_sha256", {}).get("checkpoint")
        != checkpoint_sha256
        or admission.get("safety", {}).get("flight_sim_packets_sent") != 0
        or admission.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("candidate admission does not bind the screen actor")

    source_identity = aggregate.get("source_identity")
    if not isinstance(source_identity, dict):
        raise RuntimeError("screen aggregate has no source identity")
    count_reports: list[dict[str, Any]] = []
    count_sha256: dict[str, str] = {}
    for count in COUNTS:
        path = screen / f"count_{count}.json"
        digest = sha256_path(path)
        contract = aggregate.get("count_reports", {}).get(str(count), {})
        report = json.loads(path.read_text())
        if (
            contract.get("sha256") != digest
            or report.get("source_identity") != source_identity
            or report.get("checkpoint_sha256") != checkpoint_sha256
            or report.get("num_gates") != count
        ):
            raise RuntimeError(f"count {count} does not bind the aggregate")
        count_reports.append(summarize_count(report))
        count_sha256[str(count)] = digest

    episodes = sum(report["episodes"] for report in count_reports)
    sums = {
        name: sum(int(report[name]) for report in count_reports)
        for name in (
            "successes",
            "crashes",
            "crashes_low",
            "crashes_xy",
            "crashes_high",
            "misses",
            "timeouts",
            "crossing_margin_violations",
        )
    }
    gate_reach = {
        str(gate): sum(
            int(report["gate_reach"][str(gate)]) for report in count_reports
        )
        for gate in range(1, ENGINE_GATE_CAP + 1)
    }
    transport = {
        "executed_action_max_error": max(
            report["transport"]["executed_action_max_error"]
            for report in count_reports
        ),
        "phase_changes_off_tick": sum(
            report["transport"]["phase_changes_off_tick"]
            for report in count_reports
        ),
        "phase_decreases": sum(
            report["transport"]["phase_decreases"] for report in count_reports
        ),
        "phase_skips": sum(
            report["transport"]["phase_skips"] for report in count_reports
        ),
        "raw_phase_encoding_max_error": max(
            report["transport"]["raw_phase_encoding_max_error"]
            for report in count_reports
        ),
        "nonfinite_action": any(
            report["transport"]["nonfinite_action"] for report in count_reports
        ),
        "action_envelope_violations": sum(
            report["transport"]["action_envelope_violations"]
            for report in count_reports
        ),
        "wire_rate_envelope_violations": sum(
            report["transport"]["wire_rate_envelope_violations"]
            for report in count_reports
        ),
        "thrust_envelope_violations": sum(
            report["transport"]["thrust_envelope_violations"]
            for report in count_reports
        ),
        "out_of_order": sum(
            report["transport"]["out_of_order"] for report in count_reports
        ),
    }
    hard_transport_pass = all(
        report["hard_transport_pass"] for report in count_reports
    )
    admitted = bool(aggregate.get("admitted"))
    if (
        sums["successes"] != aggregate.get("successes")
        or (sums["crashes"] == 0) != aggregate.get("zero_crash")
        or episodes != aggregate.get("total_episodes")
    ):
        raise RuntimeError("aggregate totals disagree with count evidence")
    expected_admission = bool(
        hard_transport_pass
        and sums["crashes"] == 0
        and (
            not aggregate.get("crossing_margin_admission_predicate")
            or sums["crossing_margin_violations"] == 0
        )
        and sums["successes"] / episodes + 1e-12
        >= float(aggregate["minimum_success_rate"])
    )
    if (
        admitted != expected_admission
        or admitted != (state.get("status") == "admitted")
    ):
        raise RuntimeError("aggregate admission disagrees with terminal evidence")
    analyzer_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": SCHEMA,
        "screen_tag": aggregate["tag"],
        "completed": True,
        "admitted": admitted,
        "unchanged_retry_forbidden": not admitted,
        "screen_source_commit": source_identity["source_commit"],
        "analyzer_source_commit": analyzer_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "episodes": episodes,
        **sums,
        "success_rate": sums["successes"] / episodes,
        "mean_gates_passed": sum(
            report["mean_gates_passed"] * report["episodes"]
            for report in count_reports
        )
        / episodes,
        "gate_reach": gate_reach,
        "hard_transport_pass": hard_transport_pass,
        "transport": transport,
        "counts": {
            str(report["num_gates"]): report for report in count_reports
        },
        "artifact_sha256": {
            "aggregate_report": sha256_path(aggregate_path),
            "terminal_state": sha256_path(state_path),
            "candidate_admission": sha256_path(candidate_admission),
            "analyzer": sha256_path(Path(__file__).resolve()),
            "count_reports": count_sha256,
        },
        "safety": {
            "teacher_labels_written": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "shadow_authorized": False,
            "training_authorized": False,
            "submission_authorized": False,
        },
        "next_authority": (
            "Source-lock the recurrent callable and run command-free replay/parity."
            if admitted
            else "Preregister a causally distinct offline visited-state collection; never retry the screen unchanged."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--candidate-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_evidence(
        screen=args.screen.resolve(),
        candidate_admission=args.candidate_admission.resolve(),
    )
    write_json_once(args.output.resolve(), evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
