#!/usr/bin/env python3
"""Compare same-fixture parent and candidate staged gate-count screens."""

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


SCHEMA = "vq2_staged_count5_paired_diagnostic_v1"
MAX_EXECUTED_ACTION_ERROR = 5e-5


def _integer_rate(metrics: dict[str, Any], name: str, episodes: int) -> int:
    value = float(metrics.get(name, math.nan))
    count = round(value * episodes)
    if not math.isfinite(value) or abs(value * episodes - count) > 1e-6:
        raise RuntimeError(f"staged metric {name} is not an integer rate")
    return int(count)


def summarize_count(
    report: dict[str, Any],
    *,
    num_gates: int = 5,
) -> dict[str, Any]:
    expected_schemas = {"vq2_staged_gate_count_component_v1"}
    if num_gates == 5:
        # The original count-5 tool used a count-specific alias. Generalized
        # gate-count runners use the equivalent generic component schema.
        expected_schemas.add("vq2_staged_count5_component_v1")
    if (
        report.get("schema") not in expected_schemas
        or not report.get("completed")
        or report.get("num_gates") != num_gates
        or report.get("agents") != report.get("episodes")
    ):
        raise RuntimeError("staged component report is incomplete")
    episodes = int(report["episodes"])
    distribution = {
        int(index): int(count)
        for index, count in report.get(
            "maximum_held_public_index_distribution", {}
        ).items()
    }
    if sum(distribution.values()) != episodes:
        raise RuntimeError("staged phase distribution does not cover every episode")
    reach = {
        gate: sum(count for index, count in distribution.items() if index >= gate)
        for gate in range(1, num_gates + 1)
    }
    metrics = report["metrics"]
    hard_transport_pass = bool(
        report.get("teacher_action_blend") == 0.0
        and not report.get("nonfinite_action")
        and report.get("action_envelope_violations") == 0
        and report.get("executed_action_max_error", math.inf)
        <= MAX_EXECUTED_ACTION_ERROR
        and report.get("phase_changes_off_tick") == 0
        and report.get("phase_decreases") == 0
        and report.get("phase_skips") == 0
        and report.get("raw_phase_encoding_max_error", math.inf) <= 1e-6
        and metrics.get("env/out_of_order") == 0.0
        and metrics.get("env/action_envelope_violation") == 0.0
        and metrics.get("env/wire_rate_envelope_violation") == 0.0
        and metrics.get("env/thrust_envelope_violation") == 0.0
        and report.get("safety", {}).get("teacher_labels_written") == 0
        and report.get("safety", {}).get("student_updates") == 0
        and report.get("safety", {}).get("flight_sim_packets_sent") == 0
        and report.get("safety", {}).get("sealed_test_accesses") == 0
        and not report.get("safety", {}).get("submission_authorized")
    )
    return {
        "episodes": episodes,
        "seed": int(report["seed"]),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "successes": _integer_rate(metrics, "env/success_rate", episodes),
        "crashes": _integer_rate(metrics, "env/crash", episodes),
        "misses": _integer_rate(metrics, "env/missed_gate", episodes),
        "timeouts": _integer_rate(metrics, "env/timeout", episodes),
        "mean_gates_passed": float(metrics["env/gates_passed"]),
        "gate_reach": {str(gate): value for gate, value in reach.items()},
        "hard_transport_pass": hard_transport_pass,
        "executed_action_max_error": report["executed_action_max_error"],
        "phase_distribution": {
            str(index): distribution.get(index, 0) for index in range(17)
        },
        "vector_steps": int(report["vector_steps"]),
        "wall_time_seconds": float(report["wall_time_seconds"]),
    }


def diagnostic_passes(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if parent["episodes"] != candidate["episodes"] or parent["seed"] != candidate["seed"]:
        return False
    parent_reach = parent["gate_reach"]
    candidate_reach = candidate["gate_reach"]
    downstream_improvement = bool(
        candidate["successes"] > parent["successes"]
        or int(candidate_reach["3"]) > int(parent_reach["3"])
    )
    downstream_signal = bool(
        candidate["successes"] > 0 or int(candidate_reach["3"]) > 0
    )
    return bool(
        parent["hard_transport_pass"]
        and candidate["hard_transport_pass"]
        and int(candidate_reach["1"]) >= int(parent_reach["1"])
        and candidate["crashes"] <= parent["crashes"]
        and downstream_signal
        and downstream_improvement
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gates", type=int, choices=(5, 11), default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent_path = args.parent_output.resolve() / f"count_{args.num_gates}.json"
    candidate_path = args.candidate_output.resolve() / f"count_{args.num_gates}.json"
    preregistration = args.preregistration.resolve()
    parent_report = json.loads(parent_path.read_text())
    candidate_report = json.loads(candidate_path.read_text())
    parent = summarize_count(parent_report, num_gates=args.num_gates)
    candidate = summarize_count(candidate_report, num_gates=args.num_gates)
    if (
        parent_report.get("source_identity", {}).get("source_commit")
        != candidate_report.get("source_identity", {}).get("source_commit")
    ):
        raise RuntimeError("paired staged components do not share a source commit")
    admitted = diagnostic_passes(parent, candidate)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": (
            SCHEMA
            if args.num_gates == 5
            else "vq2_staged_gate_count_paired_diagnostic_v1"
        ),
        "completed": True,
        "qualified_for_full_screen": admitted,
        "num_gates": args.num_gates,
        "source_commit": source_commit,
        "parent": parent,
        "candidate": candidate,
        "delta": {
            "successes": candidate["successes"] - parent["successes"],
            "crashes": candidate["crashes"] - parent["crashes"],
            "gate1_reach": (
                int(candidate["gate_reach"]["1"]) - int(parent["gate_reach"]["1"])
            ),
            "gate2_reach": (
                int(candidate["gate_reach"]["2"]) - int(parent["gate_reach"]["2"])
            ),
            "gate3_reach": (
                int(candidate["gate_reach"]["3"]) - int(parent["gate_reach"]["3"])
            ),
        },
        "criteria": {
            "same_seed_and_episode_count": (
                parent["episodes"] == candidate["episodes"]
                and parent["seed"] == candidate["seed"]
            ),
            "both_hard_transport_pass": (
                parent["hard_transport_pass"] and candidate["hard_transport_pass"]
            ),
            "gate1_not_regressed": (
                int(candidate["gate_reach"]["1"]) >= int(parent["gate_reach"]["1"])
            ),
            "crash_count_not_regressed": (
                candidate["crashes"] <= parent["crashes"]
            ),
            "ordered_gate3_or_finish_signal": (
                candidate["successes"] > 0
                or int(candidate["gate_reach"]["3"]) > 0
            ),
            "downstream_improves_parent": (
                candidate["successes"] > parent["successes"]
                or int(candidate["gate_reach"]["3"])
                > int(parent["gate_reach"]["3"])
            ),
        },
        "artifact_sha256": {
            "parent_count_report": sha256_path(parent_path),
            "candidate_count_report": sha256_path(candidate_path),
            "preregistration": sha256_path(preregistration),
            "component": sha256_path(
                ROOT / "scripts/eval_vq2_staged_count5_component.py"
            ),
            "comparator": sha256_path(Path(__file__).resolve()),
        },
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "shadow_authorized": False,
            "training_authorized": False,
            "submission_authorized": False,
        },
        "next_authority": (
            "Preregister a fresh full counts-5/8/11/12 teacher-free screen."
            if admitted
            else "Reject the candidate screen path and start a causally distinct offline repair loop."
        ),
    }
    output = args.output.resolve()
    if output.is_file():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        if json.loads(output.read_text()) != report:
            raise RuntimeError("completed staged diagnostic identity changed")
    else:
        write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
