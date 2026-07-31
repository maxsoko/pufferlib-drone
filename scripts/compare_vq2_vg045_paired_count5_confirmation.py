#!/usr/bin/env python3
"""Confirm VG044 against VG033 on one fresh paired count-5 fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_vq2_staged_count5_diagnostic import summarize_count
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once


SCHEMA = "vq2_vg045_paired_count5_confirmation_v1"


def confirmation_passes(
    parent: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    parent_reach = parent["gate_reach"]
    candidate_reach = candidate["gate_reach"]
    return bool(
        parent["episodes"] == candidate["episodes"]
        and parent["seed"] == candidate["seed"]
        and parent["hard_transport_pass"]
        and candidate["hard_transport_pass"]
        and int(candidate_reach["1"]) >= int(parent_reach["1"])
        and int(candidate_reach["2"]) >= int(parent_reach["2"])
        and candidate["crashes"] <= parent["crashes"]
        and (
            candidate["successes"] > parent["successes"]
            or int(candidate_reach["3"]) > int(parent_reach["3"])
        )
    )


def build_report(
    *,
    parent_path: Path,
    candidate_path: Path,
    preregistration: Path,
) -> dict[str, Any]:
    parent_raw = json.loads(parent_path.read_text())
    candidate_raw = json.loads(candidate_path.read_text())
    parent = summarize_count(parent_raw, num_gates=5)
    candidate = summarize_count(candidate_raw, num_gates=5)
    parent_identity = parent_raw.get("source_identity", {})
    candidate_identity = candidate_raw.get("source_identity", {})
    if (
        parent_identity.get("source_commit")
        != candidate_identity.get("source_commit")
    ):
        raise RuntimeError("VG045 paired source commits differ")
    admitted = confirmation_passes(parent, candidate)
    parent_reach = parent["gate_reach"]
    candidate_reach = candidate["gate_reach"]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": SCHEMA,
        "tag": "vq2_vg045_vg044_paired_count5_confirmation_001",
        "completed": True,
        "qualified_for_count11_diagnostic": admitted,
        "num_gates": 5,
        "source_commit": source_commit,
        "parent": parent,
        "candidate": candidate,
        "delta": {
            "successes": candidate["successes"] - parent["successes"],
            "crashes": candidate["crashes"] - parent["crashes"],
            "misses": candidate["misses"] - parent["misses"],
            "gate1_reach": (
                int(candidate_reach["1"]) - int(parent_reach["1"])
            ),
            "gate2_reach": (
                int(candidate_reach["2"]) - int(parent_reach["2"])
            ),
            "gate3_reach": (
                int(candidate_reach["3"]) - int(parent_reach["3"])
            ),
            "mean_gates_passed": (
                candidate["mean_gates_passed"]
                - parent["mean_gates_passed"]
            ),
        },
        "criteria": {
            "same_seed_and_episode_count": (
                parent["episodes"] == candidate["episodes"]
                and parent["seed"] == candidate["seed"]
            ),
            "both_hard_transport_pass": (
                parent["hard_transport_pass"]
                and candidate["hard_transport_pass"]
            ),
            "gate1_not_regressed": (
                int(candidate_reach["1"]) >= int(parent_reach["1"])
            ),
            "gate2_not_regressed": (
                int(candidate_reach["2"]) >= int(parent_reach["2"])
            ),
            "crash_count_not_regressed": (
                candidate["crashes"] <= parent["crashes"]
            ),
            "gate3_or_finish_strictly_improves": (
                candidate["successes"] > parent["successes"]
                or int(candidate_reach["3"]) > int(parent_reach["3"])
            ),
        },
        "artifact_sha256": {
            "parent_count_report": sha256_path(parent_path),
            "candidate_count_report": sha256_path(candidate_path),
            "preregistration": sha256_path(preregistration),
            "parent_manifest": sha256_path(
                ROOT
                / "docs/vq2_vg045_parent_count5_manifest_2026-07-31.json"
            ),
            "candidate_manifest": sha256_path(
                ROOT
                / "docs/vq2_vg045_candidate_count5_manifest_2026-07-31.json"
            ),
            "component": sha256_path(
                ROOT / "scripts/eval_vq2_staged_count5_component.py"
            ),
            "shared_summarizer": sha256_path(
                ROOT / "scripts/compare_vq2_staged_count5_diagnostic.py"
            ),
            "comparator": sha256_path(Path(__file__).resolve()),
            "vg044_admission": sha256_path(
                ROOT
                / "docs/vq2_vg044_checkpoint_line_bracket_admission_2026-07-31.json"
            ),
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
            "Preregister a fresh paired count-11 diagnostic against VG033."
            if admitted
            else "Reject VG044 without unchanged retry and retain VG033."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build_report(
        parent_path=args.parent_output.resolve() / "count_5.json",
        candidate_path=args.candidate_output.resolve() / "count_5.json",
        preregistration=args.preregistration.resolve(),
    )
    output = args.output.resolve()
    if output.is_file():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        if json.loads(output.read_text()) != report:
            raise RuntimeError("VG045 completed confirmation identity changed")
    else:
        write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified_for_count11_diagnostic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
