#!/usr/bin/env python3
"""Confirm VG044 against VG033 on a non-aliased offset-8 fixture."""

from __future__ import annotations

import argparse
import hashlib
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


SCHEMA = "vq2_vg046_offset8_paired_count5_confirmation_v1"
TAG = "vq2_vg046_vg044_offset8_paired_count5_confirmation_001"
EPISODE_OFFSET = 8
VG044_PARENT_BEHAVIOR_SHA256 = (
    "d0db42b0c5eb4ddfe4d4b0d3e3165c55d00fbc1d88688d18d9a2fd494fa2f764"
)
VG044_CANDIDATE_BEHAVIOR_SHA256 = (
    "fc25942e5f8821c2a55d8ec8c2f3bd8a851c55a2c3e634791761f8ad2daaaf43"
)
BEHAVIOR_FIELDS = (
    "metrics",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_samples",
    "maximum_held_public_index_distribution",
    "maximum_raw_public_index_distribution",
    "native_public_phase_mirror_index",
    "phase_samples",
    "status_hold_steps",
    "vector_steps",
    "raw_phase_encoding_max_error",
    "executed_action_max_error",
    "action_envelope_violations",
    "phase_changes_off_tick",
    "phase_decreases",
    "phase_skips",
    "nonfinite_action",
)


def behavior_sha256(report: dict[str, Any]) -> str:
    projection = {name: report[name] for name in BEHAVIOR_FIELDS}
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def report_uses_offset(report: dict[str, Any], offset: int) -> bool:
    overrides = report.get("loader_overrides", [])
    return any(
        overrides[index:index + 2]
        == ["--env.evaluation-episode-offset", str(offset)]
        for index in range(max(len(overrides) - 1, 0))
    )


def confirmation_passes(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    parent_offset_pass: bool,
    candidate_offset_pass: bool,
    parent_distinct: bool,
    candidate_distinct: bool,
) -> bool:
    parent_reach = parent["gate_reach"]
    candidate_reach = candidate["gate_reach"]
    return bool(
        parent["episodes"] == candidate["episodes"]
        and parent["seed"] == candidate["seed"]
        and parent_offset_pass
        and candidate_offset_pass
        and parent_distinct
        and candidate_distinct
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
    if (
        parent_raw.get("source_identity", {}).get("source_commit")
        != candidate_raw.get("source_identity", {}).get("source_commit")
    ):
        raise RuntimeError("VG046 paired source commits differ")
    parent_behavior = behavior_sha256(parent_raw)
    candidate_behavior = behavior_sha256(candidate_raw)
    parent_offset_pass = report_uses_offset(parent_raw, EPISODE_OFFSET)
    candidate_offset_pass = report_uses_offset(
        candidate_raw, EPISODE_OFFSET
    )
    parent_distinct = parent_behavior != VG044_PARENT_BEHAVIOR_SHA256
    candidate_distinct = (
        candidate_behavior != VG044_CANDIDATE_BEHAVIOR_SHA256
    )
    admitted = confirmation_passes(
        parent,
        candidate,
        parent_offset_pass=parent_offset_pass,
        candidate_offset_pass=candidate_offset_pass,
        parent_distinct=parent_distinct,
        candidate_distinct=candidate_distinct,
    )
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
        "tag": TAG,
        "completed": True,
        "qualified_for_count11_diagnostic": admitted,
        "num_gates": 5,
        "episode_offset": EPISODE_OFFSET,
        "source_commit": source_commit,
        "parent": parent,
        "candidate": candidate,
        "behavior_projection": {
            "fields": list(BEHAVIOR_FIELDS),
            "vg044_parent_sha256": VG044_PARENT_BEHAVIOR_SHA256,
            "current_parent_sha256": parent_behavior,
            "vg044_candidate_sha256": VG044_CANDIDATE_BEHAVIOR_SHA256,
            "current_candidate_sha256": candidate_behavior,
            "parent_distinct_from_vg044": parent_distinct,
            "candidate_distinct_from_vg044": candidate_distinct,
        },
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
            "both_offset8": parent_offset_pass and candidate_offset_pass,
            "both_distinct_from_vg044": (
                parent_distinct and candidate_distinct
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
                / "docs/vq2_vg046_parent_count5_manifest_2026-07-31.json"
            ),
            "candidate_manifest": sha256_path(
                ROOT
                / "docs/vq2_vg046_candidate_count5_manifest_2026-07-31.json"
            ),
            "component": sha256_path(
                ROOT / "scripts/eval_vq2_staged_count5_component.py"
            ),
            "shared_summarizer": sha256_path(
                ROOT / "scripts/compare_vq2_staged_count5_diagnostic.py"
            ),
            "comparator": sha256_path(Path(__file__).resolve()),
            "vg045_rejection": sha256_path(
                ROOT / "docs/vq2_vg045_seed_alias_rejection_2026-07-31.json"
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
            raise RuntimeError("VG046 completed confirmation identity changed")
    else:
        write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified_for_count11_diagnostic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
