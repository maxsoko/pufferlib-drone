#!/usr/bin/env python3
"""Batch-screen cumulative VG071 head rollbacks on the 24-gate course."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc015_restore_vg071_head1 as base


TAG = "vq2_lc021_cumulative_head_rollback_ladder_001"
SCHEMA = "vq2_lc021_cumulative_head_rollback_ladder_report_v1"
CHILD_SCHEMA = "vq2_lc021_cumulative_head_rollback_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc021_cumulative_head_rollback_checkpoint_v1"
ENDPOINTS = (4, 5, 6, 8, 12, 16, 24, 32)
SEED = 431210
LC020_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc020_restore_vg071_heads1_3_001/report.json"
)
LC020_REPORT_SHA256 = (
    "bcaa349cbb9bbfb7f4fe7c3e202815e8efcbfc8544737aefb04619a070b24103"
)
LC020_CHECKPOINT = LC020_REPORT.parent / "policy_selected.pt"
LC020_CHECKPOINT_SHA256 = (
    "98f132d797a07f6dfb903d10c4b202cca8823bfdbddc39f26b9698a945b9b7d0"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc021_cumulative_head_rollback_ladder_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc021_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
ORIGINAL_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    ORIGINAL_VERIFY_INPUTS()
    expected = {
        LC020_REPORT: LC020_REPORT_SHA256,
        LC020_CHECKPOINT: LC020_CHECKPOINT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC021 bound input changed: {path}")
    frontier = json.loads(LC020_REPORT.read_text())
    if (
        frontier.get("schema") != "vq2_lc020_restore_vg071_heads1_3_report_v1"
        or not frontier.get("numerically_admitted")
        or frontier.get("mean_gates_passed") != 2.8125
        or frontier.get("maximum_raw_index") != 5
        or frontier.get("crash_rate") != 0.1875
        or frontier.get("checkpoint_sha256") != LC020_CHECKPOINT_SHA256
        or frontier.get("safety", {}).get("flight_sim_packets_sent") != 0
        or frontier.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC020 does not authorize the rollback ladder")


def configure(endpoint: int) -> None:
    base.TAG = f"{TAG}_through_{endpoint:02d}"
    base.SCHEMA = CHILD_SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.RESTORED_PHASES = tuple(range(1, endpoint + 1))
    base.MIN_MEAN_GATES = 2.8125
    base.MIN_MAXIMUM_INDEX = 5
    base.MAX_CRASH_RATE = 0.50
    base.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC020_REPORT, LC020_CHECKPOINT,
    )
    base.verify_inputs = verify_inputs


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    if resume and report_path.is_file():
        return json.loads(report_path.read_text())

    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        configure(endpoint)
        child_output = output / f"through_{endpoint:02d}"
        child = base.run(
            output=child_output, device_name=device_name, resume=resume
        )
        identities.append(child["source_identity"])
        items.append({
            "restored_through_phase": endpoint,
            "candidate_state_sha256": child["candidate_state_sha256"],
            "mean_gates_passed": child["mean_gates_passed"],
            "maximum_raw_index": child["maximum_raw_index"],
            "maximum_raw_index_distribution": child[
                "maximum_raw_index_distribution"
            ],
            "crash_rate": child["crash_rate"],
            "miss_rate": child["miss_rate"],
            "timeout_rate": child["timeout_rate"],
            "diagnostic_valid": child["diagnostic_valid"],
            "numerically_admitted": child["numerically_admitted"],
            "checkpoint": (
                str(Path(f"through_{endpoint:02d}") / child["checkpoint"])
                if child["checkpoint"] else None
            ),
            "checkpoint_sha256": child["checkpoint_sha256"],
            "child_report_sha256": sha256_path(child_output / "report.json"),
        })

    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("LC021 source identity changed within the ladder")
    admitted = [item for item in items if item["numerically_admitted"]]
    selected = max(
        admitted,
        key=lambda item: (
            item["mean_gates_passed"], item["maximum_raw_index"],
            -item["crash_rate"], -item["restored_through_phase"],
        ),
        default=None,
    )
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "diagnostic_valid": all(item["diagnostic_valid"] for item in items),
        "numerically_admitted": selected is not None,
        "paired_seed": SEED,
        "episodes_per_candidate": base.EPISODES,
        "num_gates": base.NUM_GATES,
        "items": items,
        "selected": selected,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identities[0],
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Retain the selected rollback and collect only its first weak phase."
            if selected else "Reject LC021 and retain LC020."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
