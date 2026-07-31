#!/usr/bin/env python3
"""Measure VG033 on the official six-gate proxy across four offsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compare_vq2_staged_count5_diagnostic as summarizer
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from scripts.eval_vq2_variable_gate_oracle import (
    sha256_path,
    write_json_atomic,
    write_json_once,
)


TAG = "vq2_vg051_vg033_count6_multi_offset_baseline_001"
SCHEMA = "vq2_count6_multi_offset_baseline_v1"
OFFSETS = (0, 8, 16, 24)
SEEDS = (429165, 429166, 429167, 429168)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 3072
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
)
ADMISSION = ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
ADMISSION_SHA256 = (
    "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
)
REJECTION = ROOT / "docs/vq2_vg050_offset24_paired_count5_rejection_2026-07-31.json"
REJECTION_SHA256 = (
    "0cd7e8238017697e2a4ec3d78dc4193cdc0c180790789aba82cdeff252094029"
)
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
PREREGISTRATION = ROOT / "docs/vq2_vg051_count6_multi_offset_baseline_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg051_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        ADMISSION: ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG051 input hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG051 preregistration is missing")
    rejection = json.loads(REJECTION.read_text())
    if (
        rejection.get("schema")
        != "vq2_vg050_offset24_paired_count5_rejection_v1"
        or rejection.get("qualified_for_count11_diagnostic")
    ):
        raise RuntimeError("VG050 does not authorize VG051")


def configure_evaluator() -> Any:
    original = evaluator.teacher_free_config
    evaluator.SCHEMA = "vq2_staged_gate_count_component_v1"
    evaluator.AGENTS = AGENTS
    evaluator.EPISODES_PER_COUNT = EPISODES
    evaluator.TOTAL_EPISODES = EPISODES
    evaluator.MINIMUM_SUCCESS_RATE = 0.0
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        ADMISSION,
        REJECTION,
        GOAL,
    )
    return original


def offset_config(
    original: Any,
    *,
    offset: int,
) -> Any:
    def config(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        values, overrides = original(
            pufferl_module, num_gates=num_gates
        )
        values["vec"]["num_threads"] = THREADS
        values["env"]["max_steps"] = MAX_STEPS
        values["env"]["evaluation_episode_offset"] = offset
        return values, [
            *overrides,
            "--vec.num-threads",
            str(THREADS),
            "--env.max-steps",
            str(MAX_STEPS),
            "--env.evaluation-episode-offset",
            str(offset),
        ]

    return config


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG051 requires CUDA inference")
    original = configure_evaluator()
    identity = evaluator.source_identity()
    identity["episode_offsets"] = list(OFFSETS)
    state_path = output / "state.json"
    report_path = output / "report.json"
    state_identity = {
        "schema": "vq2_count6_multi_offset_state_v1",
        "tag": TAG,
        "offsets": list(OFFSETS),
        "seeds": list(SEEDS),
        "agents": AGENTS,
        "episodes_per_offset": EPISODES,
        "threads": THREADS,
        "max_steps": MAX_STEPS,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_identity": identity,
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("VG051 completed source changed")
        return report
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG051 state exists at {output}")
        state = json.loads(state_path.read_text())
        for key, value in state_identity.items():
            if state.get(key) != value:
                raise RuntimeError(f"VG051 resume mismatch: {key}")
    else:
        if output.exists():
            raise RuntimeError("VG051 output exists without state")
        output.mkdir(parents=True)
        state = {**state_identity, "status": "screening", "completed_offsets": []}
        write_json_atomic(state_path, state)

    items: list[dict[str, Any]] = []
    try:
        for offset, seed in zip(OFFSETS, SEEDS, strict=True):
            path = output / f"offset_{offset}.json"
            evaluator.TAG = f"{TAG}_offset{offset}"
            evaluator.SEEDS = {6: seed}
            evaluator.teacher_free_config = offset_config(
                original, offset=offset
            )
            if path.is_file():
                count = json.loads(path.read_text())
                if count.get("source_identity") != identity:
                    raise RuntimeError(f"VG051 offset {offset} changed")
            else:
                count = evaluator.run_count(
                    num_gates=6,
                    device=torch.device(device_name),
                    source_identity=identity,
                )
                write_json_once(path, count)
            summary = summarizer.summarize_count(count, num_gates=6)
            items.append({
                "offset": offset,
                "seed": seed,
                "count_path": path.name,
                "count_sha256": sha256_path(path),
                "summary": summary,
            })
            state["completed_offsets"] = [item["offset"] for item in items]
            write_json_atomic(state_path, state)
    finally:
        evaluator.teacher_free_config = original

    total_episodes = EPISODES * len(OFFSETS)
    aggregate = {
        "episodes": total_episodes,
        "successes": sum(item["summary"]["successes"] for item in items),
        "crashes": sum(item["summary"]["crashes"] for item in items),
        "misses": sum(item["summary"]["misses"] for item in items),
        "gate_reach": {
            str(gate): sum(
                int(item["summary"]["gate_reach"][str(gate)])
                for item in items
            )
            for gate in range(1, 7)
        },
        "mean_gates_passed": sum(
            item["summary"]["mean_gates_passed"] * EPISODES
            for item in items
        ) / total_episodes,
        "all_hard_transport_pass": all(
            item["summary"]["hard_transport_pass"] for item in items
        ),
    }
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "official_proxy_gate_count": 6,
        "offsets": list(OFFSETS),
        "items": items,
        "aggregate": aggregate,
        "source_identity": identity,
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": "Use this multi-offset count-6 baseline to preregister a rollout-aware repair; no live authority.",
    }
    write_json_once(report_path, report)
    state.update({
        "status": "completed",
        "report_sha256": sha256_path(report_path),
    })
    write_json_atomic(state_path, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["aggregate"]["all_hard_transport_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
