#!/usr/bin/env python3
"""Screen smaller VG033->VG042 update fractions on offset-8 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import scripts.eval_vq2_vg044_checkpoint_line_bracket as base


ROOT = base.ROOT
TAG = "vq2_vg047_vg033_to_vg042_offset8_fraction_bracket_001"
SCHEMA = "vq2_checkpoint_line_offset8_bracket_report_v1"
STATE_SCHEMA = "vq2_checkpoint_line_offset8_bracket_state_v1"
SEED = 429161
EPISODE_OFFSET = 8
AGENTS = 64
EPISODES = 64
NUM_THREADS = 4
MAX_STEPS = 2560
ALPHAS = (0.0, 0.025, 0.04, 0.05, 0.06, 0.075, 0.085, 0.10)
VG044_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg044_vg033_to_vg042_checkpoint_line_bracket_001/report.json"
)
VG044_REPORT_SHA256 = (
    "c6fb4544265342e2b33974196d25dba5dba4f18fd1039691f7f9ed2155b9a84f"
)
VG044_ADMISSION = (
    ROOT / "docs/vq2_vg044_checkpoint_line_bracket_admission_2026-07-31.json"
)
VG044_ADMISSION_SHA256 = (
    "85f1baf2fb5c053ac9c0f908dc0170a43aae126eaf61a1e1ccf3110dc35c0fb1"
)
REJECTION = (
    ROOT
    / "docs/vq2_vg046_offset8_paired_count5_rejection_2026-07-31.json"
)
REJECTION_SHA256 = (
    "3049a97ad216a90855a2b3ebeba199782390b09df87307d2f2d09e8bebf06055"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg047_offset8_fraction_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg047_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        base.PARENT_CHECKPOINT: base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT: base.PARENT_REPORT_SHA256,
        base.PARENT_ADMISSION: base.PARENT_ADMISSION_SHA256,
        base.UPDATE_CHECKPOINT: base.UPDATE_CHECKPOINT_SHA256,
        base.UPDATE_REPORT: base.UPDATE_REPORT_SHA256,
        base.UPDATE_ADMISSION: base.UPDATE_ADMISSION_SHA256,
        VG044_REPORT: VG044_REPORT_SHA256,
        VG044_ADMISSION: VG044_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        base.GOAL_PROMPT: base.GOAL_PROMPT_SHA256,
    }
    for path, digest in expected.items():
        if base.sha256_path(path) != digest:
            raise RuntimeError(f"VG047 input hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG047 preregistration is missing")
    report = json.loads(VG044_REPORT.read_text())
    rejection = json.loads(REJECTION.read_text())
    alpha005 = next(
        (
            item
            for item in report.get("candidates", [])
            if float(item.get("alpha", -1.0)) == 0.05
        ),
        None,
    )
    if (
        report.get("schema") != "vq2_checkpoint_line_bracket_report_v1"
        or not report.get("numerically_admitted")
        or alpha005 is None
        or not base.qualifies(report["parent"], alpha005["summary"])
    ):
        raise RuntimeError("VG044 does not authorize the smaller bracket")
    if (
        rejection.get("schema")
        != "vq2_vg046_offset8_paired_count5_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("qualified_for_count11_diagnostic")
        or rejection.get("criteria", {}).get("crash_count_not_regressed")
        or not rejection.get("criteria", {}).get(
            "gate3_or_finish_strictly_improves"
        )
    ):
        raise RuntimeError("VG046 rejection does not authorize VG047")
    base.load_endpoints()


def configure_evaluator() -> Any:
    original_config = base.evaluator.teacher_free_config

    def bracket_config(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        config, overrides = original_config(
            pufferl_module, num_gates=num_gates
        )
        config["vec"]["num_threads"] = NUM_THREADS
        config["env"]["max_steps"] = MAX_STEPS
        config["env"]["evaluation_episode_offset"] = EPISODE_OFFSET
        return config, [
            *overrides,
            "--vec.num-threads",
            str(NUM_THREADS),
            "--env.max-steps",
            str(MAX_STEPS),
            "--env.evaluation-episode-offset",
            str(EPISODE_OFFSET),
        ]

    base.evaluator.SCHEMA = "vq2_staged_count5_component_v1"
    base.evaluator.AGENTS = AGENTS
    base.evaluator.EPISODES_PER_COUNT = EPISODES
    base.evaluator.TOTAL_EPISODES = EPISODES
    base.evaluator.SEEDS = {5: SEED}
    base.evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    base.evaluator.teacher_free_config = bracket_config
    return original_config


def source_identity() -> dict[str, Any]:
    identity = _ORIGINAL_SOURCE_IDENTITY()
    sources = dict(identity["source_sha256"])
    additions = (
        Path(__file__).resolve(),
        VG044_REPORT,
        VG044_ADMISSION,
    )
    sources.update({
        str(path.relative_to(ROOT)): base.sha256_path(path)
        for path in additions
    })
    identity["source_sha256"] = sources
    identity["episode_offset"] = EPISODE_OFFSET
    return identity


def configure_contract() -> None:
    assignments = {
        "TAG": TAG,
        "SCHEMA": SCHEMA,
        "STATE_SCHEMA": STATE_SCHEMA,
        "SEED": SEED,
        "AGENTS": AGENTS,
        "EPISODES": EPISODES,
        "NUM_THREADS": NUM_THREADS,
        "MAX_STEPS": MAX_STEPS,
        "ALPHAS": ALPHAS,
        "REJECTION": REJECTION,
        "REJECTION_SHA256": REJECTION_SHA256,
        "PREREGISTRATION": PREREGISTRATION,
        "RUNNER": RUNNER,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "verify_inputs": verify_inputs,
        "configure_evaluator": configure_evaluator,
        "source_identity": source_identity,
    }
    for name, value in assignments.items():
        setattr(base, name, value)


_ORIGINAL_SOURCE_IDENTITY = base.source_identity


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    configure_contract()
    report = base.run(
        output=output,
        device_name=device_name,
        resume=resume,
    )
    if (
        report.get("source_identity", {}).get("episode_offset")
        != EPISODE_OFFSET
    ):
        raise RuntimeError("VG047 report episode offset changed")
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
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
