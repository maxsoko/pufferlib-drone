#!/usr/bin/env python3
"""Run one source-locked side of a paired count-5 policy diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator


MANIFEST_SCHEMA = "vq2_staged_count5_component_manifest_v1"
COUNT = 5
MAX_EXECUTED_ACTION_ERROR = 5e-5


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported count-5 component manifest")
    if manifest.get("role") not in {"parent", "candidate"}:
        raise RuntimeError("count-5 component role must be parent or candidate")
    if not isinstance(manifest.get("tag"), str) or not manifest["tag"]:
        raise RuntimeError("count-5 component tag is missing")
    if manifest.get("num_gates") != COUNT:
        raise RuntimeError("count-5 component must use exactly five gates")
    agents = int(manifest.get("agents", 0))
    episodes = int(manifest.get("episodes", 0))
    if agents not in range(8, 65) or episodes != agents:
        raise RuntimeError("count-5 component requires one episode per 8--64 agents")
    if int(manifest.get("num_threads", 0)) not in {4, 32}:
        raise RuntimeError("count-5 component threads must retain a parity-proven rung")
    if int(manifest.get("max_steps", 0)) not in range(2048, 4097):
        raise RuntimeError("count-5 component max_steps must be in [2048,4096]")
    if int(manifest.get("seed", 0)) <= 429120:
        raise RuntimeError("count-5 component seed is not fresh")
    required_paths = (
        "checkpoint",
        "train_report",
        "admission",
        "preregistration",
        "runner",
        "goal_prompt",
    )
    for name in required_paths:
        if not isinstance(manifest.get(name), str):
            raise RuntimeError(f"count-5 component {name} path is missing")
        _root_path(manifest[name])
    for name in (
        "checkpoint_sha256",
        "train_report_sha256",
        "admission_sha256",
        "goal_prompt_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"count-5 component {name} is not a SHA-256")
    return manifest


def verify_actor_evidence(manifest: dict[str, Any]) -> None:
    checkpoint = _root_path(manifest["checkpoint"])
    train_report_path = _root_path(manifest["train_report"])
    admission_path = _root_path(manifest["admission"])
    goal_prompt = _root_path(manifest["goal_prompt"])
    expected = {
        checkpoint: manifest["checkpoint_sha256"],
        train_report_path: manifest["train_report_sha256"],
        admission_path: manifest["admission_sha256"],
        goal_prompt: manifest["goal_prompt_sha256"],
    }
    for path, digest in expected.items():
        if evaluator.sha256_path(path) != digest:
            raise RuntimeError(f"count-5 component evidence hash mismatch: {path}")

    train_report = json.loads(train_report_path.read_text())
    admission = json.loads(admission_path.read_text())
    if (
        train_report.get("schema") != manifest.get("train_report_schema")
        or train_report.get("tag") != manifest.get("train_report_tag")
        or not train_report.get("completed")
        or not train_report.get("numerically_admitted")
        or train_report.get("checkpoint_sha256")
        != manifest["checkpoint_sha256"]
        or train_report.get("minimum_transition_window_exposure", 0.0) < 4.0
        or not train_report.get("equal_source_weight_audit")
    ):
        raise RuntimeError("count-5 component training report is not admitted")
    train_safety = train_report.get("safety", {})
    if (
        train_safety.get("actor_input_privileged_values") != 0
        or train_safety.get("teacher_blend") != 0.0
        or train_safety.get("flight_sim_packets_sent") != 0
        or train_safety.get("sealed_test_accesses") != 0
    ):
        raise RuntimeError("count-5 component training safety contract changed")
    if (
        admission.get("schema") != manifest.get("admission_schema")
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != manifest["checkpoint_sha256"]
        or admission.get("artifact_sha256", {}).get("report")
        != manifest["train_report_sha256"]
    ):
        raise RuntimeError("count-5 component admission does not bind the actor")
    admission_safety = admission.get("safety", {})
    if (
        admission_safety.get("actor_input_privileged_values") != 0
        or admission_safety.get("teacher_blend") != 0.0
        or admission_safety.get("flight_sim_packets_sent") != 0
        or admission_safety.get("sealed_test_accesses") != 0
        or admission_safety.get("submission_authorized")
    ):
        raise RuntimeError("count-5 component admission safety contract changed")


def configure_evaluator(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    generic_config = evaluator.teacher_free_config
    agents = int(manifest["agents"])
    episodes = int(manifest["episodes"])
    threads = int(manifest["num_threads"])
    max_steps = int(manifest["max_steps"])

    def diagnostic_config(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        config, overrides = generic_config(pufferl_module, num_gates=num_gates)
        config["vec"]["num_threads"] = threads
        config["env"]["max_steps"] = max_steps
        return config, [
            *overrides,
            "--vec.num-threads",
            str(threads),
            "--env.max-steps",
            str(max_steps),
        ]

    evaluator.TAG = manifest["tag"]
    evaluator.SCHEMA = "vq2_staged_count5_component_v1"
    evaluator.COUNTS = (COUNT,)
    evaluator.AGENTS = agents
    evaluator.EPISODES_PER_COUNT = episodes
    evaluator.TOTAL_EPISODES = episodes
    evaluator.SEEDS = {COUNT: int(manifest["seed"])}
    evaluator.MINIMUM_SUCCESS_RATE = 0.0
    evaluator.MAX_EXECUTED_ACTION_ERROR = MAX_EXECUTED_ACTION_ERROR
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.CHECKPOINT = _root_path(manifest["checkpoint"])
    evaluator.CHECKPOINT_SHA256 = manifest["checkpoint_sha256"]
    evaluator.TRAIN_REPORT = _root_path(manifest["train_report"])
    evaluator.TRAIN_REPORT_SHA256 = manifest["train_report_sha256"]
    evaluator.PREREGISTRATION = _root_path(manifest["preregistration"])
    evaluator.RUNNER = _root_path(manifest["runner"])
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        manifest_path.resolve(),
        _root_path(manifest["admission"]),
        _root_path(manifest["goal_prompt"]),
    )
    evaluator.teacher_free_config = diagnostic_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    verify_actor_evidence(manifest)
    configure_evaluator(manifest_path, manifest)
    report = evaluator.run_admission(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    if not report.get("completed") or report.get("total_episodes") != manifest["episodes"]:
        raise RuntimeError("count-5 component did not complete its fixed episodes")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
