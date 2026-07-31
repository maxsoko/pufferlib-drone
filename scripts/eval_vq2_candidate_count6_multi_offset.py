#!/usr/bin/env python3
"""Screen one source-locked candidate on four independent six-gate offsets."""

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


MANIFEST_SCHEMA = "vq2_candidate_count6_multi_offset_manifest_v1"
OFFSETS = (0, 8, 16, 24)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 3072


def root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported six-gate candidate manifest")
    for name in (
        "tag",
        "report_schema",
        "checkpoint",
        "train_report",
        "train_report_schema",
        "train_report_tag",
        "admission",
        "admission_schema",
        "parent_baseline",
        "goal_prompt",
        "preregistration",
        "runner",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise RuntimeError(f"candidate manifest {name} is missing")
    for name in (
        "checkpoint_sha256",
        "train_report_sha256",
        "admission_sha256",
        "parent_baseline_sha256",
        "goal_prompt_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"candidate manifest {name} is not a SHA-256")
    for name in (
        "checkpoint",
        "train_report",
        "admission",
        "parent_baseline",
        "goal_prompt",
        "preregistration",
        "runner",
    ):
        root_path(manifest[name])
    if tuple(manifest.get("offsets", ())) != OFFSETS:
        raise RuntimeError("candidate six-gate offsets changed")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(OFFSETS):
        raise RuntimeError("candidate seeds must align with offsets")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise RuntimeError("candidate seed labels must be unique")
    exact = {
        "agents": AGENTS,
        "episodes_per_offset": EPISODES,
        "num_threads": THREADS,
        "max_steps": MAX_STEPS,
    }
    for name, expected in exact.items():
        if int(manifest.get(name, -1)) != expected:
            raise RuntimeError(f"candidate manifest {name} changed")
    return manifest


def verify_inputs(manifest: dict[str, Any]) -> None:
    expected = {
        root_path(manifest["checkpoint"]): manifest["checkpoint_sha256"],
        root_path(manifest["train_report"]): manifest["train_report_sha256"],
        root_path(manifest["admission"]): manifest["admission_sha256"],
        root_path(manifest["parent_baseline"]): manifest["parent_baseline_sha256"],
        root_path(manifest["goal_prompt"]): manifest["goal_prompt_sha256"],
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"candidate six-gate evidence mismatch: {path}")
    if not root_path(manifest["preregistration"]).is_file():
        raise RuntimeError("candidate six-gate preregistration is missing")
    train = json.loads(root_path(manifest["train_report"]).read_text())
    admission = json.loads(root_path(manifest["admission"]).read_text())
    baseline = json.loads(root_path(manifest["parent_baseline"]).read_text())
    if (
        train.get("schema") != manifest["train_report_schema"]
        or train.get("tag") != manifest["train_report_tag"]
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or train.get("safety", {}).get("actor_input_privileged_values") != 0
        or train.get("safety", {}).get("teacher_blend") != 0.0
        or train.get("safety", {}).get("flight_sim_packets_sent") != 0
        or train.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("candidate training report is not admitted")
    if (
        admission.get("schema") != manifest["admission_schema"]
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != manifest["checkpoint_sha256"]
        or admission.get("artifact_sha256", {}).get("report")
        != manifest["train_report_sha256"]
        or admission.get("live_authority")
        or admission.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("candidate admission does not authorize screening")
    if (
        baseline.get("schema")
        != "vq2_vg051_count6_multi_offset_baseline_evidence_v1"
        or not baseline.get("completed")
        or baseline.get("episodes") != EPISODES * len(OFFSETS)
        or baseline.get("conclusion", {}).get("retained_parent") != "VG033"
        or baseline.get("conclusion", {}).get("live_authority")
    ):
        raise RuntimeError("candidate parent six-gate baseline changed")


def configure_evaluator(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Any:
    original = evaluator.teacher_free_config
    checkpoint = root_path(manifest["checkpoint"])
    train_report = root_path(manifest["train_report"])
    evaluator.SCHEMA = "vq2_staged_gate_count_component_v1"
    evaluator.AGENTS = AGENTS
    evaluator.EPISODES_PER_COUNT = EPISODES
    evaluator.TOTAL_EPISODES = EPISODES
    evaluator.MINIMUM_SUCCESS_RATE = 0.0
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.CHECKPOINT = checkpoint
    evaluator.CHECKPOINT_SHA256 = manifest["checkpoint_sha256"]
    evaluator.TRAIN_REPORT = train_report
    evaluator.TRAIN_REPORT_SHA256 = manifest["train_report_sha256"]
    evaluator.PREREGISTRATION = root_path(manifest["preregistration"])
    evaluator.RUNNER = root_path(manifest["runner"])
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        manifest_path.resolve(),
        root_path(manifest["admission"]),
        root_path(manifest["parent_baseline"]),
        root_path(manifest["goal_prompt"]),
    )
    return original


def offset_config(original: Any, *, offset: int) -> Any:
    def config(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        values, overrides = original(pufferl_module, num_gates=num_gates)
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


def qualification(
    baseline: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, bool]:
    parent_reach = baseline["gate_reach"]
    candidate_reach = aggregate["gate_reach"]
    downstream_parent = (
        int(baseline["successes"]),
        int(parent_reach["6"]),
        int(parent_reach["5"]),
        int(parent_reach["4"]),
        int(parent_reach["3"]),
        float(baseline["mean_gates_passed"]),
    )
    downstream_candidate = (
        int(aggregate["successes"]),
        int(candidate_reach["6"]),
        int(candidate_reach["5"]),
        int(candidate_reach["4"]),
        int(candidate_reach["3"]),
        float(aggregate["mean_gates_passed"]),
    )
    return {
        "hard_transport_pass": bool(aggregate["all_hard_transport_pass"]),
        "gate1_not_regressed": int(candidate_reach["1"]) >= int(parent_reach["1"]),
        "gate2_not_regressed": int(candidate_reach["2"]) >= int(parent_reach["2"]),
        "crashes_not_regressed": int(aggregate["crashes"]) <= int(baseline["crashes"]),
        "strict_downstream_improvement": downstream_candidate > downstream_parent,
    }


def run(
    *,
    manifest_path: Path,
    output: Path,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    verify_inputs(manifest)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("candidate six-gate screen requires CUDA")
    original = configure_evaluator(manifest_path, manifest)
    identity = evaluator.source_identity()
    identity["episode_offsets"] = list(OFFSETS)
    state_path = output / "state.json"
    report_path = output / "report.json"
    state_identity = {
        "schema": "vq2_candidate_count6_multi_offset_state_v1",
        "tag": manifest["tag"],
        "offsets": list(OFFSETS),
        "seeds": list(manifest["seeds"]),
        "agents": AGENTS,
        "episodes_per_offset": EPISODES,
        "threads": THREADS,
        "max_steps": MAX_STEPS,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_identity": identity,
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("candidate completed source changed")
        return report
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"candidate state exists at {output}")
        state = json.loads(state_path.read_text())
        for key, value in state_identity.items():
            if state.get(key) != value:
                raise RuntimeError(f"candidate resume mismatch: {key}")
    else:
        if output.exists():
            raise RuntimeError("candidate output exists without state")
        output.mkdir(parents=True)
        state = {**state_identity, "status": "screening", "completed_offsets": []}
        write_json_atomic(state_path, state)

    items: list[dict[str, Any]] = []
    try:
        for offset, seed in zip(OFFSETS, manifest["seeds"], strict=True):
            path = output / f"offset_{offset}.json"
            evaluator.TAG = f"{manifest['tag']}_offset{offset}"
            evaluator.SEEDS = {6: int(seed)}
            evaluator.teacher_free_config = offset_config(original, offset=offset)
            if path.is_file():
                count = json.loads(path.read_text())
                if count.get("source_identity") != identity:
                    raise RuntimeError(f"candidate offset {offset} changed")
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
                "seed": int(seed),
                "count_path": path.name,
                "count_sha256": sha256_path(path),
                "summary": summary,
            })
            state["completed_offsets"] = [item["offset"] for item in items]
            write_json_atomic(state_path, state)
    finally:
        evaluator.teacher_free_config = original

    total = EPISODES * len(OFFSETS)
    aggregate = {
        "episodes": total,
        "successes": sum(item["summary"]["successes"] for item in items),
        "crashes": sum(item["summary"]["crashes"] for item in items),
        "misses": sum(item["summary"]["misses"] for item in items),
        "gate_reach": {
            str(gate): sum(
                int(item["summary"]["gate_reach"][str(gate)]) for item in items
            )
            for gate in range(1, 7)
        },
        "mean_gates_passed": sum(
            item["summary"]["mean_gates_passed"] * EPISODES for item in items
        ) / total,
        "all_hard_transport_pass": all(
            item["summary"]["hard_transport_pass"] for item in items
        ),
    }
    baseline = json.loads(root_path(manifest["parent_baseline"]).read_text())
    predicates = qualification(baseline, aggregate)
    qualified = all(predicates.values())
    report = {
        "schema": manifest["report_schema"],
        "tag": manifest["tag"],
        "completed": True,
        "qualified": qualified,
        "official_proxy_gate_count": 6,
        "offsets": list(OFFSETS),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "parent_baseline_sha256": manifest["parent_baseline_sha256"],
        "aggregate": aggregate,
        "parent_aggregate": {
            "episodes": baseline["episodes"],
            "successes": baseline["successes"],
            "crashes": baseline["crashes"],
            "misses": baseline["misses"],
            "gate_reach": baseline["gate_reach"],
            "mean_gates_passed": baseline["mean_gates_passed"],
            "all_hard_transport_pass": baseline["all_hard_transport_pass"],
        },
        "qualification_predicates": predicates,
        "items": items,
        "source_identity": identity,
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Preregister a larger independent six-gate admission screen."
            if qualified
            else "Reject the full update and preregister a checkpoint-fraction bracket."
        ),
    }
    write_json_once(report_path, report)
    state.update({
        "status": "qualified" if qualified else "rejected",
        "completed_offsets": list(OFFSETS),
        "report_sha256": sha256_path(report_path),
    })
    write_json_atomic(state_path, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        manifest_path=args.manifest,
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
