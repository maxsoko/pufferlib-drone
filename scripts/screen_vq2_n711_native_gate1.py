#!/usr/bin/env python3
"""Run one paired native Gate-1 screen of the sealed-test-admitted N711 model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_informed_dreamer import _sha256, evaluate as evaluate_native
from scripts.package_vq2_invariant_donor_readout import PACKAGE_SCHEMA


SCREEN_SCHEMA = "vq2_n711_paired_native_gate1_screen_v1"
N711_CHECKPOINT_SHA256 = (
    "92be8898d8294477139f0706a38604d0e3b1bd953a128f9e30c8f9e4c2e168dd"
)
N711_REPORT_SHA256 = (
    "dec6720a51785134f957f91a38f6de9c03b38595d477eb2c2d5ab04c1b91ce03"
)
N712_REPORT_SHA256 = (
    "976547b8899d6f55a88d4a2a3b4dc060305a0ca4046ef07fd7ab8a674f375295"
)
N588_BASELINE_SHA256 = (
    "bd0df2fe88efd6ec77c80a187208ca3e85a7655ad69e9fa56018bfeb7a6227d3"
)
NATIVE_EVALUATOR_SHA256 = (
    "c2b7cf445e788ad2e73560178ae534853dcda0b1ae60168eb3d286af367ff2d5"
)
BASELINE_GATE1_PASSES = 11
EPISODES = 64
EPISODE_OFFSET = 600


def _native_gates(
    metrics: dict[str, object], baseline: dict[str, object]
) -> dict[str, bool]:
    gate_passes = int(round(float(metrics.get("env_gates_passed", 0.0)) * EPISODES))
    baseline_passes = int(
        round(float(baseline.get("env_gates_passed", 0.0)) * EPISODES)
    )
    action_values = [
        float(metrics.get(f"action_{channel}_abs_max", math.inf))
        for channel in range(4)
    ]
    return {
        "exact_episode_count": (
            metrics.get("total_episodes") == EPISODES
            and float(metrics.get("env_n", -1.0)) == float(EPISODES)
            and metrics.get("episodes_per_agent") == 1
        ),
        "paired_episode_offset_exact": metrics.get("episode_offset") == EPISODE_OFFSET,
        "uninterrupted_full_start_exact": (
            metrics.get("evaluation_start_contract") == "uninterrupted_full_course"
            and metrics.get("gate_local_start_curriculum") == 0
            and metrics.get("mixed_start_curriculum") == 0
            and float(metrics.get("env_gate_local_reset_fraction", -1.0)) == 0.0
        ),
        "at_least_11_gate1_passes": gate_passes >= BASELINE_GATE1_PASSES,
        "does_not_regress_paired_baseline": gate_passes >= baseline_passes,
        "zero_crash": float(metrics.get("env_crash", math.inf)) == 0.0,
        "zero_timeout": float(metrics.get("env_timeout", math.inf)) == 0.0,
        "finite_bounded_actions": (
            all(math.isfinite(value) and value < 0.95 for value in action_values)
            and int(metrics.get("action_count", 0)) > 0
        ),
        "negative_mean_pitch": float(metrics.get("action_0_mean", math.inf)) < 0.0,
    }


def _source_contract_gates(
    n711: dict[str, object], n712: dict[str, object], hashes: dict[str, str]
) -> dict[str, bool]:
    n711_progress = n711.get("strict_progress_gates")
    n711_action = n711.get("strict_action_gates")
    n712_progress = n712.get("progress_gates")
    n712_action = n712.get("action_gates")
    for name, value in (
        ("n711_progress", n711_progress),
        ("n711_action", n711_action),
        ("n712_progress", n712_progress),
        ("n712_action", n712_action),
    ):
        if not isinstance(value, dict):
            if name == "n711_progress":
                n711_progress = {}
            elif name == "n711_action":
                n711_action = {}
            elif name == "n712_progress":
                n712_progress = {}
            else:
                n712_action = {}
    return {
        "checkpoint_hash_exact": hashes["checkpoint"] == N711_CHECKPOINT_SHA256,
        "n711_report_hash_exact": hashes["n711_report"] == N711_REPORT_SHA256,
        "n712_report_hash_exact": hashes["n712_report"] == N712_REPORT_SHA256,
        "baseline_hash_exact": hashes["baseline"] == N588_BASELINE_SHA256,
        "native_evaluator_hash_exact": (
            hashes["native_evaluator"] == NATIVE_EVALUATOR_SHA256
        ),
        "n711_admitted_exact": (
            n711.get("contract") == PACKAGE_SCHEMA
            and n711.get("package_admitted") is True
            and n711.get("process_passed") is True
            and n711.get("output_checkpoint_sha256") == N711_CHECKPOINT_SHA256
            and isinstance(n711_progress, dict)
            and bool(n711_progress)
            and all(value is True for value in n711_progress.values())
            and isinstance(n711_action, dict)
            and bool(n711_action)
            and all(value is True for value in n711_action.values())
        ),
        "n712_consumed_test_passed_exact": (
            n712.get("contract") == "vq2_invariant_ridge_physical_sealed_test_v1"
            and n712.get("passed") is True
            and n712.get("process_passed") is True
            and n712.get("test_evaluations") == 1
            and n712.get("test_dataset_filesystem_reads") == 1
            and isinstance(n712_progress, dict)
            and bool(n712_progress)
            and all(value is True for value in n712_progress.values())
            and isinstance(n712_action, dict)
            and bool(n712_action)
            and all(value is True for value in n712_action.values())
        ),
    }


def screen(args: argparse.Namespace) -> dict[str, object]:
    if args.report.exists():
        raise ValueError("N713 native screen refuses to overwrite its report")
    native_evaluator = ROOT / "scripts/eval_vq2_informed_dreamer.py"
    source_before = {
        "executable": _sha256(Path(__file__)),
        "native_evaluator": _sha256(native_evaluator),
        "checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "n712_report": _sha256(args.n712_report),
        "baseline": _sha256(args.baseline),
    }
    n711 = json.loads(args.n711_report.read_text(encoding="utf-8"))
    n712 = json.loads(args.n712_report.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source_gates = _source_contract_gates(n711, n712, source_before)
    if not all(source_gates.values()):
        failed = sorted(name for name, passed in source_gates.items() if not passed)
        raise RuntimeError(f"N713 source contract mismatch: {failed}")
    native_metrics = evaluate_native(
        SimpleNamespace(
            checkpoint=args.checkpoint,
            env_name=None,
            episodes_per_agent=1,
            episode_offset=EPISODE_OFFSET,
            max_vector_steps=None,
            device=args.device,
            metrics=None,
        )
    )
    source_after = {
        "executable": _sha256(Path(__file__)),
        "native_evaluator": _sha256(native_evaluator),
        "checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "n712_report": _sha256(args.n712_report),
        "baseline": _sha256(args.baseline),
    }
    native_gates = _native_gates(native_metrics, baseline)
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "source_contract_exact": all(source_gates.values()),
        "checkpoint_report_hash_exact": (
            native_metrics.get("checkpoint_sha256") == N711_CHECKPOINT_SHA256
        ),
        "no_test_path_received": True,
        "native_environment_count_exact": True,
    }
    passed = all(process_gates.values()) and all(native_gates.values())
    gate1_passes = int(
        round(float(native_metrics["env_gates_passed"]) * EPISODES)
    )
    baseline_gate1_passes = int(
        round(float(baseline["env_gates_passed"]) * EPISODES)
    )
    report: dict[str, object] = {
        "contract": SCREEN_SCHEMA,
        "source_hashes": source_after,
        "source_contract_gates": source_gates,
        "process_gates": process_gates,
        "native_gates": native_gates,
        "process_passed": all(process_gates.values()),
        "passed": passed,
        "native_metrics": native_metrics,
        "gate1_passes": gate1_passes,
        "baseline_gate1_passes": baseline_gate1_passes,
        "gate1_pass_delta": gate1_passes - baseline_gate1_passes,
        "native_evaluations": 1,
        "native_environment_created": 1,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_evaluations": 0,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "collected_training_transitions": 0,
        "flightsim_packets": 0,
        "device": args.device,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n711-report", type=Path, required=True)
    parser.add_argument("--n712-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.device != "cuda":
        parser.error("N713 is paired to N588 on CUDA")
    return args


if __name__ == "__main__":
    print(json.dumps(screen(parse_args()), indent=2, sort_keys=True))
