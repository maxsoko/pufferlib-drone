#!/usr/bin/env python3
"""Run the pre-registered constrained Jacobian-CMA experiment resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    from constrained_cma import FullCovarianceCMA
    from jacobian_cma_subspace import (
        materialize_candidate,
        sha256_file,
        trace_action_drift,
    )
except ModuleNotFoundError:
    from scripts.constrained_cma import FullCovarianceCMA
    from scripts.jacobian_cma_subspace import (
        materialize_candidate,
        sha256_file,
        trace_action_drift,
    )


EVALUATION_OVERRIDES = (
    "--env.num-gates", "4",
    "--env.course-geometry-scale", "1",
    "--env.course-geometry-scale-randomize", "0",
    "--env.gate-radius", "0.75",
    "--env.gate-radius-randomize", "0",
    "--env.sitl-gate-transition-min-forward-speed-randomize", "0",
    "--env.sitl-gate-transition-from-gate-index", "2",
    "--env.sitl-gate-transition-max-accel-m-s2", "0",
    "--env.sitl-gate-transition-preserve-velocity-direction", "0",
    "--env.sitl-gate-transition-delay-steps", "0",
    "--env.sitl-gate-obs-dropout-from-index", "1",
    "--env.sitl-gate-obs-dropout-range-m", "0",
    "--env.gate-position-domain-randomize", "0",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluation_cache_key(checkpoint: Path, configuration: dict[str, Any]) -> str:
    payload = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "configuration": configuration,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_evaluation_command(
    checkpoint: Path,
    json_path: Path,
    csv_path: Path,
    *,
    floor: float,
    episodes: int,
    episode_offset: int,
    total_agents: int,
    label: str,
    layout_precision_bytes: int = 4,
) -> list[str]:
    if episodes <= 0 or total_agents <= 0 or episodes % total_agents:
        raise ValueError("episodes must be a positive multiple of total agents")
    return [
        sys.executable,
        "scripts/eval_drone_race_checkpoint.py",
        str(checkpoint),
        "--env-name", "drone_race_full_policy_official_fit",
        "--json-path", str(json_path),
        "--csv-path", str(csv_path),
        "--eval-episodes", str(episodes),
        "--episode-offset", str(episode_offset),
        "--require-exact-episodes",
        "--checkpoint-layout-precision-bytes", str(layout_precision_bytes),
        "--max-rollouts", "64",
        "--horizon", "32",
        "--label", label,
        "--vec.total-agents", str(total_agents),
        "--vec.num-buffers", "8",
        "--vec.num-threads", "16",
        *EVALUATION_OVERRIDES,
        "--env.sitl-gate-transition-min-forward-speed", repr(float(floor)),
    ]


def run_evaluation(
    checkpoint: Path,
    cache_dir: Path,
    *,
    floor: float,
    episodes: int,
    episode_offset: int,
    total_agents: int,
    label: str,
    layout_precision_bytes: int = 4,
) -> dict[str, Any]:
    configuration = {
        "floor": float(floor),
        "episodes": int(episodes),
        "episode_offset": int(episode_offset),
        "total_agents": int(total_agents),
        "layout_precision_bytes": int(layout_precision_bytes),
        "overrides": list(EVALUATION_OVERRIDES),
    }
    key = evaluation_cache_key(checkpoint, configuration)
    json_path = cache_dir / f"{key}.json"
    csv_path = cache_dir / f"{key}.csv"
    stdout_path = cache_dir / f"{key}.stdout.txt"
    if json_path.exists():
        report = json.loads(json_path.read_text(encoding="utf-8"))
        report.setdefault("cache", {})["hit"] = True
        report["cache"]["key"] = key
        report["cache"]["json_path"] = str(json_path)
        return report

    cache_dir.mkdir(parents=True, exist_ok=True)
    command = build_evaluation_command(
        checkpoint,
        json_path,
        csv_path,
        floor=floor,
        episodes=episodes,
        episode_offset=episode_offset,
        total_agents=total_agents,
        label=label,
        layout_precision_bytes=layout_precision_bytes,
    )
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not json_path.exists():
        raise RuntimeError(
            f"evaluation failed with exit {completed.returncode}; see {stdout_path}"
        )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["cache"] = {
        "hit": False,
        "key": key,
        "json_path": str(json_path),
        "command": command,
        "wall_seconds": round(time.time() - started, 6),
    }
    return report


def outcome_metrics(report: dict[str, Any] | None) -> dict[str, float]:
    if report is None:
        return {
            "success": 0.0,
            "gates": 0.0,
            "crash": 1.0,
            "gate2_crossing": 0.0,
            "terminal_radial": float("inf"),
            "crossing_radial": float("inf"),
            "final_progress": 0.0,
        }
    metrics = report["metrics"]
    terminal_sampled = float(metrics.get("env/gate2_terminal_crossing_sampled", 0.0))
    crossing_sampled = float(metrics.get("env/gate2_crossing_sampled", 0.0))
    return {
        "success": float(metrics.get("env/success_rate", 0.0)),
        "gates": float(metrics.get("env/gates_passed", 0.0)),
        "crash": float(metrics.get("env/crash", 0.0)),
        "gate2_crossing": crossing_sampled,
        "terminal_radial": (
            float(metrics.get("env/avg_gate2_terminal_crossing_radial", float("inf")))
            if terminal_sampled > 0.0
            else float("inf")
        ),
        "crossing_radial": (
            float(metrics.get("env/avg_gate2_crossing_radial", float("inf")))
            if crossing_sampled > 0.0
            else float("inf")
        ),
        "final_progress": float(metrics.get("env/final_progress", 0.0)),
    }


def candidate_rank_key(record: dict[str, Any]) -> tuple:
    outcome = outcome_metrics(record.get("development_target"))
    feasible = bool(record.get("trace_feasible", False)) and record.get(
        "anchor_actual_feasible", True
    ) is not False
    radial = outcome["crossing_radial"] if outcome["gate2_crossing"] > 0 else outcome["terminal_radial"]
    if not np.isfinite(radial):
        radial = 1e9
    anchor_drift = record.get("anchor_drift", {})
    drift = max(
        float(anchor_drift.get("full_rms", 1e9)),
        float(anchor_drift.get("focus_rms", 1e9)),
    )
    return (
        int(feasible),
        int(record.get("development_target") is not None),
        outcome["gate2_crossing"],
        outcome["success"],
        outcome["gates"],
        -radial,
        outcome["final_progress"],
        -drift,
    )


def _anchor_feasible(report: dict[str, Any], minimum_success: float, maximum_crash: float) -> bool:
    outcome = outcome_metrics(report)
    return outcome["success"] >= minimum_success and outcome["crash"] <= maximum_crash


def _record_matches(record: dict[str, Any], coefficients: np.ndarray) -> bool:
    stored = np.asarray(record.get("checkpoint", {}).get("coefficients", []), dtype=np.float32)
    expected = np.asarray(coefficients, dtype=np.float32)
    return stored.shape == expected.shape and np.array_equal(stored, expected)


def has_enough_feasible_parents(
    records: list[dict[str, Any]], parent_count: int
) -> bool:
    return sum(bool(record.get("trace_feasible", False)) for record in records) >= parent_count


def experiment_stop_reason(
    *,
    constraint_stopped: bool,
    promoted: dict[str, Any] | None,
    generation: int,
    maximum_generations: int,
) -> str | None:
    if constraint_stopped:
        return "insufficient_feasible_parents_under_anchor_trust_region"
    if promoted is None and generation >= maximum_generations:
        return "maximum_generation_budget_exhausted_without_promotion"
    return None


def load_generation_summaries(output_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(output_dir.glob("generation_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
    return summaries


def run_experiment(
    manifest_path: Path,
    basis_path: Path,
    failure_trace: Path,
    anchor_trace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cma_cfg = manifest["cma_es"]
    calibration_cfg = manifest["initial_step_calibration"]
    parent = Path(manifest["parent_checkpoint"])
    if sha256_file(parent) != manifest["parent_sha256"]:
        raise ValueError("parent checkpoint hash does not match manifest")
    basis_metadata = json.loads(str(np.load(basis_path, allow_pickle=False)["metadata"]))
    if basis_metadata["parent_sha256"] != manifest["parent_sha256"]:
        raise ValueError("basis parent does not match manifest")

    cache_dir = output_dir / "evaluation_cache"
    state_path = output_dir / "cma_state.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.json"
    generation_summaries = load_generation_summaries(output_dir)
    if final_path.exists():
        prior_final = json.loads(final_path.read_text(encoding="utf-8"))
        if prior_final.get("stop_reason") or prior_final.get("promoted"):
            if prior_final.get("generation_summaries") != generation_summaries:
                prior_final["generation_summaries"] = generation_summaries
                write_json(final_path, prior_final)
            return prior_final
    if state_path.exists():
        cma = FullCovarianceCMA.load(state_path)
    else:
        cma = FullCovarianceCMA(
            int(cma_cfg["dimension"]),
            population_size=int(cma_cfg["population_size"]),
            parent_count=int(cma_cfg["parent_count"]),
            sigma=1.0,
            seed=int(cma_cfg["seed"]),
        )

    development_episodes = int(cma_cfg["development_episodes"])
    validation_episodes = int(cma_cfg["validation_episodes"])
    development_agents = development_episodes
    exact_episodes = int(cma_cfg["authoritative_evaluation_episodes"])
    anchor_floor = float(manifest["curriculum"]["anchor_floor"])
    target_floor = float(manifest["curriculum"]["target_floor"])
    layout_precision = int(manifest["checkpoint_layout_precision_bytes"])

    baseline_dev = run_evaluation(
        parent,
        cache_dir,
        floor=target_floor,
        episodes=development_episodes,
        episode_offset=int(cma_cfg["development_episode_offset"]),
        total_agents=development_agents,
        label="jacobian_cma_parent_target_development",
        layout_precision_bytes=layout_precision,
    )
    baseline_validation = run_evaluation(
        parent,
        cache_dir,
        floor=target_floor,
        episodes=validation_episodes,
        episode_offset=int(cma_cfg["validation_episode_offset"]),
        total_agents=validation_episodes,
        label="jacobian_cma_parent_target_validation",
        layout_precision_bytes=layout_precision,
    )
    baseline_anchor = run_evaluation(
        parent,
        cache_dir,
        floor=anchor_floor,
        episodes=development_episodes,
        episode_offset=int(cma_cfg["development_episode_offset"]),
        total_agents=development_agents,
        label="jacobian_cma_parent_anchor_development",
        layout_precision_bytes=layout_precision,
    )
    baselines = {
        "target_development": baseline_dev,
        "target_validation": baseline_validation,
        "anchor_development": baseline_anchor,
    }
    write_json(output_dir / "baselines.json", baselines)
    validation_baseline_metrics = outcome_metrics(baseline_validation)

    promoted: dict[str, Any] | None = None
    maximum_generations = int(cma_cfg["maximum_generations"])
    while cma.generation < maximum_generations and promoted is None:
        generation = cma.generation
        generation_dir = output_dir / f"generation_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_before = cma.diagnostics()
        coefficients = cma.ask()
        np.save(generation_dir / "coefficients.npy", coefficients)
        records = []
        for index, coefficient in enumerate(coefficients):
            record_path = generation_dir / f"candidate_{index:02d}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if _record_matches(record, coefficient):
                    records.append(record)
                    continue
            candidate_path = generation_dir / f"candidate_{index:02d}.bin"
            checkpoint_record = materialize_candidate(
                parent, basis_path, coefficient, candidate_path
            )
            serialized = np.fromfile(candidate_path, dtype=np.float32)
            finite_checkpoint = bool(np.isfinite(serialized).all())
            failure_drift = trace_action_drift(
                parent, candidate_path, failure_trace, agents=2
            )
            anchor_drift = trace_action_drift(
                parent, candidate_path, anchor_trace, agents=2
            )
            anchor_rms = max(anchor_drift["full_rms"], anchor_drift["focus_rms"])
            anchor_max = max(anchor_drift["full_max"], anchor_drift["focus_max"])
            trace_feasible = (
                finite_checkpoint
                and failure_drift["finite"]
                and anchor_drift["finite"]
                and anchor_rms <= float(calibration_cfg["maximum_anchor_action_rms"])
                and anchor_max <= float(calibration_cfg["maximum_anchor_action_max"])
            )
            record = {
                "generation": generation,
                "index": index,
                "checkpoint": checkpoint_record,
                "failure_drift": failure_drift,
                "anchor_drift": anchor_drift,
                "trace_feasible": trace_feasible,
                "anchor_actual_feasible": None,
                "development_target": None,
            }
            if trace_feasible:
                record["development_target"] = run_evaluation(
                    candidate_path,
                    cache_dir,
                    floor=target_floor,
                    episodes=development_episodes,
                    episode_offset=int(cma_cfg["development_episode_offset"]),
                    total_agents=development_agents,
                    label=f"jacobian_cma_g{generation:03d}_c{index:02d}_target_dev",
                    layout_precision_bytes=layout_precision,
                )
            write_json(record_path, record)
            records.append(record)

        preliminary = sorted(
            range(len(records)), key=lambda idx: candidate_rank_key(records[idx]), reverse=True
        )
        actual_anchor_count = int(cma_cfg["top_anchor_evaluations_per_generation"])
        for index in preliminary[:actual_anchor_count]:
            record = records[index]
            if not record["trace_feasible"]:
                continue
            candidate_path = Path(record["checkpoint"]["path"])
            anchor_report = run_evaluation(
                candidate_path,
                cache_dir,
                floor=anchor_floor,
                episodes=development_episodes,
                episode_offset=int(cma_cfg["development_episode_offset"]),
                total_agents=development_agents,
                label=f"jacobian_cma_g{generation:03d}_c{index:02d}_anchor_dev",
                layout_precision_bytes=layout_precision,
            )
            record["development_anchor"] = anchor_report
            record["anchor_actual_feasible"] = _anchor_feasible(
                anchor_report,
                float(cma_cfg["minimum_anchor_success_rate"]),
                float(cma_cfg["maximum_crash_rate"]),
            )
            write_json(generation_dir / f"candidate_{index:02d}.json", record)

        ranking = sorted(
            range(len(records)), key=lambda idx: candidate_rank_key(records[idx]), reverse=True
        )
        winner = records[ranking[0]]
        winner_index = int(winner["index"])
        winner_validation = None
        exact_reports = None
        if candidate_rank_key(winner)[0] == 1:
            winner_path = Path(winner["checkpoint"]["path"])
            winner_validation = run_evaluation(
                winner_path,
                cache_dir,
                floor=target_floor,
                episodes=validation_episodes,
                episode_offset=int(cma_cfg["validation_episode_offset"]),
                total_agents=validation_episodes,
                label=f"jacobian_cma_g{generation:03d}_winner_target_validation",
                layout_precision_bytes=layout_precision,
            )
            validation_metrics = outcome_metrics(winner_validation)
            discrete_improvement = (
                validation_metrics["gate2_crossing"]
                > validation_baseline_metrics["gate2_crossing"]
            )
            radial_improvement = (
                validation_baseline_metrics["terminal_radial"]
                - validation_metrics["terminal_radial"]
            )
            exact_eligible = discrete_improvement or radial_improvement >= float(
                cma_cfg["minimum_radial_improvement_m"]
            )
            if exact_eligible:
                exact_target = run_evaluation(
                    winner_path,
                    cache_dir,
                    floor=target_floor,
                    episodes=exact_episodes,
                    episode_offset=0,
                    total_agents=1024,
                    label=f"jacobian_cma_g{generation:03d}_winner_target_exact1024",
                    layout_precision_bytes=layout_precision,
                )
                exact_anchor = run_evaluation(
                    winner_path,
                    cache_dir,
                    floor=anchor_floor,
                    episodes=exact_episodes,
                    episode_offset=0,
                    total_agents=1024,
                    label=f"jacobian_cma_g{generation:03d}_winner_anchor_exact1024",
                    layout_precision_bytes=layout_precision,
                )
                exact_reports = {"target": exact_target, "anchor": exact_anchor}
                target_metrics = outcome_metrics(exact_target)
                anchor_metrics = outcome_metrics(exact_anchor)
                if (
                    target_metrics["success"] >= float(cma_cfg["target_success_rate"])
                    and target_metrics["crash"] <= float(cma_cfg["maximum_crash_rate"])
                    and anchor_metrics["success"] >= float(cma_cfg["minimum_anchor_success_rate"])
                ):
                    promoted = {
                        "generation": generation,
                        "candidate_index": winner_index,
                        "checkpoint": winner["checkpoint"],
                        "validation": winner_validation,
                        "exact": exact_reports,
                    }

        constraint_stop = not has_enough_feasible_parents(
            records, int(cma_cfg["parent_count"])
        )
        if not constraint_stop:
            cma.tell(ranking)
            cma.save(state_path)
        summary = {
            "generation": generation,
            "basis_sha256": basis_metadata["basis_sha256"],
            "coefficient_scale": basis_metadata["coefficient_scale"],
            "diagnostics_before": diagnostics_before,
            "diagnostics_after": cma.diagnostics(),
            "ranking": ranking,
            "winner_index": winner_index,
            "winner_rank_key": candidate_rank_key(winner),
            "winner_validation": winner_validation,
            "winner_exact": exact_reports,
            "promoted": promoted,
            "feasible_candidates": sum(r["trace_feasible"] for r in records),
            "actual_anchor_passes": sum(
                r.get("anchor_actual_feasible") is True for r in records
            ),
            "constraint_stop": constraint_stop,
        }
        audited_generation = generation + 1
        if audited_generation == int(manifest["stop_rules"]["early_audit_generation"]):
            singular = np.asarray(basis_metadata["singular_values"], dtype=np.float64)
            captured = float(np.square(singular[: cma.dimension]).sum() / np.square(singular).sum())
            summary["early_basis_audit"] = {
                "captured_sensitivity_energy": captured,
                "discarded_sensitivity_energy": 1.0 - captured,
                "expand_to_24": (1.0 - captured) > 0.25,
                "expansion_threshold": 0.25,
            }
        write_json(generation_dir / "summary.json", summary)
        generation_summaries.append(summary)
        if constraint_stop:
            break

    constraint_stopped = bool(
        generation_summaries and generation_summaries[-1].get("constraint_stop")
    )
    stop_reason = experiment_stop_reason(
        constraint_stopped=constraint_stopped,
        promoted=promoted,
        generation=cma.generation,
        maximum_generations=maximum_generations,
    )
    final = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "basis": str(basis_path),
        "basis_sha256": basis_metadata["basis_sha256"],
        "parent_sha256": manifest["parent_sha256"],
        "cma": cma.diagnostics(),
        "promoted": promoted,
        "stop_reason": stop_reason,
        "generation_summaries": generation_summaries,
    }
    write_json(final_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("basis", type=Path)
    parser.add_argument("failure_trace", type=Path)
    parser.add_argument("anchor_trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        args.manifest,
        args.basis,
        args.failure_trace,
        args.anchor_trace,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
