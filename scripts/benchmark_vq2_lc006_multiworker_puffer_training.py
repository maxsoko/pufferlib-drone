#!/usr/bin/env python3
"""Measure six concurrent LC003 PuffeRL workers on one Vast instance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_vq2_lc003_native_puffer_training as lc003


TAG = "vq2_lc006_multiworker_puffer_throughput_001"
WORKERS = 6
BASE_SEED = 431060
PREREGISTRATION = (
    ROOT / "docs/vq2_lc006_multiworker_puffer_throughput_preregistration_2026-07-31.md"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def configure_worker(worker_index: int) -> None:
    if worker_index < 0 or worker_index >= WORKERS:
        raise ValueError("worker index out of range")
    lc003.TAG = f"{TAG}_worker{worker_index}"
    base_config = lc003.benchmark_config
    base_identity = lc003.source_identity

    def benchmark_config(pufferl_module):
        config = base_config(pufferl_module)
        config["seed"] = BASE_SEED + worker_index
        return config

    def source_identity():
        identity = base_identity()
        identity["source_sha256"].update({
            str(PREREGISTRATION.relative_to(ROOT)): lc003.sha256_path(
                PREREGISTRATION
            ),
            str(Path(__file__).resolve().relative_to(ROOT)): lc003.sha256_path(
                Path(__file__).resolve()
            ),
        })
        return identity

    if worker_index != 0:
        def idle_gpu_sampler(stop, samples):
            del samples
            stop.wait()

        lc003._poll_gpu = idle_gpu_sampler
    lc003.benchmark_config = benchmark_config
    lc003.source_identity = source_identity


def worker_integrity(report: dict[str, Any]) -> dict[str, bool]:
    ignored = {"speedup_at_least_10x", "gpu_reached_50_percent"}
    return {
        key: value for key, value in report["admission_predicates"].items()
        if key not in ignored
    }


def run_worker(*, worker_index: int, baseline: Path, output: Path) -> int:
    configure_worker(worker_index)
    report = lc003.run_benchmark(baseline_path=baseline)
    report["schema"] = "vq2_lc006_multiworker_puffer_worker_v1"
    report["worker_index"] = worker_index
    report["configuration"]["seed"] = BASE_SEED + worker_index
    integrity = worker_integrity(report)
    report["worker_integrity_predicates"] = integrity
    report["worker_integrity_admitted"] = all(integrity.values())
    # Individual speed is diagnostic; only the concurrent aggregate can admit.
    report["admitted"] = False
    lc003.write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["worker_integrity_admitted"] else 1


def aggregate_worker_reports(
    reports: list[dict[str, Any]], report_paths: list[Path]
) -> dict[str, Any]:
    if len(reports) != WORKERS or len(report_paths) != WORKERS:
        raise ValueError("LC006 requires exactly six worker reports")
    starts = [float(report["measured_started_unix"]) for report in reports]
    finishes = [float(report["measured_finished_unix"]) for report in reports]
    union_seconds = max(finishes) - min(starts)
    overlap_seconds = max(0.0, min(finishes) - max(starts))
    total_steps = sum(int(report["measured_agent_steps"]) for report in reports)
    aggregate_sps = total_steps / union_seconds
    baseline_sps = float(reports[0]["baseline_agent_steps_per_second"])
    identity_exact = all(
        report["source_commit"] == reports[0]["source_commit"]
        and report["source_sha256"] == reports[0]["source_sha256"]
        for report in reports[1:]
    )
    device_max_gpu = max(
        report["gpu_sampling"]["maximum_percent"] for report in reports
    )
    predicates = {
        "six_worker_reports": (
            [report["worker_index"] for report in reports]
            == list(range(WORKERS))
        ),
        "all_worker_integrity": all(
            report["worker_integrity_admitted"] for report in reports
        ),
        "source_identity_exact": identity_exact,
        "measurement_union_positive": union_seconds > 0.0,
        "measurement_overlap_at_least_half": (
            overlap_seconds / union_seconds >= 0.50
        ),
        "aggregate_speedup_at_least_10x": (
            aggregate_sps / baseline_sps >= lc003.MINIMUM_SPEEDUP
        ),
        "device_gpu_reached_50_percent": (
            device_max_gpu >= lc003.MINIMUM_MAX_GPU_PERCENT
        ),
        "zero_checkpoints": all(
            report["checkpoints_written"] == 0 for report in reports
        ),
        "no_flightsim_packets": all(
            report["flight_sim_packets_sent"] == 0 for report in reports
        ),
        "submission_forbidden": all(
            report["submission_authorized"] is False for report in reports
        ),
    }
    return {
        "schema": "vq2_lc006_multiworker_puffer_throughput_report_v1",
        "tag": TAG,
        "workers": WORKERS,
        "worker_report_sha256": {
            str(index): lc003.sha256_path(path)
            for index, path in enumerate(report_paths)
        },
        "worker_end_to_end_sps": [
            report["end_to_end_agent_steps_per_second"] for report in reports
        ],
        "total_measured_agent_steps": total_steps,
        "measurement_union_seconds": union_seconds,
        "measurement_overlap_seconds": overlap_seconds,
        "aggregate_agent_steps_per_second": aggregate_sps,
        "baseline_agent_steps_per_second": baseline_sps,
        "aggregate_speedup": aggregate_sps / baseline_sps,
        "device_maximum_gpu_percent": device_max_gpu,
        "device_maximum_memory_mib": max(
            report["gpu_sampling"]["maximum_memory_mib"] for report in reports
        ),
        "source_commit": reports[0]["source_commit"],
        "source_sha256": reports[0]["source_sha256"],
        "admission_predicates": predicates,
        "admitted": all(predicates.values()),
        "deployment_legal": False,
        "checkpoints_written": 0,
        "flight_sim_packets_sent": 0,
        "submission_authorized": False,
    }


def run_orchestrator(*, baseline: Path, output_root: Path) -> int:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []
    streams = []
    report_paths = [output_root / f"worker_{index}.json" for index in range(WORKERS)]
    try:
        for index, report_path in enumerate(report_paths):
            stream = (output_root / f"worker_{index}.stdout").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-index", str(index),
                    "--baseline", str(baseline),
                    "--output", str(report_path),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "OMP_DYNAMIC": "FALSE"},
            ))
        exit_codes = [process.wait() for process in processes]
    finally:
        for stream in streams:
            stream.close()
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"LC006 worker exits changed: {exit_codes}")
    reports = [json.loads(path.read_text()) for path in report_paths]
    aggregate = aggregate_worker_reports(reports, report_paths)
    lc003.write_json_once(output_root / "report.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if aggregate["admitted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--baseline", type=Path, default=lc003.DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.worker_index is not None:
        if args.output is None:
            parser.error("worker mode requires --output")
        return run_worker(
            worker_index=args.worker_index,
            baseline=args.baseline.resolve(),
            output=args.output.resolve(),
        )
    return run_orchestrator(
        baseline=args.baseline.resolve(), output_root=args.output_root.resolve()
    )


if __name__ == "__main__":
    raise SystemExit(main())
