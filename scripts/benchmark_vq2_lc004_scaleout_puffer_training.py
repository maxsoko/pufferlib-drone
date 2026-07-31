#!/usr/bin/env python3
"""Scale LC003 across the retained 255-core Vast host and RTX 4090."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.benchmark_vq2_lc003_native_puffer_training as lc003


TAG = "vq2_lc004_scaleout_puffer_throughput_001"
TOTAL_AGENTS = 4096
NUM_BUFFERS = 64
NUM_THREADS = 256
HORIZON = 16
MINIBATCH_SIZE = TOTAL_AGENTS * HORIZON
SEED = 431040
PREREGISTRATION = (
    lc003.ROOT
    / "docs/vq2_lc004_scaleout_puffer_throughput_preregistration_2026-07-31.md"
)
DEFAULT_OUTPUT = (
    lc003.ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG / "report.json"
)


def configure_lc004() -> None:
    """Install the source-locked LC004 scale into the frozen LC003 harness."""

    lc003.TAG = TAG
    lc003.TOTAL_AGENTS = TOTAL_AGENTS
    lc003.NUM_BUFFERS = NUM_BUFFERS
    lc003.NUM_THREADS = NUM_THREADS
    lc003.HORIZON = HORIZON
    lc003.MINIBATCH_SIZE = MINIBATCH_SIZE

    base_config = lc003.benchmark_config
    base_identity = lc003.source_identity

    def benchmark_config(pufferl_module):
        config = base_config(pufferl_module)
        config["seed"] = SEED
        return config

    def source_identity():
        identity = base_identity()
        identity["source_sha256"].update({
            str(PREREGISTRATION.relative_to(lc003.ROOT)): lc003.sha256_path(
                PREREGISTRATION
            ),
            str(Path(__file__).resolve().relative_to(lc003.ROOT)): lc003.sha256_path(
                Path(__file__).resolve()
            ),
        })
        return identity

    lc003.benchmark_config = benchmark_config
    lc003.source_identity = source_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=lc003.DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_lc004()
    report = lc003.run_benchmark(baseline_path=args.baseline.resolve())
    report["schema"] = "vq2_lc004_scaleout_puffer_throughput_report_v1"
    report["tag"] = TAG
    report["configuration"]["seed"] = SEED
    lc003.write_json_once(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
