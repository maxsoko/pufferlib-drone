#!/usr/bin/env python3
"""Run the LC004 scale after repairing only its executable import path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_vq2_lc003_native_puffer_training as lc003
import scripts.benchmark_vq2_lc004_scaleout_puffer_training as lc004


TAG = "vq2_lc005_scaleout_puffer_import_repair_001"
SEED = 431050
PREREGISTRATION = (
    ROOT
    / "docs/vq2_lc005_scaleout_puffer_import_repair_preregistration_2026-07-31.md"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG / "report.json"
)


def configure_lc005() -> None:
    lc004.configure_lc004()
    lc003.TAG = TAG
    base_config = lc003.benchmark_config
    base_identity = lc003.source_identity

    def benchmark_config(pufferl_module):
        config = base_config(pufferl_module)
        config["seed"] = SEED
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

    lc003.benchmark_config = benchmark_config
    lc003.source_identity = source_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=lc003.DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_lc005()
    report = lc003.run_benchmark(baseline_path=args.baseline.resolve())
    report["schema"] = "vq2_lc005_scaleout_puffer_import_repair_report_v1"
    report["tag"] = TAG
    report["configuration"]["seed"] = SEED
    lc003.write_json_once(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
