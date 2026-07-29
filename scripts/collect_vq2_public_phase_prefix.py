#!/usr/bin/env python3
"""Collect an admitted bounded prefix after SF043's nonterminal rejection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_public_phase_dagger_next as collector
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf044_public_phase_prefix_dagger_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf044_public_phase_prefix_dagger_preregistration_2026-07-28.md"
)
SEED = 42044
PREFIX_STEPS = 768


def prefix_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
) -> bool:
    lengths = np.asarray(lengths)
    terminal_count = np.asarray(terminal_count)
    terminal_is_last = np.asarray(terminal_is_last)
    return bool(
        metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
        and lengths.shape == (collector.AGENTS,)
        and np.all((lengths > 0) & (lengths <= PREFIX_STEPS))
        and np.all((terminal_count == 0) | (terminal_count == 1))
        and np.all((terminal_count == 0) | terminal_is_last)
        and labels == int(lengths.sum())
        and executed_action_max_error <= 1e-7
    )


def collect(
    output: Path = DEFAULT_OUTPUT,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    collector.TAG = TAG
    collector.PREREGISTRATION = PREREGISTRATION
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = PREFIX_STEPS
    collector.next_dagger_collection_passes = prefix_collection_passes
    report = collector.collect(output=output, device_name=device_name)

    wrapper = Path(__file__).resolve()
    wrapper_key = str(wrapper.relative_to(ROOT))
    wrapper_sha = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][wrapper_key] = wrapper_sha
    metadata["bounded_prefix_dataset"] = True
    metadata["bounded_prefix_steps"] = PREFIX_STEPS
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    terminal = np.load(output / "terminal.npy", mmap_mode="r")
    terminal_per_agent = np.asarray(terminal.sum(axis=0), dtype=np.int64)
    terminated = int((terminal_per_agent == 1).sum())
    report["source_sha256"][wrapper_key] = wrapper_sha
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["bounded_prefix_dataset"] = True
    report["bounded_prefix_steps"] = PREFIX_STEPS
    report["terminated_agents"] = terminated
    report["surviving_agents"] = collector.AGENTS - terminated
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

