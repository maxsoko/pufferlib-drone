#!/usr/bin/env python3
"""Collect one legal DAgger dataset on C009's warmed handoff trajectories."""

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

import scripts.collect_vq2_measured_visual_suffix_dagger as base
from scripts import eval_vq2_measured_visual_suffix_handoff as handoff
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_c011_c009_measured_visual_suffix_dagger_128"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
SEED = 43011
C010_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c010_c009_true_range_exact_128/report.json"
)
C010_REPORT_SHA256 = (
    "b341b49f6053110508553e805b7b1f12c5edf4ac6e9b3635fd262b1b546296c8"
)
C010_TRACE = C010_REPORT.parent / "trace_agent0.npz"
C010_TRACE_SHA256 = (
    "b387175e6a708cb98162ef1d8f72b241748350098443e8ee31ec62233f53b62d"
)


def candidate_warm_prefix(device: torch.device) -> dict[str, Any]:
    """Return the legal prefix with C009's own deterministic warm outputs."""

    prefix = handoff.load_warm_prefix()
    actor, _, _ = handoff.load_candidate_suffix(device, "c009")
    replayed, _ = handoff.warm_suffix_stepwise(
        actor, prefix["visual_observation"], device=device
    )
    result = {name: values.copy() for name, values in prefix.items()}
    result["suffix_action"] = replayed
    return result


def configure_base(device: torch.device) -> None:
    base.TAG = TAG
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.SEED = SEED

    def load_candidate(
        requested_device: torch.device,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        actor, payload, _ = handoff.load_candidate_suffix(
            requested_device, "c009"
        )
        return actor, payload

    base.load_suffix = load_candidate
    base.load_warm_prefix = lambda: candidate_warm_prefix(device)


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict[str, Any]:
    frozen = {
        C010_REPORT: C010_REPORT_SHA256,
        C010_TRACE: C010_TRACE_SHA256,
        handoff.C009_CHECKPOINT: handoff.C009_CHECKPOINT_SHA256,
        handoff.C009_REPORT: handoff.C009_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    prior_screen = json.loads(C010_REPORT.read_text())
    if not prior_screen.get("diagnostic_valid") or prior_screen.get(
        "reached_next_gate"
    ):
        raise RuntimeError("C010 is not the admitted failed C009 trajectory")
    device = torch.device(device_name)
    configure_base(device)
    report = base.collect(output=output, device_name=device_name)

    wrapper = Path(__file__).resolve()
    added_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{
            str(path.relative_to(ROOT)): expected for path, expected in frozen.items()
        },
    }
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tag"] = TAG
    metadata["action"]["plant_source"] = "c009_deterministic_mean"
    metadata["action"]["plant_checkpoint_sha256"] = (
        handoff.C009_CHECKPOINT_SHA256
    )
    metadata["source_sha256"].update(added_sources)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["tag"] = TAG
    report["suffix_candidate"] = "c009"
    report["suffix_checkpoint_sha256"] = handoff.C009_CHECKPOINT_SHA256
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["source_sha256"].update(added_sources)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
