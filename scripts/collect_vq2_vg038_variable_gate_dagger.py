#!/usr/bin/env python3
"""Run the VG033-visited Gate-3--5 DAgger collection for VG038."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_staged_variable_gate_dagger as staged
import scripts.collect_vq2_variable_gate_dagger as collector
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = staged.load_manifest(path)
    gate4_rate = float(manifest.get("minimum_gate4_rate", -1.0))
    if not (
        float(manifest["minimum_gate3_rate"]) >= gate4_rate > 0.0
    ):
        raise RuntimeError("VG038 Gate-4 reach rate is inconsistent")
    for name in ("minimum_phase4_records", "minimum_phase5_records"):
        if int(manifest.get(name, 0)) <= 0:
            raise RuntimeError(f"VG038 {name} must be positive")
    return manifest


def add_later_phase_predicates(
    base: Callable[..., dict[str, bool]],
    manifest: dict[str, Any],
) -> Callable[..., dict[str, bool]]:
    minimum_gate4_rate = float(manifest["minimum_gate4_rate"])
    minimum_phase4_records = int(manifest["minimum_phase4_records"])
    minimum_phase5_records = int(manifest["minimum_phase5_records"])

    def later_phase_predicates(
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, bool]:
        predicates = base(metrics, **kwargs)
        phase_records = kwargs["phase_records"]
        phase_shape = (
            isinstance(phase_records, np.ndarray)
            and phase_records.shape == (ENGINE_GATE_CAP + 1,)
        )
        predicates["gate_4_reach_rate"] = bool(
            metrics.get("env/ordered_gate3_sampled", 0.0)
            >= minimum_gate4_rate
        )
        predicates["minimum_phase_4_records"] = bool(
            phase_shape and phase_records[4] >= minimum_phase4_records
        )
        predicates["minimum_phase_5_records"] = bool(
            phase_shape and phase_records[5] >= minimum_phase5_records
        )
        return predicates

    return later_phase_predicates


def configure_collector(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    staged.configure_collector(manifest_path, manifest)
    collector.dagger_collection_predicates = add_later_phase_predicates(
        collector.dagger_collection_predicates,
        manifest,
    )
    collector.EXTRA_SOURCE_PATHS = (
        *collector.EXTRA_SOURCE_PATHS,
        Path(__file__).resolve(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    staged.verify_bound_inputs(manifest)
    configure_collector(manifest_path, manifest)
    report = collector.collect(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
