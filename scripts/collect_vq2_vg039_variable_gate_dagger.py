#!/usr/bin/env python3
"""Run the fresh evidence-calibrated VG033-visited collection for VG039."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_staged_variable_gate_dagger as staged
import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.collect_vq2_vg038_variable_gate_dagger as vg038


def root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = vg038.load_manifest(path)
    for name in (
        "previous_collection_rejection",
        "previous_collection_rejection_sha256",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise RuntimeError(f"VG039 {name} is missing")
    if len(manifest["previous_collection_rejection_sha256"]) != 64:
        raise RuntimeError("VG039 rejection digest is not a SHA-256")
    root_path(manifest["previous_collection_rejection"])
    return manifest


def verify_bound_inputs(manifest: dict[str, Any]) -> None:
    staged.verify_bound_inputs(manifest)
    rejection_path = root_path(manifest["previous_collection_rejection"])
    if (
        collector.sha256_path(rejection_path)
        != manifest["previous_collection_rejection_sha256"]
    ):
        raise RuntimeError("VG039 does not bind the VG038 rejection")
    rejection = json.loads(rejection_path.read_text())
    predicates = rejection.get("admission_predicates", {})
    failed = rejection.get("failed_admission_predicates")
    if (
        rejection.get("schema")
        != "vq2_vg038_variable_gate_dagger_collection_rejection_v1"
        or rejection.get("admitted") is not False
        or failed != ["gate_3_reach_rate", "gate_4_reach_rate"]
        or any(
            not passed
            for name, passed in predicates.items()
            if name not in failed
        )
        or rejection.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or rejection.get("records", 0) < 1_000_000
        or any(value <= 0 for value in rejection.get("phase_records", [])[2:6])
        or rejection.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejection.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG038 is not the source-locked floor-only rejection")


def configure_collector(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    vg038.configure_collector(manifest_path, manifest)
    rejection_path = root_path(manifest["previous_collection_rejection"])
    collector.EVIDENCE_PATHS = (*collector.EVIDENCE_PATHS, rejection_path)
    collector.EXTRA_SOURCE_PATHS = (
        *collector.EXTRA_SOURCE_PATHS,
        Path(__file__).resolve(),
    )
    collector.EVIDENCE_SHA256 = {
        **collector.EVIDENCE_SHA256,
        "previous_collection_rejection_sha256": manifest[
            "previous_collection_rejection_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    verify_bound_inputs(manifest)
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
