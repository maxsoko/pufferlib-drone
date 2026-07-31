#!/usr/bin/env python3
"""Screen micro update fractions on the offset-16 fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.eval_vq2_vg047_offset8_fraction_bracket as template


base = template.base
TAG = "vq2_vg049_vg033_to_vg042_offset16_micro_fraction_bracket_001"
SCHEMA = "vq2_checkpoint_line_offset16_micro_bracket_report_v1"
STATE_SCHEMA = "vq2_checkpoint_line_offset16_micro_bracket_state_v1"
SEED = 429163
EPISODE_OFFSET = 16
ALPHAS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025)
REJECTION = (
    _ROOT
    / "docs/vq2_vg048_offset16_paired_count5_rejection_2026-07-31.json"
)
REJECTION_SHA256 = (
    "a8f5b91f24d8713348e860149520300438175dfe5b7bb1d4f0877cb5e60bf440"
)
PREREGISTRATION = (
    _ROOT
    / "docs/vq2_vg049_offset16_micro_fraction_bracket_preregistration_2026-07-31.md"
)
RUNNER = _ROOT / "scripts/run_vq2_vg049_vast.sh"
DEFAULT_OUTPUT = (
    _ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
_TEMPLATE_SOURCE_IDENTITY = template.source_identity


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        parent["hard_transport_pass"]
        and candidate["hard_transport_pass"]
        and int(candidate["gate_reach"]["1"])
        >= int(parent["gate_reach"]["1"])
        and int(candidate["gate_reach"]["2"])
        >= int(parent["gate_reach"]["2"])
        and candidate["crashes"] <= parent["crashes"]
        and (
            candidate["successes"] > parent["successes"]
            or int(candidate["gate_reach"]["4"])
            > int(parent["gate_reach"]["4"])
            or int(candidate["gate_reach"]["3"])
            > int(parent["gate_reach"]["3"])
        )
    )


def verify_inputs() -> None:
    expected = {
        base.PARENT_CHECKPOINT: base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT: base.PARENT_REPORT_SHA256,
        base.PARENT_ADMISSION: base.PARENT_ADMISSION_SHA256,
        base.UPDATE_CHECKPOINT: base.UPDATE_CHECKPOINT_SHA256,
        base.UPDATE_REPORT: base.UPDATE_REPORT_SHA256,
        base.UPDATE_ADMISSION: base.UPDATE_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        base.GOAL_PROMPT: base.GOAL_PROMPT_SHA256,
    }
    for path, digest in expected.items():
        if base.sha256_path(path) != digest:
            raise RuntimeError(f"VG049 input hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG049 preregistration is missing")
    rejection = json.loads(REJECTION.read_text())
    if (
        rejection.get("schema")
        != "vq2_vg048_offset16_paired_count5_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("qualified_for_count11_diagnostic")
        or not rejection.get("criteria", {}).get(
            "candidate_first_gate4_reach"
        )
        or rejection.get("criteria", {}).get("crash_count_not_regressed")
    ):
        raise RuntimeError("VG048 rejection does not authorize VG049")
    base.load_endpoints()


def source_identity() -> dict[str, Any]:
    identity = _TEMPLATE_SOURCE_IDENTITY()
    sources = dict(identity["source_sha256"])
    additions = (Path(__file__).resolve(), REJECTION)
    sources.update({
        str(path.relative_to(_ROOT)): base.sha256_path(path)
        for path in additions
    })
    identity["source_sha256"] = sources
    identity["episode_offset"] = EPISODE_OFFSET
    return identity


def configure_contract() -> None:
    assignments = {
        "TAG": TAG,
        "SCHEMA": SCHEMA,
        "STATE_SCHEMA": STATE_SCHEMA,
        "SEED": SEED,
        "EPISODE_OFFSET": EPISODE_OFFSET,
        "ALPHAS": ALPHAS,
        "REJECTION": REJECTION,
        "REJECTION_SHA256": REJECTION_SHA256,
        "PREREGISTRATION": PREREGISTRATION,
        "RUNNER": RUNNER,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "verify_inputs": verify_inputs,
        "source_identity": source_identity,
    }
    for name, value in assignments.items():
        setattr(template, name, value)
    base.qualifies = qualifies


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    configure_contract()
    return template.run(
        output=output,
        device_name=device_name,
        resume=resume,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
