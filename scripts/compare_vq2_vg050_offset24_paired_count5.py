#!/usr/bin/env python3
"""Confirm VG049 against VG033 on a distinct offset-24 fixture."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.compare_vq2_vg046_offset8_paired_count5 as base


def configure_contract() -> None:
    base.SCHEMA = "vq2_vg050_offset24_paired_count5_confirmation_v1"
    base.TAG = "vq2_vg050_vg049_offset24_paired_count5_confirmation_001"
    base.EPISODE_OFFSET = 24
    base.VG044_PARENT_BEHAVIOR_SHA256 = (
        "eff69b3fc3af2458e57f64e2b3570ae565087a5f5cae3cfc771b916b584efff2"
    )
    base.VG044_CANDIDATE_BEHAVIOR_SHA256 = (
        "7f99e6bc1e38633f1b4a0dcdb3fd54fcf72e1a60b247984c6bbdc49aa2c4d643"
    )
    base.PARENT_MANIFEST = (
        _ROOT / "docs/vq2_vg050_parent_count5_manifest_2026-07-31.json"
    )
    base.CANDIDATE_MANIFEST = (
        _ROOT / "docs/vq2_vg050_candidate_count5_manifest_2026-07-31.json"
    )
    base.PRIOR_EVIDENCE = (
        _ROOT
        / "docs/vq2_vg049_offset16_micro_fraction_bracket_admission_2026-07-31.json"
    )


def main() -> int:
    configure_contract()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
