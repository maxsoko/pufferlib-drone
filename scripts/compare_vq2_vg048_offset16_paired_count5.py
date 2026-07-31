#!/usr/bin/env python3
"""Confirm VG047 against VG033 on a distinct offset-16 fixture."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.compare_vq2_vg046_offset8_paired_count5 as base


SCHEMA = "vq2_vg048_offset16_paired_count5_confirmation_v1"
TAG = "vq2_vg048_vg047_offset16_paired_count5_confirmation_001"
EPISODE_OFFSET = 16
PRIOR_PARENT_BEHAVIOR_SHA256 = (
    "faa021347a723c9353249cc72b145cddb27ffef7d9a443fbe5ae12606d4cc69c"
)
PRIOR_CANDIDATE_BEHAVIOR_SHA256 = (
    "b742e6029976d0fdeac091bcc87b2270ee8d00f40eb3b1ac32f3fcf11c3c07f2"
)
PARENT_MANIFEST = (
    _ROOT / "docs/vq2_vg048_parent_count5_manifest_2026-07-31.json"
)
CANDIDATE_MANIFEST = (
    _ROOT / "docs/vq2_vg048_candidate_count5_manifest_2026-07-31.json"
)
PRIOR_EVIDENCE = (
    _ROOT
    / "docs/vq2_vg047_offset8_fraction_bracket_admission_2026-07-31.json"
)


def configure_contract() -> None:
    base.SCHEMA = SCHEMA
    base.TAG = TAG
    base.EPISODE_OFFSET = EPISODE_OFFSET
    base.VG044_PARENT_BEHAVIOR_SHA256 = PRIOR_PARENT_BEHAVIOR_SHA256
    base.VG044_CANDIDATE_BEHAVIOR_SHA256 = (
        PRIOR_CANDIDATE_BEHAVIOR_SHA256
    )
    base.PARENT_MANIFEST = PARENT_MANIFEST
    base.CANDIDATE_MANIFEST = CANDIDATE_MANIFEST
    base.PRIOR_EVIDENCE = PRIOR_EVIDENCE


def main() -> int:
    configure_contract()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
