from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import scripts.collect_vq2_staged_variable_gate_dagger as staged
import scripts.collect_vq2_vg038_variable_gate_dagger as vg038
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "docs/vq2_vg038_variable_gate_dagger_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg038_vast.sh"


def test_vg038_manifest_binds_admitted_actor_and_terminal_screen() -> None:
    manifest = vg038.load_manifest(MANIFEST)
    assert manifest["seed"] == 429151
    assert manifest["checkpoint_sha256"].startswith("56a8e3b8")
    assert manifest["screen_evidence_sha256"].startswith("bb14e1e1")
    assert manifest["minimum_gate4_rate"] == 0.20
    assert manifest["minimum_phase5_records"] == 1
    required = (
        ROOT / manifest["oracle_report"],
        ROOT / manifest["checkpoint"],
        ROOT / manifest["screen_evidence"],
    )
    if not all(path.is_file() for path in required):
        pytest.skip("retained remote evidence is not mirrored in this checkout")
    staged.verify_bound_inputs(manifest)


def test_vg038_later_phase_predicates_are_hard() -> None:
    manifest = json.loads(MANIFEST.read_text())
    predicate = vg038.add_later_phase_predicates(
        lambda metrics, **kwargs: {"base": True},
        manifest,
    )
    phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
    phase_records[4:6] = 1
    passed = predicate(
        {"env/ordered_gate3_sampled": 0.20},
        phase_records=phase_records,
    )
    assert all(passed.values())

    phase_records[5] = 0
    failed = predicate(
        {"env/ordered_gate3_sampled": 0.19},
        phase_records=phase_records,
    )
    assert not failed["gate_4_reach_rate"]
    assert not failed["minimum_phase_5_records"]


def test_vg038_runner_is_source_locked_bounded_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "vq2_vg038_variable_gate_dagger_round8_vg033_visited_512" in text
    assert text.count("OMP_NUM_THREADS=4 MKL_NUM_THREADS=1") >= 2
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
