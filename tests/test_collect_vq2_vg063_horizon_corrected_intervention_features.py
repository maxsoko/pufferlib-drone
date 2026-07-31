from __future__ import annotations

import scripts.collect_vq2_vg062_warmed_teacher_intervention_features as core
import scripts.collect_vq2_vg063_horizon_corrected_intervention_features as vg063


def test_vg063_only_changes_horizon_identity_and_corpus_floors() -> None:
    assert vg063.SEED > core.SEED
    assert vg063.STEP_LIMIT == 360 * 64
    assert vg063.MINIMUM_SUCCESS_RATE == 0.99
    assert vg063.MINIMUM_RECORDS == 1_000_000
    assert core.INTERVENTION_PHASE_MIN == 1


def test_vg063_configures_the_shared_collector() -> None:
    old = (
        core.TAG, core.SCHEMA, core.STATE_SCHEMA, core.SEED, core.STEP_LIMIT,
        core.MINIMUM_SUCCESS_RATE, core.MINIMUM_RECORDS, core.PREREGISTRATION,
        core.RUNNER, core.DEFAULT_OUTPUT, core.EXTRA_SOURCE_PATHS, core.verify_inputs,
    )
    try:
        vg063.configure()
        assert core.TAG == vg063.TAG
        assert core.SCHEMA == vg063.REPORT_SCHEMA
        assert core.STATE_SCHEMA == vg063.STATE_SCHEMA
        assert core.SEED == vg063.SEED
        assert core.STEP_LIMIT == vg063.STEP_LIMIT
        assert core.MINIMUM_SUCCESS_RATE == vg063.MINIMUM_SUCCESS_RATE
        assert core.MINIMUM_RECORDS == vg063.MINIMUM_RECORDS
        assert core.verify_inputs is vg063.verify_inputs
        assert vg063.REJECTION in core.EXTRA_SOURCE_PATHS
    finally:
        (
            core.TAG, core.SCHEMA, core.STATE_SCHEMA, core.SEED, core.STEP_LIMIT,
            core.MINIMUM_SUCCESS_RATE, core.MINIMUM_RECORDS, core.PREREGISTRATION,
            core.RUNNER, core.DEFAULT_OUTPUT, core.EXTRA_SOURCE_PATHS, core.verify_inputs,
        ) = old
