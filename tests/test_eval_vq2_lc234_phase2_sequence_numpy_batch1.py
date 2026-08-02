from pathlib import Path

import scripts.eval_vq2_lc234_phase2_sequence_numpy_batch1 as lc234


def test_candidate_source_lock() -> None:
    lc234.verify_inputs()
    assert lc234.CHECKPOINT_SHA256.startswith("d3fd2544")
    assert lc234.SOURCE_CHECKPOINT_SHA256.startswith("ebab0d42")


def test_bound_policy_has_both_sequences() -> None:
    policy = lc234.bound_policy(lc234.CHECKPOINT)
    assert policy.values["phase2_action_sequence"].shape == (1433, 4)
    assert policy.values["phase_action_sequence"].shape == (9592, 4)


def test_two_stage_offline_authority() -> None:
    assert lc234.default_output(3).name.endswith("raw3_001")
    assert lc234.default_output(24).name.endswith("all24_001")
    text = Path(lc234.PREREGISTRATION).read_text()
    assert "FlightSim control stays frozen" in text
    assert "VQ2 Submission remains forbidden" in text
