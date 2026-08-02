from pathlib import Path

import scripts.eval_vq2_lc232_phase2_numpy_batch1_native as lc232


def test_candidate_source_lock() -> None:
    lc232.verify_inputs()
    assert lc232.CHECKPOINT_SHA256.startswith("a0156a42")
    assert lc232.SOURCE_CHECKPOINT_SHA256.startswith("ba16dd1c")


def test_bound_policy_uses_candidate_identity() -> None:
    policy = lc232.bound_policy(lc232.CHECKPOINT)
    assert policy.values["phase_action_sequence"].shape == (9592, 4)


def test_two_stage_offline_authority() -> None:
    assert lc232.default_output(3).name.endswith("raw3_001")
    assert lc232.default_output(24).name.endswith("all24_001")
    text = Path(lc232.PREREGISTRATION).read_text()
    assert "FlightSim control remains" in text
    assert "VQ2 Submission remains forbidden" in text
