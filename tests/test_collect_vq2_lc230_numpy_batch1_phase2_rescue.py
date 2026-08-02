from pathlib import Path

import scripts.collect_vq2_lc230_numpy_batch1_phase2_rescue as lc230


def test_pairing_and_phase_contract() -> None:
    assert lc230.GROUP_SIZE == 8
    assert lc230.TOTAL_AGENTS == 16
    assert lc230.ENV_SEED_GROUP_SIZE == lc230.GROUP_SIZE
    assert (lc230.PHASE_MIN, lc230.PHASE_MAX_EXCLUSIVE) == (2, 3)
    assert lc230.TARGET_RAW_INDEX == 3


def test_bound_failure_and_authority() -> None:
    lc230.verify_inputs()
    text = Path(lc230.PREREGISTRATION).read_text()
    assert "controls reproduce 0/8 Gate-3 passes" in text
    assert "VQ2 Submission remains forbidden" in text


def test_numpy_actor_adapter_is_explicitly_unbatched() -> None:
    assert "for index, policy in enumerate(self.policies)" in Path(
        lc230.__file__
    ).read_text()
