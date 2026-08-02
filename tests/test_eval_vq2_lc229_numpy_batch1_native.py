from pathlib import Path

import numpy as np
import pytest

import scripts.eval_vq2_lc229_numpy_batch1_native as lc229


def test_target_contract() -> None:
    assert lc229.max_steps_for_target(3) == 10_000
    assert lc229.max_steps_for_target(24) == 45_000
    with pytest.raises(ValueError):
        lc229.max_steps_for_target(6)


def test_deployment_observation_is_one_unbatched_row() -> None:
    legal = np.arange(lc229.LEGAL_OBS_SIZE, dtype=np.float32)
    result = lc229.deployment_observation(legal, 0.5)
    assert result.shape == (lc229.OBSERVATION_SIZE,)
    assert np.array_equal(result[:-1], legal)
    assert result[-1] == np.float32(0.5)


def test_source_lock_and_no_submission_authority() -> None:
    lc229.verify_inputs()
    text = Path(lc229.PREREGISTRATION).read_text()
    assert "eight independent `NumpyLC216Policy` objects" in text
    assert "VQ2 Submission remains forbidden" in text
