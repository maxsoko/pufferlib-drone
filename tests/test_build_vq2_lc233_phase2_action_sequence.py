import numpy as np

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
import scripts.build_vq2_lc233_phase2_action_sequence as lc233


def test_source_sequence_is_exact_and_contiguous() -> None:
    lc233.verify_inputs()
    records = np.memmap(lc233.FEATURES, dtype=FEATURE_DTYPE, mode="r")
    steps, actions = lc233.source_sequence(records)
    assert len(actions) == 1433
    assert (int(steps[0]), int(steps[-1])) == (1760, 3192)
    assert np.array_equal(np.diff(steps), np.ones(1432, dtype=np.int64))


def test_only_one_new_checkpoint_tensor() -> None:
    parent = lc233.verify_inputs()
    assert "phase2_action_sequence" not in parent["model_state"]
    assert parent["model"]["sequence_length"] == 9592


def test_authority_remains_offline() -> None:
    text = lc233.PREREGISTRATION.read_text()
    assert "FlightSim\nremains frozen" in text
    assert "VQ2 Submission remains forbidden" in text
