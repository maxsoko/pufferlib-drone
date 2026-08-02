from pathlib import Path

import scripts.train_vq2_lc231_phase2_full_residual as lc231


def test_fit_contract() -> None:
    assert lc231.PHASES == (2,)
    assert lc231.OPTIMIZER_STEPS == 512
    assert lc231.BATCH_SIZE == 4096
    assert lc231.LEARNING_RATE == 3e-3


def test_causal_dataset_is_source_locked() -> None:
    parent = lc231.verify_inputs()
    assert parent["model"]["sequence_length"] == 9592
    assert lc231.FEATURES_SHA256 == (
        "761bfea1700a86c8a9923036c440d1fbb7e5c535fe36c8f5d7d23d1b35e1273a"
    )


def test_authority_stays_offline() -> None:
    text = Path(lc231.PREREGISTRATION).read_text()
    assert "receives no FlightSim authority" in text
    assert "VQ2 Submission remains" in text
