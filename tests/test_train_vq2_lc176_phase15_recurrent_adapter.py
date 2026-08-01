from __future__ import annotations

import ast
import inspect
import numpy as np

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
import scripts.train_vq2_lc176_phase15_recurrent_adapter as lc176


def test_lc176_source_lock() -> None:
    parent = lc176.verify_inputs()
    assert parent["numerically_admitted"]


def test_lc176_sequence_contract() -> None:
    records = np.memmap(lc176.FEATURES, dtype=FEATURE_DTYPE, mode="r")
    contract = lc176.sequence_contract(records)
    assert contract["agents"] == 512
    assert contract["control_length_min"] == contract["control_length_max"] == 481
    assert contract["rescue_length_min"] == contract["rescue_length_max"] == 986
    assert contract["contiguous"]


def test_lc176_agent_split_and_model_contract() -> None:
    validation = np.arange(0, 512, 4)
    training = np.setdiff1d(np.arange(512), validation)
    assert len(training) == 384
    assert len(validation) == 128
    assert (validation < 256).sum() == 64
    assert (validation >= 256).sum() == 64
    assert lc176.TARGET_PHASE == 15
    assert lc176.ADAPTER_SIZE == 64


def test_lc176_fit_does_not_shadow_output_path() -> None:
    tree = ast.parse(inspect.getsource(lc176.fit))
    stores = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "output" not in stores
