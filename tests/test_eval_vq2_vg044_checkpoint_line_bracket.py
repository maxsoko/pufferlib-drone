from __future__ import annotations

import os

import pytest
import torch

import scripts.eval_vq2_vg044_checkpoint_line_bracket as vg044


def test_vg044_fixed_fresh_fixture_and_small_alphas() -> None:
    assert vg044.SEED == 429158
    assert vg044.AGENTS == vg044.EPISODES == 64
    assert vg044.NUM_THREADS == 4
    assert vg044.MAX_STEPS == 2560
    assert vg044.ALPHAS == (
        0.0,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
    )


def test_vg044_interpolation_retains_exact_endpoints() -> None:
    parent = {
        "a": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "b": torch.tensor([3], dtype=torch.int64),
    }
    update = {
        "a": torch.tensor([5.0, 2.0], dtype=torch.float32),
        "b": torch.tensor([3], dtype=torch.int64),
    }
    alpha0 = vg044.interpolate_state(parent, update, 0.0)
    alpha1 = vg044.interpolate_state(parent, update, 1.0)
    middle = vg044.interpolate_state(parent, update, 0.25)
    assert torch.equal(alpha0["a"], parent["a"])
    assert torch.equal(alpha1["a"], update["a"])
    assert torch.equal(middle["a"], torch.tensor([2.0, -1.0]))
    assert torch.equal(middle["b"], parent["b"])
    assert vg044.state_sha256(alpha0) != vg044.state_sha256(middle)


def test_vg044_rejects_nonfloating_endpoint_mismatch() -> None:
    with pytest.raises(RuntimeError, match="nonfloating endpoint"):
        vg044.interpolate_state(
            {"a": torch.tensor([1], dtype=torch.int64)},
            {"a": torch.tensor([2], dtype=torch.int64)},
            0.1,
        )


def test_vg044_qualification_requires_early_safety_and_downstream_gain() -> None:
    parent = {
        "hard_transport_pass": True,
        "gate_reach": {"1": 64, "2": 60, "3": 12, "4": 1},
        "crashes": 2,
        "successes": 0,
        "mean_gates_passed": 2.1,
    }
    candidate = {
        **parent,
        "gate_reach": {"1": 64, "2": 61, "3": 13, "4": 1},
    }
    assert vg044.qualifies(parent, candidate)
    candidate["crashes"] = 3
    assert not vg044.qualifies(parent, candidate)
    candidate["crashes"] = 2
    candidate["gate_reach"]["2"] = 59
    assert not vg044.qualifies(parent, candidate)


def test_vg044_inputs_and_runner_are_source_locked_and_offline() -> None:
    required = (
        vg044.PARENT_CHECKPOINT,
        vg044.PARENT_REPORT,
        vg044.UPDATE_CHECKPOINT,
        vg044.UPDATE_REPORT,
    )
    if all(path.is_file() for path in required):
        vg044.verify_inputs()
        parent, update = vg044.load_endpoints()
        assert parent["model"] == update["model"]
    text = vg044.RUNNER.read_text()
    assert os.access(vg044.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "OMP_NUM_THREADS=4 MKL_NUM_THREADS=1" in text
    assert "tests/test_eval_vq2_vg044_checkpoint_line_bracket.py" in text
    assert "--resume" in text
    assert (
        'ROOT / "tests/test_eval_vq2_vg044_checkpoint_line_bracket.py"'
        in vg044.source_identity.__code__.co_consts
        or "tests/test_eval_vq2_vg044_checkpoint_line_bracket.py"
        in vg044.Path(vg044.__file__).read_text()
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
