from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.train_vq2_recurrent_bc import weighted_action_mse
import scripts.train_vq2_variable_gate_dagger_refit as refit
from scripts.train_vq2_variable_gate_dagger_refit import (
    RecurrentChunkStream,
    RefitConfig,
    source_balanced_loss,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_vq2_vg010_vast.sh"


class _FakeDataset:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def chunk(
        self,
        agents: np.ndarray,
        start: int,
        end: int,
        *,
        device: torch.device,
    ):
        indices = np.asarray(agents, dtype=np.int64)
        self.calls.append((tuple(map(int, indices)), start, end))
        steps = end - start
        observation = torch.zeros(
            (len(indices), steps, PHASE_LEGAL_OBS_SIZE), device=device
        )
        target = torch.zeros((len(indices), steps, ACTION_SIZE), device=device)
        timeline = torch.arange(start, end, device=device)[None]
        length_tensor = torch.from_numpy(self.lengths[indices]).to(device)[:, None]
        valid = timeline < length_tensor
        transition = torch.zeros_like(valid)
        observation[..., 0] = timeline
        return observation, target, valid, transition


class _FakeModel:
    hidden_size = 3

    def initial_state(self, batch_size: int, *, device: torch.device):
        return torch.zeros((1, batch_size, self.hidden_size), device=device)


def test_source_balanced_loss_is_equal_weight_despite_record_counts() -> None:
    config = replace(RefitConfig(), smoothness_weight=0.0)
    weights = torch.tensor(config.action_weights)
    clean_prediction = torch.zeros((1, 1, ACTION_SIZE))
    clean_target = torch.ones_like(clean_prediction)
    clean_valid = torch.ones((1, 1), dtype=torch.bool)
    dagger_prediction = torch.zeros((1, 7, ACTION_SIZE))
    dagger_target = torch.full_like(dagger_prediction, 2.0)
    dagger_valid = torch.ones((1, 7), dtype=torch.bool)
    clean_loss, _ = weighted_action_mse(
        clean_prediction, clean_target, clean_valid, weights
    )
    dagger_loss, _ = weighted_action_mse(
        dagger_prediction, dagger_target, dagger_valid, weights
    )
    loss, components = source_balanced_loss(
        clean_prediction,
        clean_target,
        clean_valid,
        dagger_prediction,
        dagger_target,
        dagger_valid,
        weights=weights,
        clean_previous_prediction=None,
        clean_previous_valid=None,
        dagger_previous_prediction=None,
        dagger_previous_valid=None,
        config=config,
    )
    assert torch.equal(loss, 0.5 * clean_loss + 0.5 * dagger_loss)
    assert torch.equal(components["clean_total"], clean_loss)
    assert torch.equal(components["dagger_total"], dagger_loss)


def test_recurrent_chunk_stream_preserves_contiguous_batches_and_all_records() -> None:
    dataset = _FakeDataset([2, 5, 3, 1])
    model = _FakeModel()
    stream = RecurrentChunkStream(
        dataset,
        np.arange(4),
        batch_size=2,
        sequence_chunk=2,
        rng=np.random.default_rng(9),
        device=torch.device("cpu"),
        cyclic=False,
    )
    chunks = 0
    while (item := stream.next(model)) is not None:
        output = torch.zeros((*item.valid.shape, ACTION_SIZE))
        stream.commit(output, item.start_state, item.valid)
        chunks += 1
    assert chunks == stream.source_chunks
    assert stream.source_records == sum(dataset.lengths)
    grouped: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for agents, start, end in dataset.calls:
        grouped.setdefault(agents, []).append((start, end))
    assert len(grouped) == 2
    for intervals in grouped.values():
        assert intervals[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_dagger_stream_cycles_only_at_episode_batch_boundaries() -> None:
    dataset = _FakeDataset([1, 3])
    model = _FakeModel()
    stream = RecurrentChunkStream(
        dataset,
        np.arange(2),
        batch_size=1,
        sequence_chunk=2,
        rng=np.random.default_rng(4),
        device=torch.device("cpu"),
        cyclic=True,
    )
    while stream.cycles_started < 3:
        item = stream.next(model)
        assert item is not None
        output = torch.zeros((*item.valid.shape, ACTION_SIZE))
        stream.commit(output, item.start_state, item.valid)
    starts_by_call = [start for _agents, start, _end in dataset.calls]
    assert starts_by_call.count(0) >= 5
    assert stream.cycles_started == 3


def test_refit_config_keeps_required_bptt_exposure_and_equal_source_weights() -> None:
    config = RefitConfig()
    assert config.sequence_chunk >= 256
    assert config.transition_window_exposure >= 3
    assert config.clean_objective_weight == pytest.approx(0.5)
    assert config.dagger_objective_weight == pytest.approx(0.5)
    assert config.learning_rate == pytest.approx(3e-5)


def test_vg010_vast_runner_is_source_locked_resumable_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "vq2_vg010_variable_gate_source_balanced_refit_001" in text
    assert "git clang ccache nvcc nvidia-smi python" in text
    assert "clang -fopenmp -x c - -fsyntax-only" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "assert _C.precision_bytes == 4" in text
    assert "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest" in text
    assert "tests/test_train_vq2_variable_gate_dagger_refit.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index("test_drone_race_native_regressions")
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text


def test_completed_output_resume_binds_report_checkpoint_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "policy_best.pt"
    checkpoint_path.write_bytes(b"source-locked-checkpoint")
    source_commit = "f" * 40
    source_sha256 = {"scripts/trainer.py": "e" * 64}
    monkeypatch.setattr(
        refit,
        "current_source_identity",
        lambda: (source_commit, source_sha256),
    )
    report = {
        "schema": "vq2_variable_gate_source_balanced_refit_report_v1",
        "tag": refit.TAG,
        "completed": True,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "checkpoint_sha256": refit.sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": refit.PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": refit.PARENT_REPORT_SHA256,
        "clean_report_sha256": refit.CLEAN_REPORT_SHA256,
        "clean_metadata_sha256": refit.CLEAN_METADATA_SHA256,
        "dagger_report_sha256": refit.DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": refit.DAGGER_METADATA_SHA256,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True))
    state = {
        "schema": "vq2_variable_gate_source_balanced_refit_state_v1",
        "tag": refit.TAG,
        "status": "completed",
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report_sha256": refit.sha256_path(report_path),
    }
    state_path = tmp_path / "training_state.pt"
    torch.save(state, state_path)
    refit.verify_completed_output(tmp_path, report)

    state["status"] = "training"
    torch.save(state, state_path)
    with pytest.raises(RuntimeError, match="does not bind"):
        refit.verify_completed_output(tmp_path, report)
