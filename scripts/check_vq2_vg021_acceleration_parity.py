#!/usr/bin/env python3
"""Prove VG021 decode/grouping parity before any continuation update."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE, VQ2PhaseRecurrentActor
from scripts.eval_vq2_variable_gate_oracle import write_json_atomic
from scripts.train_vq2_variable_gate_dagger_refit import RecurrentChunkStream
from scripts.train_vq2_variable_gate_recurrent_bc import PHASE_TAIL_INDEX
import scripts.train_vq2_variable_gate_five_source_refit as refit
import scripts.train_vq2_variable_gate_three_source_refit as three_source


SCHEMA = "vq2_vg021_acceleration_parity_report_v1"


def _legacy_reconstruct(
    mask: np.ndarray, tail: np.ndarray, *, device: torch.device
) -> torch.Tensor:
    decoded_mask = torch.from_numpy(np.asarray(mask).copy()).float().mul_(1.0 / 255.0)
    decoded_tail = torch.from_numpy(np.asarray(tail, dtype=np.float32).copy())
    observation = torch.cat((decoded_mask, decoded_tail), -1)
    return observation.transpose(0, 1).contiguous().to(device)


def _synchronized_samples(
    fn: Callable[[], None], *, warmup: int = 3, samples: int = 9
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    result = []
    for _ in range(samples):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        result.append(time.perf_counter() - started)
    return result


def _model(saved: dict[str, Any], device: torch.device) -> VQ2PhaseRecurrentActor:
    config = refit.FiveSourceConfig()
    model = VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    model.load_state_dict(saved["model_state"])
    return model.train()


def check(*, migration_state: Path, output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("VG021 parity requires CUDA")
    if refit.sha256_path(migration_state) != refit.MIGRATION_STATE_SHA256:
        raise RuntimeError("VG021 parity migration hash mismatch")
    refit.verify_inputs()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=refit.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=refit.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("VG021 parity requires a clean tracked worktree")

    device = torch.device("cuda")
    saved = torch.load(migration_state, map_location="cpu", weights_only=False)
    if (
        saved.get("schema") != refit.MIGRATION_SCHEMA
        or saved.get("tag") != refit.MIGRATION_TAG
        or saved.get("source_commit") != refit.MIGRATION_SOURCE_COMMIT
        or saved.get("status") != "training"
        or saved.get("completed_epoch") != 3
        or saved.get("optimizer_updates") != 31_742
    ):
        raise RuntimeError("VG021 parity parent identity mismatch")

    config = refit.FiveSourceConfig()
    datasets = {
        "clean": refit.VariableGateBCDataset(
            refit.CLEAN_DATASET,
            expected_report_sha256=refit.CLEAN_REPORT_SHA256,
            expected_metadata_sha256=refit.CLEAN_METADATA_SHA256,
        ),
        "dagger1": refit.VariableGateBCDataset(
            refit.DAGGER1_DATASET,
            expected_report_sha256=refit.DAGGER1_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER1_METADATA_SHA256,
        ),
        "dagger2": refit.VariableGateBCDataset(
            refit.DAGGER2_DATASET,
            report_path=refit.DAGGER2_RECOVERY_REPORT,
            expected_report_sha256=refit.DAGGER2_RECOVERY_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER2_METADATA_SHA256,
        ),
        "dagger3": refit.VariableGateBCDataset(
            refit.DAGGER3_DATASET,
            expected_report_sha256=refit.DAGGER3_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER3_METADATA_SHA256,
        ),
        "dagger4": refit.VariableGateBCDataset(
            refit.DAGGER4_DATASET,
            expected_report_sha256=refit.DAGGER4_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER4_METADATA_SHA256,
        ),
    }
    split_counts = {
        "clean": config.clean_validation_agents,
        "dagger1": config.dagger1_validation_agents,
        "dagger2": config.dagger2_validation_agents,
        "dagger3": config.dagger3_validation_agents,
        "dagger4": config.dagger4_validation_agents,
    }
    splits = {
        name: three_source._dataset_split(dataset.agents, split_counts[name])
        for name, dataset in datasets.items()
    }

    # Actual stored rows prove that moving uint8 expansion to CUDA is lossless.
    indices = splits["clean"][0][: config.source_agent_batch_size]
    sample_mask = np.take(datasets["clean"].mask[: config.sequence_chunk], indices, axis=1)
    sample_tail = np.take(datasets["clean"].tail[: config.sequence_chunk], indices, axis=1)
    legacy_observation = _legacy_reconstruct(sample_mask, sample_tail, device=device)
    device_observation = refit.reconstruct_phase_batch(
        sample_mask, sample_tail, device=device
    )
    decode_max_error = float((legacy_observation - device_observation).abs().max())
    decode_bitwise = bool(torch.equal(legacy_observation, device_observation))

    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(saved["numpy_rng_state"])
    seed_model = _model(saved, device)
    streams = {
        name: RecurrentChunkStream(
            dataset,
            splits[name][0],
            batch_size=config.source_agent_batch_size,
            sequence_chunk=config.sequence_chunk,
            rng=rng,
            device=device,
            cyclic=name != "clean",
        )
        for name, dataset in datasets.items()
    }
    items = {}
    for name, stream in streams.items():
        item = stream.next(seed_model)
        if item is None:
            raise RuntimeError(f"VG021 parity {name} emitted no next chunk")
        items[name] = item
    groups = refit.prepare_source_groups(items)
    action_weights = torch.tensor(config.action_weights, device=device)

    def calculate(model: VQ2PhaseRecurrentActor, *, grouped: bool) -> tuple[
        dict[str, Any], dict[str, torch.Tensor], torch.Tensor
    ]:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if grouped:
                outputs, states = refit.forward_source_groups(model, groups)
            else:
                outputs, states = {}, {}
                for name, item in items.items():
                    outputs[name], states[name] = model.forward_sequence(
                        item.observation, item.start_state
                    )
            loss, _ = refit.five_source_loss(
                {name: value.mean for name, value in outputs.items()},
                {name: item.target for name, item in items.items()},
                {name: item.valid for name, item in items.items()},
                weights=action_weights,
                previous_predictions={
                    name: item.previous_prediction for name, item in items.items()
                },
                previous_valid={
                    name: item.previous_valid for name, item in items.items()
                },
                config=config,
            )
        loss.backward()
        return outputs, states, loss

    legacy_model = _model(saved, device)
    grouped_model = _model(saved, device)
    legacy_outputs, legacy_states, legacy_loss = calculate(legacy_model, grouped=False)
    grouped_outputs, grouped_states, grouped_loss = calculate(grouped_model, grouped=True)
    mean_error = max(
        float((legacy_outputs[name].mean - grouped_outputs[name].mean).abs().max())
        for name in items
    )
    pre_tanh_error = max(
        float(
            (
                legacy_outputs[name].pre_tanh_mean
                - grouped_outputs[name].pre_tanh_mean
            ).abs().max()
        )
        for name in items
    )
    state_error = max(
        float((legacy_states[name] - grouped_states[name]).abs().max())
        for name in items
    )
    loss_error = abs(float(legacy_loss.detach()) - float(grouped_loss.detach()))
    gradient_error = max(
        float((left.grad - right.grad).abs().max())
        for left, right in zip(legacy_model.parameters(), grouped_model.parameters())
        if left.grad is not None and right.grad is not None
    )

    def legacy_step() -> None:
        calculate(legacy_model, grouped=False)

    def grouped_step() -> None:
        calculate(grouped_model, grouped=True)

    legacy_samples = _synchronized_samples(legacy_step)
    grouped_samples = _synchronized_samples(grouped_step)
    legacy_median = statistics.median(legacy_samples)
    grouped_median = statistics.median(grouped_samples)
    speedup = legacy_median / grouped_median
    admitted = bool(
        decode_bitwise
        and decode_max_error == 0.0
        and mean_error <= 1e-6
        and pre_tanh_error <= 1e-6
        and state_error <= 1e-6
        and loss_error <= 1e-7
        and gradient_error <= 1e-5
        and speedup >= 1.5
    )
    report = {
        "schema": SCHEMA,
        "tag": refit.TAG,
        "source_commit": source_commit,
        "migration_state_sha256": refit.MIGRATION_STATE_SHA256,
        "admitted": admitted,
        "optimizer_steps": 0,
        "state_writes": 0,
        "chunk_horizons": {
            name: int(item.observation.shape[1]) for name, item in items.items()
        },
        "group_count": len(groups),
        "decode_bitwise_equal": decode_bitwise,
        "decode_max_abs_error": decode_max_error,
        "mean_max_abs_error": mean_error,
        "pre_tanh_max_abs_error": pre_tanh_error,
        "next_state_max_abs_error": state_error,
        "loss_abs_error": loss_error,
        "gradient_max_abs_error": gradient_error,
        "legacy_median_seconds": legacy_median,
        "grouped_median_seconds": grouped_median,
        "speedup": speedup,
        "runtime": three_source.runtime_manifest(),
        "safety": {
            "actor_input_privileged_values": 0,
            "teacher_blend": 0.0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    write_json_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(
        migration_state=args.migration_state.resolve(), output=args.output.resolve()
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
