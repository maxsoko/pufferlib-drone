#!/usr/bin/env python3
"""Prove VG022 device-decode parity before any continuation update."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase import VQ2PhaseRecurrentActor
from scripts.eval_vq2_variable_gate_oracle import write_json_atomic
from scripts.train_vq2_variable_gate_dagger_refit import RecurrentChunkStream
import scripts.train_vq2_variable_gate_five_source_refit as refit
import scripts.train_vq2_variable_gate_three_source_refit as three_source


SCHEMA = "vq2_vg022_device_decode_parity_report_v1"


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
        raise RuntimeError("VG022 parity requires CUDA")
    if refit.sha256_path(migration_state) != refit.MIGRATION_STATE_SHA256:
        raise RuntimeError("VG022 parity migration hash mismatch")
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
        raise RuntimeError("VG022 parity requires a clean tracked worktree")

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
        raise RuntimeError("VG022 parity parent identity mismatch")
    runtime = three_source.runtime_manifest()
    if runtime != saved.get("runtime"):
        raise RuntimeError(f"VG022 parity runtime mismatch: {runtime}")

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
    raw_chunks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, stream in streams.items():
        item = stream.next(seed_model)
        if item is None:
            raise RuntimeError(f"VG022 parity {name} emitted no next chunk")
        items[name] = item
        assert stream._batch_agents is not None
        end = int(stream._start)
        start = end - int(item.observation.shape[1])
        raw_chunks[name] = (
            np.take(datasets[name].mask[start:end], stream._batch_agents, axis=1),
            np.take(datasets[name].tail[start:end], stream._batch_agents, axis=1),
        )
    action_weights = torch.tensor(config.action_weights, device=device)

    def decoded_items(*, legacy: bool) -> dict[str, Any]:
        decoded = {}
        for name, item in items.items():
            mask, tail = raw_chunks[name]
            observation = (
                _legacy_reconstruct(mask, tail, device=device)
                if legacy
                else refit.reconstruct_phase_batch(mask, tail, device=device)
            )
            decoded[name] = replace(item, observation=observation)
        return decoded

    def calculate(
        model: VQ2PhaseRecurrentActor, run_items: dict[str, Any]
    ) -> tuple[
        dict[str, Any], dict[str, torch.Tensor], torch.Tensor
    ]:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs, states = {}, {}
            for name, item in run_items.items():
                outputs[name], states[name] = model.forward_sequence(
                    item.observation, item.start_state
                )
            loss, _ = refit.five_source_loss(
                {name: value.mean for name, value in outputs.items()},
                {name: item.target for name, item in run_items.items()},
                {name: item.valid for name, item in run_items.items()},
                weights=action_weights,
                previous_predictions={
                    name: item.previous_prediction for name, item in run_items.items()
                },
                previous_valid={
                    name: item.previous_valid for name, item in run_items.items()
                },
                config=config,
            )
        loss.backward()
        return outputs, states, loss

    legacy_items = decoded_items(legacy=True)
    device_items = decoded_items(legacy=False)
    decode_max_error = max(
        float(
            (
                legacy_items[name].observation - device_items[name].observation
            ).abs().max()
        )
        for name in items
    )
    decode_bitwise = all(
        torch.equal(
            legacy_items[name].observation, device_items[name].observation
        )
        for name in items
    )
    legacy_model = _model(saved, device)
    device_model = _model(saved, device)
    legacy_outputs, legacy_states, legacy_loss = calculate(legacy_model, legacy_items)
    device_outputs, device_states, device_loss = calculate(device_model, device_items)
    mean_error = max(
        float(
            (
                legacy_outputs[name].mean.detach()
                - device_outputs[name].mean.detach()
            ).abs().max()
        )
        for name in items
    )
    pre_tanh_error = max(
        float(
            (
                legacy_outputs[name].pre_tanh_mean.detach()
                - device_outputs[name].pre_tanh_mean.detach()
            ).abs().max()
        )
        for name in items
    )
    state_error = max(
        float(
            (
                legacy_states[name].detach() - device_states[name].detach()
            ).abs().max()
        )
        for name in items
    )
    loss_error = abs(float(legacy_loss.detach()) - float(device_loss.detach()))
    gradient_error = max(
        float((left.grad - right.grad).abs().max())
        for left, right in zip(legacy_model.parameters(), device_model.parameters())
        if left.grad is not None and right.grad is not None
    )

    def legacy_step() -> None:
        run_items = decoded_items(legacy=True)
        for _ in range(config.transition_window_exposure):
            calculate(legacy_model, run_items)

    def device_step() -> None:
        run_items = decoded_items(legacy=False)
        for _ in range(config.transition_window_exposure):
            calculate(device_model, run_items)

    legacy_samples = _synchronized_samples(legacy_step, warmup=2, samples=7)
    device_samples = _synchronized_samples(device_step, warmup=2, samples=7)
    legacy_median = statistics.median(legacy_samples)
    device_median = statistics.median(device_samples)
    speedup = legacy_median / device_median
    admitted = bool(
        decode_bitwise
        and decode_max_error == 0.0
        and mean_error == 0.0
        and pre_tanh_error == 0.0
        and state_error == 0.0
        and loss_error == 0.0
        and gradient_error == 0.0
        and speedup >= 1.15
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
        "transition_exposures_per_timed_step": config.transition_window_exposure,
        "decode_bitwise_equal": decode_bitwise,
        "decode_max_abs_error": decode_max_error,
        "mean_max_abs_error": mean_error,
        "pre_tanh_max_abs_error": pre_tanh_error,
        "next_state_max_abs_error": state_error,
        "loss_abs_error": loss_error,
        "gradient_max_abs_error": gradient_error,
        "legacy_median_seconds": legacy_median,
        "device_decode_median_seconds": device_median,
        "speedup": speedup,
        "runtime": runtime,
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
