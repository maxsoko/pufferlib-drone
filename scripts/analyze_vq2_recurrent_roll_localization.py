#!/usr/bin/env python3
"""Localize SF031 held-out roll error over the SF030 causal prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent import VQ2RecurrentActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_recurrent_bc import OracleBCDataset


TAG = "vq2_sf032_recurrent_roll_localization"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf032_recurrent_roll_localization_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf031_recurrent_dagger_causal_prefix_next_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "b7c0e1b0516a2ae410b77a4558c0298fd2fe8c11793e7e814d2009cc88b336fc"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "8c22f07a40ce12737a88db930d92c9e1a7f0298d2392884b9cb0dfbbf2380649"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf030_recurrent_dagger_causal_prefix_next_512"
)
DATASET_REPORT_SHA256 = (
    "7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393"
)
DATASET_METADATA_SHA256 = (
    "99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2"
)
BINS = (
    (0, 128),
    (128, 192),
    (192, 256),
    (256, 320),
    (320, 384),
    (384, 448),
    (448, 512),
)
CAPS = (128, 192, 256, 320, 384, 448, 512)
ACTION_WEIGHTS = np.asarray((1.0, 1.0, 4.0, 1.0), dtype=np.float64)
VALIDATION_AGENTS = np.arange(448, 512, dtype=np.int64)


def action_summary(
    square_sum: np.ndarray,
    absolute_sum: np.ndarray,
    count: int,
) -> dict[str, Any]:
    divisor = max(count, 1)
    mse = square_sum / divisor
    return {
        "count": int(count),
        "mse": mse.tolist(),
        "mae": (absolute_sum / divisor).tolist(),
        "weighted_mse": float(np.dot(mse, ACTION_WEIGHTS) / ACTION_WEIGHTS.sum()),
    }


def moment_summary(total: float, square: float, count: int) -> dict[str, float | int]:
    divisor = max(count, 1)
    mean = total / divisor
    return {
        "count": int(count),
        "mean": float(mean),
        "rms": float(np.sqrt(square / divisor)),
        "std": float(np.sqrt(max(square / divisor - mean * mean, 0.0))),
    }


def analyze(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF032 preregisters CUDA inference")
    frozen = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        DATASET / "report.json": DATASET_REPORT_SHA256,
        DATASET / "metadata.json": DATASET_METADATA_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")

    device = torch.device(device_name)
    dataset = OracleBCDataset(
        DATASET,
        verify_hashes=True,
        expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
    )
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    contract = payload["model"]
    model = VQ2RecurrentActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    bin_square = {interval: np.zeros(4, dtype=np.float64) for interval in BINS}
    bin_absolute = {interval: np.zeros(4, dtype=np.float64) for interval in BINS}
    bin_count = {interval: 0 for interval in BINS}
    visibility_square = {
        "mask_present": np.zeros(4, dtype=np.float64),
        "mask_absent": np.zeros(4, dtype=np.float64),
    }
    visibility_absolute = {
        key: np.zeros(4, dtype=np.float64) for key in visibility_square
    }
    visibility_count = {key: 0 for key in visibility_square}
    agent_roll_square = np.zeros(len(VALIDATION_AGENTS), dtype=np.float64)
    agent_roll_absolute = np.zeros(len(VALIDATION_AGENTS), dtype=np.float64)
    agent_count = np.zeros(len(VALIDATION_AGENTS), dtype=np.int64)
    target_roll_sum = target_roll_square = 0.0
    prediction_roll_sum = prediction_roll_square = 0.0

    state = model.initial_state(len(VALIDATION_AGENTS), device=device)
    with torch.no_grad():
        for start in range(0, dataset.time_steps, 64):
            end = min(start + 64, dataset.time_steps)
            legal, target, valid = dataset.chunk(
                VALIDATION_AGENTS, start, end, device=device
            )
            actor_output, state = model.forward_sequence(legal, state)
            difference = (actor_output.mean - target).float().cpu().numpy()
            square = np.square(difference)
            absolute = np.abs(difference)
            valid_np = valid.cpu().numpy()
            mask_present = legal[:, :, :4096].sum(-1).cpu().numpy() > 0.0

            for interval in BINS:
                lo = max(start, interval[0])
                hi = min(end, interval[1])
                if hi <= lo:
                    continue
                section = slice(lo - start, hi - start)
                selected = valid_np[:, section]
                count = int(selected.sum())
                if count:
                    bin_square[interval] += (
                        square[:, section] * selected[..., None]
                    ).sum((0, 1))
                    bin_absolute[interval] += (
                        absolute[:, section] * selected[..., None]
                    ).sum((0, 1))
                    bin_count[interval] += count

            for key, visible in (
                ("mask_present", mask_present),
                ("mask_absent", ~mask_present),
            ):
                selected = valid_np & visible
                count = int(selected.sum())
                if count:
                    visibility_square[key] += (
                        square * selected[..., None]
                    ).sum((0, 1))
                    visibility_absolute[key] += (
                        absolute * selected[..., None]
                    ).sum((0, 1))
                    visibility_count[key] += count

            agent_roll_square += (square[..., 1] * valid_np).sum(1)
            agent_roll_absolute += (absolute[..., 1] * valid_np).sum(1)
            agent_count += valid_np.sum(1)
            target_roll = target[..., 1].float().cpu().numpy()
            prediction_roll = actor_output.mean[..., 1].float().cpu().numpy()
            target_roll_sum += float((target_roll * valid_np).sum())
            target_roll_square += float((np.square(target_roll) * valid_np).sum())
            prediction_roll_sum += float((prediction_roll * valid_np).sum())
            prediction_roll_square += float(
                (np.square(prediction_roll) * valid_np).sum()
            )

    bins = []
    for interval in BINS:
        summary = action_summary(
            bin_square[interval], bin_absolute[interval], bin_count[interval]
        )
        summary.update({"start": interval[0], "end": interval[1]})
        bins.append(summary)

    horizons: dict[str, Any] = {}
    for cap in CAPS:
        square = np.zeros(4, dtype=np.float64)
        absolute = np.zeros(4, dtype=np.float64)
        count = 0
        for interval in BINS:
            if interval[1] <= cap:
                square += bin_square[interval]
                absolute += bin_absolute[interval]
                count += bin_count[interval]
        horizons[str(cap)] = action_summary(square, absolute, count)

    per_agent_roll_mse = agent_roll_square / np.maximum(agent_count, 1)
    per_agent_roll_mae = agent_roll_absolute / np.maximum(agent_count, 1)
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    worst_order = np.argsort(per_agent_roll_mse)[::-1][:10]
    total_count = int(agent_count.sum())
    report = {
        "schema": "vq2_recurrent_roll_localization_v1",
        "tag": TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "train_report_sha256": TRAIN_REPORT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "validation_agents": VALIDATION_AGENTS.tolist(),
        "bins": bins,
        "horizons": horizons,
        "visibility": {
            key: action_summary(
                visibility_square[key],
                visibility_absolute[key],
                visibility_count[key],
            )
            for key in visibility_square
        },
        "roll_moments": {
            "target": moment_summary(target_roll_sum, target_roll_square, total_count),
            "prediction": moment_summary(
                prediction_roll_sum, prediction_roll_square, total_count
            ),
        },
        "per_agent_roll_mse_quantiles": {
            str(quantile): float(np.quantile(per_agent_roll_mse, quantile))
            for quantile in quantiles
        },
        "worst_agents": [
            {
                "agent": int(VALIDATION_AGENTS[index]),
                "length": int(dataset.lengths[VALIDATION_AGENTS[index]]),
                "count": int(agent_count[index]),
                "roll_mse": float(per_agent_roll_mse[index]),
                "roll_mae": float(per_agent_roll_mae[index]),
            }
            for index in worst_order
        ],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path)
            for path in (
                Path(__file__).resolve(),
                PREREGISTRATION,
                ROOT / "pufferlib/vq2_recurrent.py",
                ROOT / "scripts/train_vq2_recurrent_bc.py",
                CHECKPOINT,
                TRAIN_REPORT,
                DATASET / "report.json",
                DATASET / "metadata.json",
            )
        },
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
            "successor_authorized": False,
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(output=args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
