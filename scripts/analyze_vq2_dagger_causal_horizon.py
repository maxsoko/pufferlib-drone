#!/usr/bin/env python3
"""Localize SF026 imitation error over the SF024 failure horizon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent import VQ2RecurrentActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_recurrent_bc import OracleBCDataset


TAG = "vq2_sf027_dagger_causal_horizon"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf027_dagger_causal_horizon_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf026_recurrent_dagger_aggregate_continuation_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "166c9b28bc1d4848e0e5e6d7028f9637835540cbe2f8e0ae883a8962d8e0e2b7"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "a1df63ceb0bf7fe85cc50a0010ed32d08395d76bc4f162b38aa900a67c643c03"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf024_recurrent_dagger_next_512"
)
DATASET_REPORT_SHA256 = (
    "5f3d344643a75a66cdbb1b159ee2105c563201774d8e7122d19e888fc414cb26"
)
DATASET_METADATA_SHA256 = (
    "1b7491bc8130ff1cb1a2a6b4c64cf7abb580723ccaa65600f842df3af8a498fa"
)
BINS = ((0, 128), (128, 256), (256, 384), (384, 512), (512, 768), (768, 1024), (1024, 1400))
CAPS = (128, 256, 384, 512, 1400)
ACTION_WEIGHTS = np.asarray((1.0, 1.0, 4.0, 1.0), dtype=np.float64)


def summarize_horizon(
    square_sum: np.ndarray, count: int
) -> dict[str, object]:
    mse = square_sum / max(count, 1)
    return {
        "count": int(count),
        "mse": mse.tolist(),
        "weighted_mse": float(np.dot(mse, ACTION_WEIGHTS) / ACTION_WEIGHTS.sum()),
    }


def analyze(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF027 preregisters CUDA inference")
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF026 checkpoint hash mismatch")
    if sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("SF026 report hash mismatch")
    device = torch.device(device_name)
    dataset = OracleBCDataset(
        DATASET,
        verify_hashes=True,
        expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
    )
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model_contract = payload["model"]
    model = VQ2RecurrentActor(
        hidden_size=int(model_contract["hidden_size"]),
        initial_std=float(model_contract["initial_std"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    agents = np.arange(448, 512, dtype=np.int64)
    state = model.initial_state(len(agents), device=device)
    bin_square = {interval: np.zeros(4, dtype=np.float64) for interval in BINS}
    bin_count = {interval: 0 for interval in BINS}
    bin_mask_mass = {interval: 0.0 for interval in BINS}
    with torch.no_grad():
        for start in range(0, dataset.time_steps, 64):
            end = min(start + 64, dataset.time_steps)
            legal, target, valid = dataset.chunk(
                agents, start, end, device=device
            )
            actor_output, state = model.forward_sequence(legal, state)
            error = (actor_output.mean - target).square().float().cpu().numpy()
            valid_np = valid.cpu().numpy()
            mask_mass = legal[:, :, :4096].sum(-1).cpu().numpy()
            for interval in BINS:
                lo = max(start, interval[0])
                hi = min(end, interval[1])
                if hi <= lo:
                    continue
                subset = slice(lo - start, hi - start)
                selected = valid_np[:, subset]
                count = int(selected.sum())
                if count == 0:
                    continue
                bin_square[interval] += (
                    error[:, subset] * selected[..., None]
                ).sum((0, 1))
                bin_mask_mass[interval] += float(
                    (mask_mass[:, subset] * selected).sum()
                )
                bin_count[interval] += count

    bins = []
    for interval in BINS:
        summary = summarize_horizon(bin_square[interval], bin_count[interval])
        summary.update(
            {
                "start": interval[0],
                "end": interval[1],
                "mean_mask_mass": bin_mask_mass[interval]
                / max(bin_count[interval], 1),
            }
        )
        bins.append(summary)
    horizons = {}
    for cap in CAPS:
        square = np.zeros(4, dtype=np.float64)
        count = 0
        for interval in BINS:
            if interval[1] <= cap:
                square += bin_square[interval]
                count += bin_count[interval]
        horizons[str(cap)] = summarize_horizon(square, count)

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        PREREGISTRATION,
        CHECKPOINT,
        TRAIN_REPORT,
        DATASET / "report.json",
        DATASET / "metadata.json",
    ]
    report = {
        "schema": "vq2_dagger_causal_horizon_analysis_v1",
        "tag": TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "validation_agents": agents.tolist(),
        "bins": bins,
        "horizons": horizons,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
        "safety": {
            "student_actions_executed": 0,
            "teacher_actions_executed": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
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
    print(json.dumps(analyze(output=args.output.resolve(), device_name=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
