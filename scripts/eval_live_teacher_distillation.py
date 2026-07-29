#!/usr/bin/env python3
"""Evaluate a 32-input checkpoint on held-out live-teacher BC episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy
from train_full_policy_bc import ACTIONS, OBSERVATIONS, load_episodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_phase_metrics() -> dict:
    return {
        "samples": 0,
        "squared_error": np.zeros(ACTIONS, dtype=np.float64),
        "max_abs_error": np.zeros(ACTIONS, dtype=np.float64),
        "squared_source_drift": np.zeros(ACTIONS, dtype=np.float64),
    }


def _finish_phase_metrics(metrics: dict) -> dict:
    samples = int(metrics["samples"])
    return {
        "samples": samples,
        "rmse": np.sqrt(metrics["squared_error"] / max(samples, 1)).tolist(),
        "max_abs_error": metrics["max_abs_error"].tolist(),
        "source_drift_rmse": np.sqrt(
            metrics["squared_source_drift"] / max(samples, 1)
        ).tolist(),
    }


def evaluate(
    checkpoint_path: Path,
    datasets: list[Path],
    *,
    source_path: Path | None = None,
    layout_precision_bytes: int = 4,
) -> dict:
    episodes = load_episodes(datasets)
    model = CheckpointPolicy.load(
        str(checkpoint_path),
        input_dim=OBSERVATIONS,
        layout_precision_bytes=layout_precision_bytes,
    )
    source = None
    if source_path is not None:
        source = CheckpointPolicy.load(
            str(source_path),
            input_dim=OBSERVATIONS,
            layout_precision_bytes=layout_precision_bytes,
        )

    phases: dict[int, dict] = {}
    for episode in episodes:
        model.reset_state()
        if source is not None:
            source.reset_state()
        for record in episode:
            observation = record[:OBSERVATIONS]
            target = record[OBSERVATIONS : OBSERVATIONS + ACTIONS]
            prediction = np.asarray(model.infer(observation), dtype=np.float64)
            source_prediction = (
                np.asarray(source.infer(observation), dtype=np.float64)
                if source is not None
                else prediction
            )
            phase = int(round(float(observation[23]) * 6.0))
            metrics = phases.setdefault(phase, _new_phase_metrics())
            error = prediction - target
            source_drift = prediction - source_prediction
            metrics["samples"] += 1
            metrics["squared_error"] += error * error
            metrics["max_abs_error"] = np.maximum(
                metrics["max_abs_error"], np.abs(error)
            )
            metrics["squared_source_drift"] += source_drift * source_drift

    isolation = None
    if source is not None:
        selected = {25, 26}
        encoder_equal = {
            str(index): bool(
                np.array_equal(model.encoder[:, index], source.encoder[:, index])
            )
            for index in range(OBSERVATIONS)
        }
        outside_selected_exact = all(
            encoder_equal[str(index)]
            for index in range(OBSERVATIONS)
            if index not in selected
        )
        mingru_layer_exact = [
            bool(np.array_equal(child, parent))
            for child, parent in zip(model.mingru_proj, source.mingru_proj)
        ]
        decoder_row_exact = [
            bool(np.array_equal(model.decoder[index], source.decoder[index]))
            for index in range(model.decoder.shape[0])
        ]
        isolation = {
            "passed": bool(
                outside_selected_exact
                and np.array_equal(model.decoder, source.decoder)
                and np.array_equal(model.log_std, source.log_std)
                and all(mingru_layer_exact)
            ),
            "registered_n111_passed": bool(
                outside_selected_exact
                and np.array_equal(model.log_std, source.log_std)
                and mingru_layer_exact[:2] == [True, True]
                and decoder_row_exact[-1]
            ),
            "changed_encoder_columns": [
                index
                for index in range(OBSERVATIONS)
                if not encoder_equal[str(index)]
            ],
            "decoder_exact": bool(np.array_equal(model.decoder, source.decoder)),
            "decoder_row_exact": decoder_row_exact,
            "log_std_exact": bool(np.array_equal(model.log_std, source.log_std)),
            "mingru_exact": bool(all(mingru_layer_exact)),
            "mingru_layer_exact": mingru_layer_exact,
        }

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source": None if source_path is None else str(source_path.resolve()),
        "source_sha256": None if source_path is None else sha256_file(source_path),
        "datasets": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in datasets
        ],
        "episodes": len(episodes),
        "phases": {
            str(phase): _finish_phase_metrics(metrics)
            for phase, metrics in sorted(phases.items())
        },
        "isolation": isolation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--checkpoint-layout-precision-bytes", type=int, choices=(2, 4), default=4
    )
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint,
        args.dataset,
        source_path=args.source,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
