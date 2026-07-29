#!/usr/bin/env python3
"""Create source-locked whole-Puffer checkpoint interpolants."""

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

from scripts.collect_vq2_oracle_bc_dataset import sha256_path


def interpolate_state_dict(
    base: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0,1]")
    if base.keys() != candidate.keys():
        raise ValueError("checkpoint state dictionaries differ")
    result: dict[str, torch.Tensor] = {}
    for name in base:
        left = base[name]
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"tensor contract differs for {name}")
        if left.is_floating_point():
            result[name] = left + alpha * (right - left)
        elif torch.equal(left, right):
            result[name] = left.clone()
        else:
            raise ValueError(f"non-floating tensor differs for {name}")
    return result


def generate(
    *,
    base_path: Path,
    candidate_path: Path,
    output: Path,
    alphas: tuple[float, ...],
    tag: str,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be nonempty and unique")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if (
        base.get("schema") != "vq2_public_phase_recurrent_checkpoint_v1"
        or candidate.get("schema") != base.get("schema")
        or candidate.get("model") != base.get("model")
    ):
        raise RuntimeError("checkpoint actor contract differs")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        state = interpolate_state_dict(
            base["model_state"], candidate["model_state"], alpha
        )
        payload = dict(base)
        payload["tag"] = f"{tag}_alpha_{alpha:.2f}"
        payload["model_state"] = state
        payload["combined_numerical_admission"] = False
        payload["offline_interpolation"] = {
            "alpha": alpha,
            "base_sha256": sha256_path(base_path),
            "candidate_sha256": sha256_path(candidate_path),
            "whole_checkpoint": True,
            "runtime_action_blend": False,
        }
        name = f"alpha_{alpha:.2f}".replace(".", "p") + ".pt"
        path = output / name
        torch.save(payload, path)
        records.append(
            {"alpha": alpha, "path": name, "sha256": sha256_path(path)}
        )
    manifest = {
        "schema": "vq2_whole_puffer_interpolation_manifest_v1",
        "tag": tag,
        "base": str(base_path.resolve().relative_to(ROOT)),
        "base_sha256": sha256_path(base_path),
        "candidate": str(candidate_path.resolve().relative_to(ROOT)),
        "candidate_sha256": sha256_path(candidate_path),
        "alphas": list(alphas),
        "candidates": records,
        "whole_checkpoint": True,
        "runtime_action_blend": False,
        "safety": {
            "flight_sim_packets_sent": 0,
            "student_updates": 0,
            "submission_actions": 0,
        },
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_path(
                Path(__file__).resolve()
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    manifest = generate(
        base_path=args.base,
        candidate_path=args.candidate,
        output=args.output,
        alphas=tuple(args.alphas),
        tag=args.tag,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
