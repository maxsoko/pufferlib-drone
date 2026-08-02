#!/usr/bin/env python3
"""Export the frozen LC216 recurrent Puffer checkpoint for NumPy deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc216_all24_action_sequence_001/policy_selected.pt"
)
SOURCE_SHA256 = "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
DEFAULT_OUTPUT = (
    ROOT
    / "checkpoints/vq2_lc216_all24_action_sequence_numpy.npz"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(source: Path, output: Path) -> dict[str, object]:
    if sha256_path(source) != SOURCE_SHA256:
        raise RuntimeError("LC216 source checkpoint changed")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        contract.get("class") != "VQ2PhaseActionSequenceActor"
        or int(contract.get("legal_observation_size", 0)) != 4119
        or int(contract.get("sequence_phase_min", -1)) != 16
        or int(contract.get("sequence_phase_max_exclusive", -1)) != 24
        or int(contract.get("sequence_length", 0)) != 9592
    ):
        raise RuntimeError("LC216 model contract changed")
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in payload["model_state"].items()
    }
    metadata = {
        "schema": "vq2_lc216_all24_numpy_checkpoint_v1",
        "source_checkpoint_sha256": SOURCE_SHA256,
        "model": contract,
    }
    arrays["__metadata_json__"] = np.asarray(
        json.dumps(metadata, sort_keys=True), dtype=np.str_
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    np.savez_compressed(output, **arrays)
    return {
        **metadata,
        "output": str(output),
        "output_sha256": sha256_path(output),
        "bytes": output.stat().st_size,
        "tensor_count": len(arrays) - 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
