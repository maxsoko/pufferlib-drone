#!/usr/bin/env python3
"""Read-only float32/BF16 drift audit for a sequence-only VQ2 child."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.continue_vq2_plane_representation_offline import (
    _progress_metrics,
    _reference,
    _rolling_samples,
)
from scripts.continue_vq2_plane_sequence_offline import _predictions
from scripts.train_vq2_informed_dreamer import QuantizedSequenceReplay


AUDIT_SCHEMA = "vq2_sequence_precision_drift_v1"


def _model(payload: dict, device: torch.device) -> VQ2InformedDreamer:
    args = payload.get("args")
    if not isinstance(args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    model = VQ2InformedDreamer(
        deterministic_size=int(args["deterministic_size"]),
        stochastic_groups=int(args["stochastic_groups"]),
        stochastic_classes=int(args["stochastic_classes"]),
        actor_initial_std=float(args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(args.get("actor_distribution_mode", "legacy_tanh_normal")),
        distributional_reward=bool(args.get("distributional_reward", False)),
        action_conditioned_reward=bool(args.get("action_conditioned_reward", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _drift(parent: dict[str, torch.Tensor], child: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "rmse": float((child[name] - parent[name]).square().mean().sqrt().cpu()),
            "max_abs": float((child[name] - parent[name]).abs().max().cpu()),
        }
        for name in parent
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    source_hashes_before = {
        "parent": _sha256(args.parent_checkpoint),
        "child": _sha256(args.child_checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n637_report": _sha256(args.n637_report),
        "replay": _replay_hashes(args.replay_dir),
    }
    parent_payload = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False
    )
    child_payload = torch.load(
        args.child_checkpoint, map_location=device, weights_only=False
    )
    parent = _model(parent_payload, device)
    child = _model(child_payload, device)
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    changed = [
        name for name, value in child_state.items() if not torch.equal(value, parent_state[name])
    ]
    forbidden = [name for name in changed if not name.startswith("rssm.sequence.")]
    auxiliary = child_payload.get("plane_sequence_aux")
    if not isinstance(auxiliary, dict):
        raise RuntimeError("child lacks sequence plane auxiliary")
    probe = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in auxiliary["probe"].items()
    }

    replay_state = parent_payload.get("replay")
    if not isinstance(replay_state, dict):
        raise RuntimeError("parent lacks replay state")
    replay = QuantizedSequenceReplay(
        int(replay_state["capacity"]),
        int(replay_state["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    dense_observation, dense_action, _reward, _continuation = replay.sample(
        args.dense_batch_size,
        args.feature_context,
        device=device,
        rng=random.Random(args.dense_seed),
    )

    dataset_context = 16
    dataset = _load_event_dataset(
        args.event_dataset,
        context_length=dataset_context,
        event_threshold=args.event_threshold,
    )
    full_observation, full_action, _reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    split = _stratified_group_split(
        groups,
        phases,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    validation_events = np.flatnonzero(split == 1)
    targets = tuple(
        dataset_context - int(value)
        for value in sorted(args.negative_offsets_values, reverse=True)
    ) + (dataset_context,)
    validation_observation, validation_action, validation_target = _rolling_samples(
        full_observation,
        full_action,
        validation_events,
        targets=targets,
        context=args.feature_context,
    )
    validation_target_np = validation_target.float().cpu().numpy().reshape(
        len(validation_events), len(targets)
    )
    precision_results: dict[str, object] = {}
    for amp_dtype in ("none", "bfloat16"):
        parent_dense = _reference(
            parent,
            dense_observation,
            dense_action,
            microbatch_size=args.microbatch_size,
            amp_dtype=amp_dtype,
        )
        child_dense = _reference(
            child,
            dense_observation,
            dense_action,
            microbatch_size=args.microbatch_size,
            amp_dtype=amp_dtype,
        )
        parent_prediction = _predictions(
            parent,
            probe,
            validation_observation,
            validation_action,
            microbatch_size=args.microbatch_size,
            amp_dtype=amp_dtype,
            target_mean=float(auxiliary["target_mean"]),
            target_std=float(auxiliary["target_std"]),
        ).reshape(len(validation_events), len(targets))
        child_prediction = _predictions(
            child,
            probe,
            validation_observation,
            validation_action,
            microbatch_size=args.microbatch_size,
            amp_dtype=amp_dtype,
            target_mean=float(auxiliary["target_mean"]),
            target_std=float(auxiliary["target_std"]),
        ).reshape(len(validation_events), len(targets))
        precision_results[amp_dtype] = {
            "dense_drift": _drift(parent_dense, child_dense),
            "parent_validation": _progress_metrics(
                validation_target_np, parent_prediction
            ),
            "child_validation": _progress_metrics(
                validation_target_np, child_prediction
            ),
        }
    n637 = json.loads(args.n637_report.read_text(encoding="utf-8"))
    reproduced = precision_results["bfloat16"]["dense_drift"]
    reference = n637["dense_drift"]
    reproduction_error = max(
        abs(float(reproduced[name][metric]) - float(reference[name][metric]))
        for name in reference
        for metric in ("rmse", "max_abs")
    )
    source_hashes_after = {
        "parent": _sha256(args.parent_checkpoint),
        "child": _sha256(args.child_checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n637_report": _sha256(args.n637_report),
        "replay": _replay_hashes(args.replay_dir),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "precision_results": precision_results,
        "n637_bfloat16_dense_reproduction_max_abs_error": reproduction_error,
        "validation_events": int(len(validation_events)),
        "test_events": int((split == 2).sum()),
        "test_evaluations": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--child-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--n637-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-context", type=int, default=12)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=628)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=636001)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.negative_offsets_values = tuple(
        int(value) for value in args.negative_offsets.split(",")
    )
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
