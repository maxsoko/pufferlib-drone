#!/usr/bin/env python3
"""Filter exact-history VQ2 event windows by deterministic RSSM replay.

This is a read-only checkpoint audit. It may write a filtered training corpus,
but it never creates a native environment, updates a model, or saves probe
weights. Selection depends only on replay consistency, never privileged target
quality or task outcome beyond the source collector's event contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from scripts.collect_vq2_gate_event_windows import (
    EVENT_WINDOW_EXACT_HISTORY_SCHEMA,
    _sha256_bytes,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


FILTER_SCHEMA = "vq2_actor_frozen_gate_event_windows_replay_filtered_v1"


def _select_replay_exact_indices(
    deterministic_error: np.ndarray,
    logits_error: np.ndarray,
    stochastic_mismatch: np.ndarray,
    *,
    target_events: int,
    maximum_deterministic_error: float,
    maximum_logits_error: float,
) -> tuple[np.ndarray, np.ndarray]:
    deterministic_error = np.asarray(deterministic_error, dtype=np.float64)
    logits_error = np.asarray(logits_error, dtype=np.float64)
    stochastic_mismatch = np.asarray(stochastic_mismatch, dtype=np.int64)
    if not (
        deterministic_error.shape
        == logits_error.shape
        == stochastic_mismatch.shape
    ) or deterministic_error.ndim != 1:
        raise ValueError("per-event replay metrics must be aligned vectors")
    eligible = (
        (deterministic_error <= maximum_deterministic_error)
        & (logits_error <= maximum_logits_error)
        & (stochastic_mismatch == 0)
    )
    indices = np.nonzero(eligible)[0]
    if len(indices) < target_events:
        raise RuntimeError(
            f"only {len(indices)} replay-exact events, expected {target_events}"
        )
    return indices[:target_events], eligible


def _filter_event_arrays(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("event corpus is empty")
    event_count = int(next(iter(arrays.values())).shape[0])
    for name, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != event_count:
            raise ValueError(f"event array {name} does not align on axis zero")
    return {name: value[indices].copy() for name, value in arrays.items()}


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("contract") != EVENT_WINDOW_EXACT_HISTORY_SCHEMA:
        raise RuntimeError("source corpus is not exact-history schema v4")
    if source_report.get("dataset_sha256") != source_hashes_before["dataset"]:
        raise RuntimeError("source report dataset hash mismatch")
    if source_report.get("checkpoint_sha256") != source_hashes_before["checkpoint"]:
        raise RuntimeError("source report checkpoint hash mismatch")
    if source_report.get("boundary_storage_dtype") != "float32":
        raise RuntimeError("source boundary storage is not float32")
    if source_report.get("window_storage_dtype") != "float32":
        raise RuntimeError("source window storage is not float32")

    with np.load(args.dataset) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "mask",
        "tail",
        "action",
        "initial_deterministic",
        "initial_stochastic_index",
        "initial_logits",
        "preevent_deterministic",
        "preevent_stochastic_index",
        "preevent_logits",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise RuntimeError(f"source corpus lacks arrays: {missing}")
    event_count = int(arrays["mask"].shape[0])
    if event_count < args.target_events:
        raise RuntimeError("source corpus has fewer events than requested")

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(
            training_args.get("actor_distribution_mode", "legacy_tanh_normal")
        ),
        distributional_reward=bool(
            training_args.get("distributional_reward", False)
        ),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    source_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mask = torch.from_numpy(arrays["mask"].astype(np.float32)).to(device) / 255.0
    tail = torch.from_numpy(arrays["tail"].astype(np.float32)).to(device)
    observation = torch.cat((mask, tail), -1)
    action = torch.from_numpy(arrays["action"].astype(np.float32)).to(device)
    initial_index = torch.from_numpy(
        arrays["initial_stochastic_index"].astype(np.int64)
    ).to(device)
    initial = RSSMState(
        torch.from_numpy(arrays["initial_deterministic"].astype(np.float32)).to(
            device
        ),
        F.one_hot(initial_index, model.rssm.stochastic_classes).to(torch.float32),
        torch.from_numpy(arrays["initial_logits"].astype(np.float32)).to(device),
    )
    expected_index = torch.from_numpy(
        arrays["preevent_stochastic_index"].astype(np.int64)
    ).to(device)
    expected = RSSMState(
        torch.from_numpy(arrays["preevent_deterministic"].astype(np.float32)).to(
            device
        ),
        F.one_hot(expected_index, model.rssm.stochastic_classes).to(torch.float32),
        torch.from_numpy(arrays["preevent_logits"].astype(np.float32)).to(device),
    )
    with torch.no_grad():
        replayed_sequence = model.observe_sequence(
            observation[:, :-1],
            action[:, :-1],
            initial_state=initial,
            deterministic_latent=True,
        ).posterior
        replayed = RSSMState(
            replayed_sequence.deterministic[:, -1],
            replayed_sequence.stochastic[:, -1],
            replayed_sequence.logits[:, -1],
        )
        expected_action = model.deterministic_actor_action(
            model.actor_distribution(expected.features)
        )
        replayed_action = model.deterministic_actor_action(
            model.actor_distribution(replayed.features)
        )

    deterministic_error = (
        replayed.deterministic - expected.deterministic
    ).abs().amax(-1).cpu().numpy()
    logits_error = (
        replayed.logits - expected.logits
    ).abs().flatten(1).amax(-1).cpu().numpy()
    stochastic_mismatch = (
        replayed.stochastic.argmax(-1) != expected_index
    ).sum(-1).cpu().numpy()
    action_error = (replayed_action - expected_action).abs().amax(-1).cpu().numpy()
    selected, eligible = _select_replay_exact_indices(
        deterministic_error,
        logits_error,
        stochastic_mismatch,
        target_events=args.target_events,
        maximum_deterministic_error=args.maximum_deterministic_error,
        maximum_logits_error=args.maximum_logits_error,
    )
    filtered = _filter_event_arrays(arrays, selected)

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary_dataset = args.output_dataset.with_suffix(
        args.output_dataset.suffix + ".tmp"
    )
    with temporary_dataset.open("wb") as stream:
        np.savez_compressed(stream, **filtered)
    temporary_dataset.replace(args.output_dataset)

    source_hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_model_state[name])
    ]
    selected_bool = np.zeros(event_count, dtype=bool)
    selected_bool[selected] = True
    report: dict[str, object] = {
        "contract": FILTER_SCHEMA,
        "source_contract": EVENT_WINDOW_EXACT_HISTORY_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "source_events": event_count,
        "eligible_events": int(eligible.sum()),
        "rejected_events": int((~eligible).sum()),
        "target_events": args.target_events,
        "selected_events": int(len(selected)),
        "selected_indices": selected.tolist(),
        "maximum_deterministic_error": args.maximum_deterministic_error,
        "maximum_logits_error": args.maximum_logits_error,
        "per_event_deterministic_max_abs_error": deterministic_error.tolist(),
        "per_event_logits_max_abs_error": logits_error.tolist(),
        "per_event_stochastic_index_mismatch_count": stochastic_mismatch.tolist(),
        "per_event_action_max_abs_error": action_error.tolist(),
        "selected_deterministic_max_abs_error": float(
            deterministic_error[selected_bool].max()
        ),
        "selected_logits_max_abs_error": float(logits_error[selected_bool].max()),
        "selected_stochastic_index_mismatch_count": int(
            stochastic_mismatch[selected_bool].sum()
        ),
        "selected_action_max_abs_error": float(action_error[selected_bool].max()),
        "output_dataset": str(args.output_dataset),
        "output_dataset_sha256": _sha256_bytes(args.output_dataset),
        "output_dataset_bytes": args.output_dataset.stat().st_size,
        "model_tensor_changes": model_changes,
        "process_passed": bool(
            source_hashes_after == source_hashes_before
            and not model_changes
            and len(selected) == args.target_events
        ),
        "optimizer_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-events", type=int, required=True)
    parser.add_argument("--maximum-deterministic-error", type=float, default=1e-5)
    parser.add_argument("--maximum-logits-error", type=float, default=1e-5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.target_events <= 0:
        parser.error("--target-events must be positive")
    if args.maximum_deterministic_error < 0.0:
        parser.error("--maximum-deterministic-error cannot be negative")
    if args.maximum_logits_error < 0.0:
        parser.error("--maximum-logits-error cannot be negative")
    if args.output_dataset.exists() or args.report.exists():
        parser.error("filter refuses to overwrite output dataset or report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
