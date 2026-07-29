#!/usr/bin/env python3
"""Read-only N711 posterior/prior/reward compatibility audit.

The audit rolls the frozen RSSM prior for fixed horizons from observed N681
validation states and compares it with the corresponding posterior trajectory.
It writes one JSON report and cannot train, save a checkpoint, create a native
environment, or receive the permanently consumed sealed-test path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256, _ridge_predict
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.audit_vq2_success_prefix_sequence_capacity import _forward_metres
from scripts.package_vq2_invariant_donor_readout import (
    N587_CHECKPOINT_SHA256,
    N710_FIT_SHA256,
    PACKAGE_SCHEMA,
    _fit_from_auxiliary,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _initial_state,
    _model_from_payload,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_imagination_readiness_audit_v1"
N711_CHECKPOINT_SHA256 = (
    "92be8898d8294477139f0706a38604d0e3b1bd953a128f9e30c8f9e4c2e168dd"
)
N711_REPORT_SHA256 = (
    "dec6720a51785134f957f91a38f6de9c03b38595d477eb2c2d5ab04c1b91ce03"
)
N713_REPORT_SHA256 = (
    "161d7e8da4b7c67b85552f09181134b4ed41f3b6c34120bb60d68dda790eb472"
)
VALIDATION_SHA256 = (
    "c8ded2bd9191a5c548b3bfc1f0a7142b66e0d75558ff80c883819075a5f48ad1"
)
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)
FIXED_HORIZONS = (1, 2, 4, 8, 16)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("summary requires finite nonempty values")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "median": float(np.median(values)),
    }


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("regression target and prediction must be nonempty and aligned")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise RuntimeError("regression inputs must be finite")
    error = prediction - target
    target_std = float(target.std())
    prediction_std = float(prediction.std())
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target_std > 0.0 and prediction_std > 0.0
        else 0.0
    )
    return {
        "count": int(target.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "correlation": correlation,
    }


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if target.shape != probability.shape or target.size == 0:
        raise ValueError("binary target and probability must be nonempty and aligned")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("binary probabilities must be finite and lie in [0,1]")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    bce = -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    return {
        "count": int(target.size),
        "bce": float(bce.mean()),
        "mae": float(np.abs(probability - target).mean()),
        "accuracy": float(((probability >= 0.5) == (target >= 0.5)).mean()),
        "target_positive_fraction": float((target >= 0.5).mean()),
        "probability_mean": float(probability.mean()),
    }


def _anchor_indices(
    sequence_length: int, horizon: int, *, burn_in: int, stride: int
) -> np.ndarray:
    """Observed start indices whose rollout endpoints remain pre-event."""

    if sequence_length <= 1 or horizon <= 0 or burn_in <= 0 or stride <= 0:
        raise ValueError("sequence, horizon, burn-in, and stride must be positive")
    # The final row is the gate-event row; keep progress diagnostics before it.
    maximum_start = sequence_length - 2 - horizon
    first = burn_in - 1
    if maximum_start < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, maximum_start + 1, stride, dtype=np.int64)


def _stack_state(rows: list[RSSMState]) -> RSSMState:
    return RSSMState(
        *(torch.stack([getattr(row, field) for row in rows], 1)
          for field in RSSMState._fields)
    )


def _select_state(state: RSSMState, indices: torch.Tensor) -> RSSMState:
    return RSSMState(
        *(value[:, indices].reshape(-1, *value.shape[2:]) for value in state)
    )


def _soft_features(state: RSSMState) -> torch.Tensor:
    return torch.cat(
        (state.deterministic.float(), state.logits.float().softmax(-1).flatten(-2)),
        -1,
    )


def _hard_actor_action(model: VQ2InformedDreamer, state: RSSMState) -> torch.Tensor:
    return model.deterministic_actor_action(model.actor_distribution(state.features))


def _load_validation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    required = {
        "mask", "tail", "action", "reward", "continuation",
        "initial_deterministic", "initial_stochastic_index", "initial_logits",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"validation dataset lacks arrays: {sorted(required-set(arrays))}")
    if arrays["mask"].shape[:2] != (64, 321):
        raise RuntimeError("N681 validation geometry must remain 64x321")
    return arrays


def _legal_action(
    arrays: dict[str, np.ndarray], events: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = arrays["mask"][events].astype(np.float32) / 255.0
    tail = arrays["tail"][events, :, : LEGAL_OBS_SIZE - MASK_SIZE].astype(np.float32)
    legal = torch.from_numpy(np.concatenate((mask, tail), -1)).to(device)
    action = torch.from_numpy(arrays["action"][events].astype(np.float32)).to(device)
    return legal, action


@torch.no_grad()
def _observe(
    model: VQ2InformedDreamer,
    legal: torch.Tensor,
    action: torch.Tensor,
    initial_state: RSSMState,
) -> tuple[RSSMState, torch.Tensor]:
    state = initial_state
    states: list[RSSMState] = []
    priors: list[torch.Tensor] = []
    for step in range(legal.shape[1]):
        state, prior = model.rssm.observe_step(
            state, action[:, step], legal[:, step], deterministic_latent=True
        )
        states.append(state)
        priors.append(prior)
    return _stack_state(states), torch.stack(priors, 1)


@torch.no_grad()
def _rollout(
    model: VQ2InformedDreamer,
    posterior: RSSMState,
    action: torch.Tensor,
    starts: torch.Tensor,
    horizon: int,
) -> RSSMState:
    state = _select_state(posterior, starts)
    batch = action.shape[0]
    for delta in range(1, horizon + 1):
        chosen = action[:, starts + delta].reshape(batch * len(starts), -1)
        state = model.rssm.imagine_step(state, chosen, deterministic_latent=True)
    return state


def _append(store: dict[str, list[np.ndarray]], name: str, value: torch.Tensor | np.ndarray) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    store.setdefault(name, []).append(np.asarray(value))


def _finish_rollout_metrics(
    store: dict[str, list[np.ndarray]],
    *,
    fit: dict[str, np.ndarray | float] | None,
) -> dict[str, object]:
    values = {name: np.concatenate(rows, 0) for name, rows in store.items()}
    action_error = values["open_action"] - values["posterior_action"]
    result: dict[str, object] = {
        "samples": int(len(values["kl"])),
        "posterior_to_open_prior_kl": _summary(values["kl"]),
        "categorical_argmax_agreement": _summary(values["agreement"]),
        "deterministic_state_rmse": _summary(values["deterministic_rmse"]),
        "actor_action_drift": {
            "rmse": float(np.sqrt(np.square(action_error).mean())),
            "max_abs": float(np.abs(action_error).max()),
            "mean_by_channel": action_error.mean(0).tolist(),
        },
        "posterior_reward": _regression_metrics(values["reward_target"], values["posterior_reward"]),
        "open_prior_reward": _regression_metrics(values["reward_target"], values["open_reward"]),
        "posterior_continuation": _binary_metrics(values["continuation_target"], values["posterior_continue"]),
        "open_prior_continuation": _binary_metrics(values["continuation_target"], values["open_continue"]),
    }
    if fit is not None:
        posterior_progress = _ridge_predict(fit, values["posterior_soft"])
        open_progress = _ridge_predict(fit, values["open_soft"])
        result["posterior_progress"] = _regression_metrics(
            values["progress_target"], posterior_progress
        )
        result["open_prior_progress"] = _regression_metrics(
            values["progress_target"], open_progress
        )
        result["open_vs_posterior_progress"] = _regression_metrics(
            posterior_progress, open_progress
        )
    return result


@torch.no_grad()
def _audit_model(
    model: VQ2InformedDreamer,
    arrays: dict[str, np.ndarray],
    *,
    horizons: tuple[int, ...],
    burn_in: int,
    stride: int,
    microbatch_size: int,
    device: torch.device,
    fit: dict[str, np.ndarray | float] | None,
) -> dict[str, object]:
    stores = {horizon: {} for horizon in horizons}
    event_stores = {horizon: {} for horizon in horizons}
    full_progress = _forward_metres(
        arrays["tail"][..., (LEGAL_OBS_SIZE - MASK_SIZE) + 3].astype(np.float32)
    )
    model.eval()
    for offset in range(0, 64, microbatch_size):
        events = np.arange(offset, min(64, offset + microbatch_size), dtype=np.int64)
        legal, action = _legal_action(arrays, events, device)
        posterior, _one_step_priors = _observe(
            model, legal, action, _initial_state(arrays, events, device=device)
        )
        for horizon in horizons:
            for event_endpoint, store in ((False, stores[horizon]), (True, event_stores[horizon])):
                starts_np = (
                    np.asarray([legal.shape[1] - 1 - horizon], dtype=np.int64)
                    if event_endpoint
                    else _anchor_indices(
                        legal.shape[1], horizon, burn_in=burn_in, stride=stride
                    )
                )
                starts = torch.from_numpy(starts_np).to(device)
                endpoints = starts + horizon
                open_state = _rollout(model, posterior, action, starts, horizon)
                endpoint_state = _select_state(posterior, endpoints)
                endpoint_action = action[:, endpoints].reshape(len(events) * len(starts), -1)
                posterior_probability = endpoint_state.logits.float().softmax(-1)
                posterior_log_probability = endpoint_state.logits.float().log_softmax(-1)
                open_log_probability = open_state.logits.float().log_softmax(-1)
                kl = (
                    posterior_probability
                    * (posterior_log_probability - open_log_probability)
                ).sum(-1).sum(-1)
                agreement = (
                    endpoint_state.logits.argmax(-1) == open_state.logits.argmax(-1)
                ).float().mean(-1)
                deterministic_rmse = (
                    endpoint_state.deterministic.float() - open_state.deterministic.float()
                ).square().mean(-1).sqrt()
                posterior_reward = model.predict_reward(endpoint_state.features, endpoint_action)
                open_reward = model.predict_reward(open_state.features, endpoint_action)
                posterior_continue = torch.sigmoid(
                    model.continue_predictor(endpoint_state.features).squeeze(-1)
                )
                open_continue = torch.sigmoid(
                    model.continue_predictor(open_state.features).squeeze(-1)
                )
                event_grid = events[:, None]
                endpoint_grid = endpoints.detach().cpu().numpy()[None, :]
                _append(store, "kl", kl)
                _append(store, "agreement", agreement)
                _append(store, "deterministic_rmse", deterministic_rmse)
                _append(store, "posterior_action", _hard_actor_action(model, endpoint_state))
                _append(store, "open_action", _hard_actor_action(model, open_state))
                _append(store, "posterior_reward", posterior_reward)
                _append(store, "open_reward", open_reward)
                _append(store, "posterior_continue", posterior_continue)
                _append(store, "open_continue", open_continue)
                _append(store, "reward_target", arrays["reward"][event_grid, endpoint_grid].reshape(-1))
                _append(store, "continuation_target", arrays["continuation"][event_grid, endpoint_grid].reshape(-1))
                if fit is not None and not event_endpoint:
                    _append(store, "posterior_soft", _soft_features(endpoint_state))
                    _append(store, "open_soft", _soft_features(open_state))
                    _append(store, "progress_target", full_progress[event_grid, endpoint_grid].reshape(-1))
    return {
        str(horizon): {
            "pre_event": _finish_rollout_metrics(stores[horizon], fit=fit),
            "event_endpoint": _finish_rollout_metrics(event_stores[horizon], fit=None),
        }
        for horizon in horizons
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "n711_checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "n713_report": _sha256(args.n713_report),
        "n587_checkpoint": _sha256(args.parent_checkpoint),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "n711_checkpoint": N711_CHECKPOINT_SHA256,
        "n711_report": N711_REPORT_SHA256,
        "n713_report": N713_REPORT_SHA256,
        "n587_checkpoint": N587_CHECKPOINT_SHA256,
        "validation_dataset": VALIDATION_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("N714 source hash mismatch")
    n711_report = json.loads(args.n711_report.read_text(encoding="utf-8"))
    n713_report = json.loads(args.n713_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n711_report.get("contract") != PACKAGE_SCHEMA
        or n711_report.get("package_admitted") is not True
        or n713_report.get("contract") != "vq2_n711_paired_native_gate1_screen_v1"
        or n713_report.get("passed") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("N714 predecessor contract mismatch")
    arrays = _load_validation(args.validation_dataset)
    n711_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n587_payload = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    auxiliary = n711_payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("N711 fixed ridge auxiliary is missing")
    fit = _fit_from_auxiliary(auxiliary)
    if _fit_sha256(fit) != N710_FIT_SHA256:
        raise RuntimeError("N711 fixed ridge hash mismatch")
    n711_model = _model_from_payload(n711_payload, device)
    n711_state = {name: value.detach().cpu().clone() for name, value in n711_model.state_dict().items()}
    n711_metrics = _audit_model(
        n711_model, arrays, horizons=FIXED_HORIZONS, burn_in=args.burn_in,
        stride=args.anchor_stride, microbatch_size=args.microbatch_size,
        device=device, fit=fit,
    )
    n711_changes = [
        name for name, value in n711_model.state_dict().items()
        if not torch.equal(value.detach().cpu(), n711_state[name])
    ]
    del n711_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n587_model = _model_from_payload(n587_payload, device)
    n587_state = {name: value.detach().cpu().clone() for name, value in n587_model.state_dict().items()}
    n587_metrics = _audit_model(
        n587_model, arrays, horizons=FIXED_HORIZONS, burn_in=args.burn_in,
        stride=args.anchor_stride, microbatch_size=args.microbatch_size,
        device=device, fit=None,
    )
    n587_changes = [
        name for name, value in n587_model.state_dict().items()
        if not torch.equal(value.detach().cpu(), n587_state[name])
    ]
    source_after = {
        "executable": _sha256(Path(__file__)),
        "n711_checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "n713_report": _sha256(args.n713_report),
        "n587_checkpoint": _sha256(args.parent_checkpoint),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "fixed_horizons_exact": FIXED_HORIZONS == (1, 2, 4, 8, 16),
        "validation_geometry_exact": arrays["mask"].shape[:2] == (64, 321),
        "n711_model_tensors_exact": n711_changes == [],
        "n587_model_tensors_exact": n587_changes == [],
        "fixed_ridge_hash_exact": _fit_sha256(fit) == N710_FIT_SHA256,
        "no_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "horizons": list(FIXED_HORIZONS),
        "burn_in": args.burn_in,
        "anchor_stride": args.anchor_stride,
        "n711": n711_metrics,
        "n587_paired_parent": n587_metrics,
        "n711_model_tensor_changes": n711_changes,
        "n587_model_tensor_changes": n587_changes,
        "optimizer_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "prior_updates": 0,
        "reward_updates": 0,
        "checkpoint_written": 0,
        "train_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if not report["process_passed"]:
        raise RuntimeError("N714 process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n711-report", type=Path, required=True)
    parser.add_argument("--n713-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("N714 refuses to overwrite its report")
    if args.burn_in != 32 or args.anchor_stride != 4:
        parser.error("N714 burn-in and anchor stride are fixed")
    if args.microbatch_size <= 0:
        parser.error("microbatch size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
