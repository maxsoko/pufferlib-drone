#!/usr/bin/env python3
"""Read-only virtual surface for admission-matched open-loop prior progress."""

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

from pufferlib.vq2_dreamer import RSSMState
from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256
from scripts.audit_vq2_prior_progress_virtual_surface import _restore, _virtual_adam_step
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _sampling_schedule,
)
from scripts.continue_vq2_prior_progress_offline import (
    MATERIALIZATION_SHA256,
    TRAIN_SHA256,
    _ridge_predict_torch,
    _ridge_tensors,
    _soft_feature,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_prior_multihorizon_progress_virtual_surface_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N721_REPORT_SHA256 = (
    "b9ed4b751f4b14e9e1c6092200ecd9d05600b6344aa8d03036469b6d10049c29"
)
N723_REPORT_SHA256 = (
    "a176d5fa4305491ff1961c91aef06a40491ad4658893a59065d52aa9a72916c2"
)
HORIZONS = (1, 2, 4, 8, 16)
LEARNING_RATES = (1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7)


def _anchors(length: int, horizon: int, *, burn_in: int, stride: int) -> torch.Tensor:
    if min(length, horizon, burn_in, stride) <= 0:
        raise ValueError("anchor arguments must be positive")
    return torch.arange(burn_in - 1, length - horizon, stride, dtype=torch.long)


def _stack_state(rows: list[RSSMState]) -> RSSMState:
    return RSSMState(
        *(torch.stack([getattr(row, field) for row in rows], 1) for field in RSSMState._fields)
    )


def _select_state(state: RSSMState, indices: torch.Tensor) -> RSSMState:
    return RSSMState(
        *(value[:, indices].reshape(-1, *value.shape[2:]) for value in state)
    )


@torch.no_grad()
def _observe(model, legal: torch.Tensor, action: torch.Tensor, initial: RSSMState) -> RSSMState:
    state = initial
    rows: list[RSSMState] = []
    for step in range(legal.shape[1]):
        state, _prior = model.rssm.observe_step(
            state, action[:, step], legal[:, step], deterministic_latent=True
        )
        rows.append(state)
    return _stack_state(rows)


def _rollout(
    model,
    posterior: RSSMState,
    action: torch.Tensor,
    anchors: torch.Tensor,
    horizon: int,
) -> RSSMState:
    state = _select_state(posterior, anchors).detached()
    batch = action.shape[0]
    for delta in range(1, horizon + 1):
        chosen = action[:, anchors + delta].reshape(batch * len(anchors), -1)
        state = model.rssm.imagine_step(state, chosen, deterministic_latent=True)
    return state


def _surface_metrics(
    model,
    posterior: RSSMState,
    action: torch.Tensor,
    ridge: dict[str, torch.Tensor],
    *,
    burn_in: int,
    stride: int,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    losses: list[torch.Tensor] = []
    rows: dict[str, dict[str, torch.Tensor]] = {}
    for horizon in HORIZONS:
        anchors = _anchors(
            action.shape[1], horizon, burn_in=burn_in, stride=stride
        ).to(action.device)
        endpoints = anchors + horizon
        open_state = _rollout(model, posterior, action, anchors, horizon)
        endpoint = _select_state(posterior, endpoints).detached()
        open_progress = _ridge_predict_torch(
            _soft_feature(open_state.deterministic, open_state.logits), ridge
        )
        posterior_progress = _ridge_predict_torch(
            _soft_feature(endpoint.deterministic, endpoint.logits), ridge
        ).detach()
        progress_error = open_progress - posterior_progress
        loss = F.smooth_l1_loss(open_progress, posterior_progress, beta=0.10)
        losses.append(loss)
        kl = model._categorical_kl(endpoint.logits.detach(), open_state.logits).mean()
        open_action = model.deterministic_actor_action(
            model.actor_distribution(open_state.features)
        )
        posterior_action = model.deterministic_actor_action(
            model.actor_distribution(endpoint.features)
        ).detach()
        action_error = open_action - posterior_action
        rows[str(horizon)] = {
            "loss": loss,
            "progress_mae": progress_error.abs().mean(),
            "progress_rmse": progress_error.square().mean().sqrt(),
            "progress_bias": progress_error.mean(),
            "kl": kl,
            "action_rmse": action_error.square().mean().sqrt(),
            "action_max_abs": action_error.abs().max(),
        }
    return torch.stack(losses).mean(), rows


def _float_rows(rows: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return {
        horizon: {name: float(value.detach().cpu()) for name, value in metrics.items()}
        for horizon, metrics in rows.items()
    }


def _eligible(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> bool:
    return all(
        after[str(h)]["progress_mae"] < before[str(h)]["progress_mae"]
        and after[str(h)]["kl"] <= before[str(h)]["kl"] + 0.001
        and after[str(h)]["action_rmse"] <= 0.001
        and after[str(h)]["action_max_abs"] <= 0.01
        for h in HORIZONS
    )


def _selection_key(row: dict[str, object]) -> tuple[float, ...]:
    after = row["after"]
    return (
        -max(float(after[str(h)]["progress_mae"]) for h in HORIZONS),
        -sum(float(after[str(h)]["progress_mae"]) for h in HORIZONS),
        -float(row["learning_rate"]),
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n721_report": _sha256(args.n721_report),
        "n723_report": _sha256(args.n723_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n721_report": N721_REPORT_SHA256,
        "n723_report": N723_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("multi-horizon surface source hash mismatch")
    n721 = json.loads(args.n721_report.read_text(encoding="utf-8"))
    n723 = json.loads(args.n723_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n721.get("process_passed") is not True
        or n721.get("prior_admitted") is not False
        or n723.get("process_passed") is not True
        or n723.get("selection_informative") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("multi-horizon predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("multi-horizon fixed ridge mismatch")
    model = _model_from_payload(payload, device)
    model_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_prior = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith("rssm.prior.")
    ]
    for _name, parameter in named_prior:
        parameter.requires_grad_(True)
    reference = _posterior_reference(
        model, arrays, crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    batch = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=724,
        batch_size=16, updates=1,
    )[0]
    event, start = batch[:, 0], batch[:, 1]
    legal, action = _legal_action_batch(
        arrays, event, start, sequence_length=128, device=device
    )
    initial = _pair_initial_state(arrays, reference, event, start, device=device)
    posterior = _observe(model, legal, action, initial)
    ridge = _ridge_tensors(auxiliary, device)
    model.zero_grad(set_to_none=True)
    objective, before_t = _surface_metrics(
        model, posterior, action, ridge, burn_in=32, stride=8
    )
    before = _float_rows(before_t)
    objective.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _name, parameter in named_prior], 1.0
    )
    gradients = {name: parameter.grad.detach().clone() for name, parameter in named_prior}
    rows: list[dict[str, object]] = []
    for learning_rate in LEARNING_RATES:
        _virtual_adam_step(named_prior, model_before, gradients, learning_rate)
        with torch.no_grad():
            after_objective, after_t = _surface_metrics(
                model, posterior, action, ridge, burn_in=32, stride=8
            )
        after = _float_rows(after_t)
        rows.append(
            {
                "learning_rate": learning_rate,
                "before": before,
                "after": after,
                "before_objective": float(objective.detach().cpu()),
                "after_objective": float(after_objective.detach().cpu()),
                "eligible": _eligible(before, after),
            }
        )
        _restore(named_prior, model_before)
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=_selection_key) if eligible_rows else None
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n721_report": _sha256(args.n721_report),
        "n723_report": _sha256(args.n723_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_restored_exact": changed == [],
        "fixed_horizons_exact": HORIZONS == (1, 2, 4, 8, 16),
        "six_virtual_rates": len(rows) == 6,
        "finite_surface": all(
            math.isfinite(float(value))
            for row in rows
            for section in ("before", "after")
            for horizon in row[section].values()
            for value in horizon.values()
        ) and math.isfinite(float(gradient_norm.cpu())),
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "aggregate_gradient_norm_preclip": float(gradient_norm.cpu()),
        "surface": rows,
        "eligible_candidates": len(eligible_rows),
        "selected_candidate": selected,
        "selection_informative": selected is not None,
        "model_tensor_changes_after_restore": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "prior_updates": 0,
        "actor_updates": 0,
        "reward_updates": 0,
        "validation_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if not report["process_passed"]:
        raise RuntimeError("multi-horizon surface process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n721-report", type=Path, required=True)
    parser.add_argument("--n723-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("multi-horizon surface refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
