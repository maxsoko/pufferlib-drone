#!/usr/bin/env python3
"""Apply one optimizer-resumed progress-prior step after N720."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _project_prior_parameters,
    _sampling_schedule,
)
from scripts.continue_vq2_prior_progress_offline import (
    MATERIALIZATION_SHA256,
    TRAIN_SHA256,
    _losses,
    _metrics_to_float,
    _posterior_actions,
    _ridge_tensors,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    _fit_from_auxiliary,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import _finite_clip_grad_norm, _save_checkpoint


CONTINUATION_SCHEMA = "vq2_prior_frozen_progress_resume_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N720_REPORT_SHA256 = (
    "8cae6c61eaa96cc4e109d4fc60b500fc075ada33c2b896336f37acbd1f9b8fa8"
)
N721_REPORT_SHA256 = (
    "b9ed4b751f4b14e9e1c6092200ecd9d05600b6344aa8d03036469b6d10049c29"
)


def continue_prior(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n720_report": _sha256(args.n720_report),
        "n721_report": _sha256(args.n721_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n720_report": N720_REPORT_SHA256,
        "n721_report": N721_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("resumed progress-prior source hash mismatch")
    n720 = json.loads(args.n720_report.read_text(encoding="utf-8"))
    n721 = json.loads(args.n721_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n720.get("process_passed") is not True
        or n720.get("updates") != 1
        or n721.get("process_passed") is not True
        or n721.get("improved_all_horizons") is not True
        or n721.get("prior_admitted") is not False
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("resumed progress-prior predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    optimizer_state = payload.get("offline_progress_prior_optimizer")
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(optimizer_state, dict) or not isinstance(auxiliary, dict):
        raise RuntimeError("N720 optimizer or ridge state is missing")
    if _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("N720 frozen ridge hash mismatch")
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
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_prior],
        lr=3e-7, weight_decay=0.0,
    )
    optimizer.load_state_dict(optimizer_state)
    if any(group["lr"] != 3e-7 or group["weight_decay"] != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("N720 optimizer hyperparameters changed")
    reference = _posterior_reference(
        model, arrays, crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=719,
        batch_size=16, updates=2,
    )
    first_pairs, second_pairs = schedule[0], schedule[1]
    if len({tuple(row) for row in schedule.reshape(-1, 2)}) != 32:
        raise RuntimeError("first two progress-prior batches must be distinct")
    event, start = second_pairs[:, 0], second_pairs[:, 1]
    legal, action = _legal_action_batch(
        arrays, event, start, sequence_length=128, device=device
    )
    initial = _pair_initial_state(arrays, reference, event, start, device=device)
    ridge = _ridge_tensors(auxiliary, device)
    with torch.no_grad():
        _before_total, before_t = _losses(model, legal, action, initial, ridge, burn_in=32)
        before_actions = _posterior_actions(model, legal, action, initial).cpu()
    before = _metrics_to_float(before_t)
    optimizer.zero_grad(set_to_none=True)
    _combined, metrics_t = _losses(model, legal, action, initial, ridge, burn_in=32)
    objective = metrics_t["level"]
    objective.backward()
    gradient = _finite_clip_grad_norm(
        [parameter for _name, parameter in named_prior], 1.0,
        label="resumed-progress-RSSM-prior",
    )
    optimizer.step()
    projected = _project_prior_parameters(named_prior, model_before, 0.001)
    with torch.no_grad():
        _after_total, after_t = _losses(model, legal, action, initial, ridge, burn_in=32)
        after_actions = _posterior_actions(model, legal, action, initial).cpu()
    after = _metrics_to_float(after_t)
    child_state = model.state_dict()
    changed = sorted(
        name for name, value in child_state.items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    )
    forbidden = [name for name in changed if not name.startswith("rssm.prior.")]
    maximum_delta = max(
        float((child_state[name].detach().cpu() - model_before[name]).abs().max())
        for name in changed
    )
    action_difference = after_actions - before_actions
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n720_report": _sha256(args.n720_report),
        "n721_report": _sha256(args.n721_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "optimizer_resumed": bool(optimizer.state),
        "second_batch_distinct": not any(
            bool((first_pairs == row).all(1).any()) for row in second_pairs
        ),
        "only_prior_tensors_changed": bool(changed) and forbidden == [],
        "posterior_actor_actions_bit_exact": bool(torch.equal(before_actions, after_actions)),
        "batch_progress_mae_decreased": after["progress_mae"] < before["progress_mae"],
        "batch_kl_preserved": after["kl"] <= before["kl"] + 0.001,
        "finite_metrics_and_gradient": all(math.isfinite(value) for value in [*before.values(), *after.values(), float(gradient.cpu())]),
        "parameter_delta_bounded": maximum_delta <= 0.001,
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "additional_updates": 1,
        "cumulative_progress_prior_updates": 2,
        "objective": "level_only",
        "learning_rate": 3e-7,
        "before_second_batch": before,
        "after_second_batch": after,
        "last_gradient_norm": float(gradient.cpu()),
        "changed_model_keys": changed,
        "forbidden_model_keys": forbidden,
        "maximum_parameter_delta_from_n720": maximum_delta,
        "projected_parameter_values": projected,
        "posterior_actor_action_rmse": float(action_difference.square().mean().sqrt()),
        "posterior_actor_action_max_abs": float(action_difference.abs().max()),
        "optimizer_steps": 1,
        "prior_updates": 1,
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
        raise RuntimeError(f"N722 gates failed: {[k for k,v in process_gates.items() if not v]}")
    output = dict(payload)
    output["model"] = child_state
    output["offline_progress_prior_optimizer"] = optimizer.state_dict()
    output["offline_progress_prior_updates"] = 2
    output["offline_progress_prior_parent_sha256"] = source_before["checkpoint"]
    output["metrics"] = report
    _save_checkpoint(args.output_checkpoint, output)
    report["checkpoint_written"] = 1
    report["output_checkpoint"] = str(args.output_checkpoint)
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n720-report", type=Path, required=True)
    parser.add_argument("--n721-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("N722 refuses to overwrite outputs")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_prior(parse_args()), indent=2, sort_keys=True))
