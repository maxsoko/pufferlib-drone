#!/usr/bin/env python3
"""Apply one source-locked N724 multi-horizon prior update."""

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
from scripts.audit_vq2_prior_multihorizon_surface import (
    AUDIT_SCHEMA,
    HORIZONS,
    _eligible,
    _float_rows,
    _observe,
    _restore,
    _surface_metrics,
    _virtual_adam_step,
)
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _sampling_schedule,
    _sampling_summary,
)
from scripts.continue_vq2_prior_progress_offline import _ridge_tensors
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import _finite_clip_grad_norm, _save_checkpoint


CONTINUATION_SCHEMA = "vq2_prior_multihorizon_progress_single_update_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N724_REPORT_SHA256 = "4b3f03a3efcdebc2cc681538f0239648f8b84a06fefa2f8eb9a580d857fc0f80"
N724_PREREGISTRATION_SHA256 = (
    "96780e64bd18c45b00d12dfef718b865128983beb04a0aba9e0d25fa868d009c"
)
TRAIN_SHA256 = "6bb788c0ae27264fa8a68b75500228d9be3269e80bfa454c8d01f8bb654d91ec"
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)
LEARNING_RATE = 1e-8


def _metric_parity(
    actual: dict[str, dict[str, float]], expected: dict[str, object]
) -> float:
    return max(
        abs(float(actual[str(horizon)][name]) - float(expected[str(horizon)][name]))
        for horizon in HORIZONS
        for name in (
            "loss",
            "progress_mae",
            "progress_rmse",
            "progress_bias",
            "kl",
            "action_rmse",
            "action_max_abs",
        )
    )


def _posterior_actions(model, posterior) -> torch.Tensor:
    return model.deterministic_actor_action(
        model.actor_distribution(posterior.features)
    ).detach().cpu()


def continue_prior(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("multi-horizon continuation must not overwrite its parent")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n724_report": _sha256(args.n724_report),
        "n724_preregistration": _sha256(args.n724_preregistration),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n724_report": N724_REPORT_SHA256,
        "n724_preregistration": N724_PREREGISTRATION_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("multi-horizon continuation source hash mismatch")
    n724 = json.loads(args.n724_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    selected = n724.get("selected_candidate")
    if (
        n724.get("contract") != AUDIT_SCHEMA
        or n724.get("process_passed") is not True
        or n724.get("selection_informative") is not True
        or not isinstance(selected, dict)
        or selected.get("learning_rate") != LEARNING_RATE
        or selected.get("eligible") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("multi-horizon continuation predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("multi-horizon fixed ridge mismatch")
    model = _model_from_payload(payload, device)
    model_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_prior = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("rssm.prior.")
    ]
    for _name, parameter in named_prior:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_prior],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    reference = _posterior_reference(
        model,
        arrays,
        crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=724, batch_size=16, updates=1
    )
    event, start = schedule[0][:, 0], schedule[0][:, 1]
    legal, action = _legal_action_batch(
        arrays, event, start, sequence_length=128, device=device
    )
    initial = _pair_initial_state(arrays, reference, event, start, device=device)
    posterior = _observe(model, legal, action, initial)
    posterior_actions_before = _posterior_actions(model, posterior)
    ridge = _ridge_tensors(auxiliary, device)
    optimizer.zero_grad(set_to_none=True)
    objective, before_t = _surface_metrics(
        model, posterior, action, ridge, burn_in=32, stride=8
    )
    before = _float_rows(before_t)
    objective.backward()
    gradient_norm = _finite_clip_grad_norm(
        [parameter for _name, parameter in named_prior],
        1.0,
        label="multi-horizon-progress-RSSM-prior",
    )
    gradients = {name: parameter.grad.detach().clone() for name, parameter in named_prior}

    _virtual_adam_step(named_prior, model_before, gradients, LEARNING_RATE)
    with torch.no_grad():
        virtual_objective_t, virtual_after_t = _surface_metrics(
            model, posterior, action, ridge, burn_in=32, stride=8
        )
    virtual_after = _float_rows(virtual_after_t)
    virtual_state = {
        name: parameter.detach().cpu().clone() for name, parameter in named_prior
    }
    _restore(named_prior, model_before)

    optimizer.step()
    with torch.no_grad():
        final_objective_t, final_after_t = _surface_metrics(
            model, posterior, action, ridge, burn_in=32, stride=8
        )
    final_after = _float_rows(final_after_t)
    posterior_actions_after = _posterior_actions(model, posterior)
    child_state = model.state_dict()
    changed = sorted(
        name
        for name, value in child_state.items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    )
    forbidden = [name for name in changed if not name.startswith("rssm.prior.")]
    maximum_delta = max(
        float((child_state[name].detach().cpu() - model_before[name]).abs().max())
        for name in changed
    )
    tensor_parity = max(
        float((parameter.detach().cpu() - virtual_state[name]).abs().max())
        for name, parameter in named_prior
    )
    metric_parity = _metric_parity(final_after, selected["after"])
    virtual_metric_parity = _metric_parity(virtual_after, selected["after"])
    action_difference = posterior_actions_after - posterior_actions_before
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n724_report": _sha256(args.n724_report),
        "n724_preregistration": _sha256(args.n724_preregistration),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    sampling = _sampling_summary(schedule, len(arrays["mask"]))
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "only_prior_tensors_changed": bool(changed) and forbidden == [],
        "posterior_actor_actions_bit_exact": torch.equal(
            posterior_actions_before, posterior_actions_after
        ),
        "every_horizon_progress_mae_decreased": _eligible(before, final_after),
        "finite_loss_and_gradient": all(
            math.isfinite(value)
            for value in (
                float(objective.detach().cpu()),
                float(final_objective_t.detach().cpu()),
                float(gradient_norm.detach().cpu()),
            )
        ),
        "maximum_parameter_delta_bounded": maximum_delta <= 1e-6,
        "fixed_ridge_unchanged": _fit_sha256(_fit_from_auxiliary(auxiliary))
        == N710_FIT_SHA256,
        "sampling_exact": sampling["unique_pairs"] == 16,
        "n724_virtual_metric_parity": virtual_metric_parity <= 1e-6,
        "real_virtual_metric_parity": metric_parity <= 1e-6,
        "real_virtual_tensor_parity": tensor_parity <= 1e-7,
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "learning_rate": LEARNING_RATE,
        "horizons": list(HORIZONS),
        "sampling": sampling,
        "initial": before,
        "virtual_after": virtual_after,
        "final": final_after,
        "initial_objective": float(objective.detach().cpu()),
        "virtual_after_objective": float(virtual_objective_t.detach().cpu()),
        "final_objective": float(final_objective_t.detach().cpu()),
        "gradient_norm_preclip": float(gradient_norm.detach().cpu()),
        "n724_virtual_metric_parity_max_abs_error": virtual_metric_parity,
        "real_virtual_metric_parity_max_abs_error": metric_parity,
        "real_virtual_tensor_parity_max_abs_error": tensor_parity,
        "changed_model_keys": changed,
        "forbidden_model_keys": forbidden,
        "maximum_parameter_delta": maximum_delta,
        "posterior_actor_action_rmse": float(action_difference.square().mean().sqrt()),
        "posterior_actor_action_max_abs": float(action_difference.abs().max()),
        "optimizer_steps": 1,
        "prior_updates": 1,
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "representation_updates": 0,
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
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    if not report["process_passed"]:
        failed = [name for name, passed in process_gates.items() if not passed]
        raise RuntimeError(f"multi-horizon continuation gates failed: {failed}")
    output = dict(payload)
    output["model"] = child_state
    output["offline_multihorizon_prior_optimizer"] = optimizer.state_dict()
    output["offline_multihorizon_prior_updates"] = 1
    output["offline_multihorizon_prior_parent_sha256"] = source_before["checkpoint"]
    output["metrics"] = report
    _save_checkpoint(args.output_checkpoint, output)
    report["checkpoint_written"] = 1
    report["output_checkpoint"] = str(args.output_checkpoint)
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n724-report", type=Path, required=True)
    parser.add_argument("--n724-preregistration", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("multi-horizon continuation refuses to overwrite outputs")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_prior(parse_args()), indent=2, sort_keys=True))
