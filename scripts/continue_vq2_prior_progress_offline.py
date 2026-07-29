#!/usr/bin/env python3
"""Fine-tune only N716's prior along the frozen N710 progress direction."""

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
from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256
from scripts.audit_vq2_probe_gradient_alignment import _sha256
from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _load_train,
    _pair_initial_state,
    _posterior_reference,
    _project_prior_parameters,
    _sampling_schedule,
    _sampling_summary,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    PACKAGE_SCHEMA,
    _fit_from_auxiliary,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import _finite_clip_grad_norm, _save_checkpoint


CONTINUATION_SCHEMA = "vq2_prior_frozen_progress_distillation_v1"
N716_CHECKPOINT_SHA256 = (
    "0a0a3af0fe9846c1a05001c1d8fabed8ec9b4f82237be2519aca47307ab4208f"
)
N716_REPORT_SHA256 = (
    "e00494ee63bb6fdd6f0842b3a94e202e4a327e7265c0f13da31cc4a13215f492"
)
N717_REPORT_SHA256 = (
    "f7605851aeb7accd790f0fbd3fab5fe006acfa8b8d1710cc6b24dbf660787be8"
)
N719_REPORT_SHA256 = (
    "f60d872061beee8d9ec6a9ca982d6868d5df33d3b92be4d0c59852f0fb3087a3"
)
TRAIN_SHA256 = "6bb788c0ae27264fa8a68b75500228d9be3269e80bfa454c8d01f8bb654d91ec"
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)


def _ridge_tensors(
    auxiliary: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    active = auxiliary["active"].to(device=device, dtype=torch.bool)
    return {
        "mean": auxiliary["feature_mean"].to(device=device, dtype=torch.float32),
        "std": auxiliary["feature_std"].to(device=device, dtype=torch.float32),
        "active": active,
        "coefficient": auxiliary["coefficient"].to(device=device, dtype=torch.float32),
        "target_mean": torch.as_tensor(float(auxiliary["target_mean"]), device=device),
        "target_std": torch.as_tensor(float(auxiliary["target_std"]), device=device),
    }


def _ridge_predict_torch(feature: torch.Tensor, ridge: dict[str, torch.Tensor]) -> torch.Tensor:
    active = ridge["active"]
    standardized = (feature[..., active] - ridge["mean"][active]) / ridge["std"][active]
    coefficient = ridge["coefficient"]
    return (
        standardized @ coefficient[:-1] + coefficient[-1]
    ) * ridge["target_std"] + ridge["target_mean"]


def _soft_feature(deterministic: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.cat((deterministic.float(), logits.float().softmax(-1).flatten(-2)), -1)


def _losses(
    model: VQ2InformedDreamer,
    legal: torch.Tensor,
    action: torch.Tensor,
    state: RSSMState,
    ridge: dict[str, torch.Tensor],
    *,
    burn_in: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    kl_rows: list[torch.Tensor] = []
    posterior_progress: list[torch.Tensor] = []
    prior_progress: list[torch.Tensor] = []
    for step in range(legal.shape[1]):
        state, prior = model.rssm.observe_step(
            state, action[:, step], legal[:, step], deterministic_latent=True
        )
        kl_rows.append(model._categorical_kl(state.logits.detach(), prior))
        posterior_progress.append(
            _ridge_predict_torch(
                _soft_feature(state.deterministic.detach(), state.logits.detach()), ridge
            )
        )
        prior_progress.append(
            _ridge_predict_torch(_soft_feature(state.deterministic.detach(), prior), ridge)
        )
    kl = torch.stack(kl_rows, 1)[:, burn_in:].mean()
    posterior = torch.stack(posterior_progress, 1)[:, burn_in:].detach()
    prior = torch.stack(prior_progress, 1)[:, burn_in:]
    level = F.smooth_l1_loss(prior, posterior, beta=0.10)
    delta = F.smooth_l1_loss(torch.diff(prior, 1), torch.diff(posterior, 1), beta=0.05)
    total = kl + level + delta
    return total, {
        "kl": kl,
        "level": level,
        "delta": delta,
        "progress_mae": (prior - posterior).abs().mean(),
        "progress_rmse": (prior - posterior).square().mean().sqrt(),
    }


@torch.no_grad()
def _posterior_actions(
    model: VQ2InformedDreamer,
    legal: torch.Tensor,
    action: torch.Tensor,
    state: RSSMState,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for step in range(legal.shape[1]):
        state, _prior = model.rssm.observe_step(
            state, action[:, step], legal[:, step], deterministic_latent=True
        )
        rows.append(model.deterministic_actor_action(model.actor_distribution(state.features)))
    return torch.stack(rows, 1).float()


def _metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}


def continue_prior(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("progress prior distillation must not overwrite its parent")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n717_report": _sha256(args.n717_report),
        "n719_report": _sha256(args.n719_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N716_CHECKPOINT_SHA256,
        "n716_report": N716_REPORT_SHA256,
        "n717_report": N717_REPORT_SHA256,
        "n719_report": N719_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("progress prior source hash mismatch")
    n716 = json.loads(args.n716_report.read_text(encoding="utf-8"))
    n717 = json.loads(args.n717_report.read_text(encoding="utf-8"))
    n719 = json.loads(args.n719_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n716.get("process_passed") is not True
        or n717.get("process_passed") is not True
        or n717.get("prior_admitted") is not False
        or n719.get("process_passed") is not True
        or n719.get("selection_informative") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("progress prior predecessor contract mismatch")
    selected = n719.get("selected_candidate")
    if not isinstance(selected, dict) or (
        selected.get("objective") != "level_only"
        or selected.get("learning_rate") != 3e-7
        or selected.get("eligible") is not True
    ):
        raise RuntimeError("N719 selected-candidate contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("frozen ridge auxiliary is missing")
    if _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("frozen ridge hash mismatch")
    ridge = _ridge_tensors(auxiliary, device)
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
        lr=args.learning_rate, weight_decay=0.0,
    )
    reference = _posterior_reference(
        model, arrays, crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=719,
        batch_size=16, updates=args.updates,
    )
    first = schedule[0]
    first_event, first_start = first[:, 0], first[:, 1]
    first_legal, first_action = _legal_action_batch(
        arrays, first_event, first_start, sequence_length=128, device=device
    )
    first_initial = _pair_initial_state(
        arrays, reference, first_event, first_start, device=device
    )
    with torch.no_grad():
        _initial_total, initial_metrics_t = _losses(
            model, first_legal, first_action, first_initial, ridge, burn_in=32
        )
        initial_actions = _posterior_actions(
            model, first_legal, first_action, first_initial
        ).cpu()
    initial_metrics = _metrics_to_float(initial_metrics_t)
    last_metrics: dict[str, float] = {}
    last_gradient = math.nan
    projected_values = 0
    for update, batch in enumerate(schedule, 1):
        event, start = batch[:, 0], batch[:, 1]
        legal, action = _legal_action_batch(
            arrays, event, start, sequence_length=128, device=device
        )
        initial = _pair_initial_state(arrays, reference, event, start, device=device)
        optimizer.zero_grad(set_to_none=True)
        _combined, metrics_t = _losses(model, legal, action, initial, ridge, burn_in=32)
        total = metrics_t["level"]
        total.backward()
        gradient = _finite_clip_grad_norm(
            [parameter for _name, parameter in named_prior], 1.0,
            label="progress-RSSM-prior",
        )
        optimizer.step()
        projected_values += _project_prior_parameters(
            named_prior, model_before, args.maximum_parameter_delta
        )
        last_metrics = _metrics_to_float(metrics_t)
        last_metrics["total"] = float(total.detach().cpu())
        last_gradient = float(gradient.detach().cpu())
        if args.progress_every and update % args.progress_every == 0:
            print(json.dumps({"update": update, **last_metrics}, sort_keys=True), flush=True)
    with torch.no_grad():
        _final_total, final_metrics_t = _losses(
            model, first_legal, first_action, first_initial, ridge, burn_in=32
        )
        final_actions = _posterior_actions(
            model, first_legal, first_action, first_initial
        ).cpu()
    final_metrics = _metrics_to_float(final_metrics_t)
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
    action_difference = final_actions - initial_actions
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n716_report": _sha256(args.n716_report),
        "n717_report": _sha256(args.n717_report),
        "n719_report": _sha256(args.n719_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    sampling = _sampling_summary(schedule, len(arrays["mask"]))
    virtual_after = selected.get("after")
    if not isinstance(virtual_after, dict):
        raise RuntimeError("N719 selected candidate lacks after metrics")
    virtual_parity_error = max(
        abs(final_metrics[name] - float(virtual_after[name]))
        for name in ("kl", "level", "delta", "progress_mae", "progress_rmse")
    )
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "only_prior_tensors_changed": bool(changed) and forbidden == [],
        "posterior_actor_actions_bit_exact": bool(torch.equal(initial_actions, final_actions)),
        "first_batch_progress_mae_decreased": final_metrics["progress_mae"] < initial_metrics["progress_mae"],
        "first_batch_kl_preserved": final_metrics["kl"] <= initial_metrics["kl"] + 0.001,
        "finite_loss_and_gradient": all(math.isfinite(value) for value in [*last_metrics.values(), last_gradient]),
        "maximum_parameter_delta_bounded": maximum_delta <= args.maximum_parameter_delta,
        "fixed_ridge_unchanged": _fit_sha256(_fit_from_auxiliary(auxiliary)) == N710_FIT_SHA256,
        "no_validation_or_test_path": True,
        "sampling_exact": (
            args.updates == 1 and sampling["unique_pairs"] == 16
        ),
        "n719_virtual_metric_parity": virtual_parity_error <= 1e-6,
    }
    process_passed = all(process_gates.values())
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "maximum_parameter_delta_contract": args.maximum_parameter_delta,
        "sampling": sampling,
        "initial_first_batch": initial_metrics,
        "last_preupdate_batch": last_metrics,
        "final_first_batch": final_metrics,
        "n719_virtual_metric_parity_max_abs_error": virtual_parity_error,
        "objective": "level_only",
        "last_gradient_norm": last_gradient,
        "changed_model_keys": changed,
        "forbidden_model_keys": forbidden,
        "maximum_parameter_delta": maximum_delta,
        "projected_parameter_values": projected_values,
        "posterior_actor_action_rmse": float(action_difference.square().mean().sqrt()),
        "posterior_actor_action_max_abs": float(action_difference.abs().max()),
        "optimizer_steps": args.updates,
        "prior_updates": args.updates,
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
        report["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if not process_passed:
        raise RuntimeError(f"progress prior gates failed: {[k for k,v in process_gates.items() if not v]}")
    output = dict(payload)
    output["model"] = child_state
    output["offline_progress_prior_optimizer"] = optimizer.state_dict()
    output["offline_progress_prior_updates"] = args.updates
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
    parser.add_argument("--n716-report", type=Path, required=True)
    parser.add_argument("--n717-report", type=Path, required=True)
    parser.add_argument("--n719-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1,))
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--maximum-parameter-delta", type=float, default=0.001)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("progress prior refuses to overwrite outputs")
    if args.learning_rate != 3e-7 or args.maximum_parameter_delta != 0.001:
        parser.error("progress-prior learning rate and trust region are fixed")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_prior(parse_args()), indent=2, sort_keys=True))
