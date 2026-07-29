#!/usr/bin/env python3
"""Calibrate only N711's RSSM prior on N681 train event windows."""

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
from scripts.package_vq2_invariant_donor_readout import (
    N710_FIT_SHA256,
    PACKAGE_SCHEMA,
    _fit_from_auxiliary,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import _finite_clip_grad_norm, _save_checkpoint


CONTINUATION_SCHEMA = "vq2_invariant_posterior_prior_calibration_v1"
N711_CHECKPOINT_SHA256 = (
    "92be8898d8294477139f0706a38604d0e3b1bd953a128f9e30c8f9e4c2e168dd"
)
N711_REPORT_SHA256 = (
    "dec6720a51785134f957f91a38f6de9c03b38595d477eb2c2d5ab04c1b91ce03"
)
N714_REPORT_SHA256 = (
    "647bf221f375d44c9c70b706db6603a94c913d935389a53737ac1db9e97fc4a0"
)
N715_REPORT_SHA256 = (
    "5e2170b04b8a6d4cf0beb9db4ac84774cc389c7530cf781bb3c810b0800f3369"
)
TRAIN_SHA256 = "6bb788c0ae27264fa8a68b75500228d9be3269e80bfa454c8d01f8bb654d91ec"
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)
FIXED_CROP_STARTS = (0, 32, 64, 96, 128, 160, 192)


def _load_train(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if arrays.get("mask", np.empty(0)).shape[:2] != (272, 321):
        raise RuntimeError("N681 train geometry must remain 272x321")
    return arrays


def _balanced_pairs(events: int, crop_starts: tuple[int, ...], seed: int) -> np.ndarray:
    if events <= 0 or not crop_starts:
        raise ValueError("balanced sampling requires events and crop starts")
    pairs = np.asarray(
        [(event, start) for event in range(events) for start in crop_starts],
        dtype=np.int64,
    )
    np.random.default_rng(seed).shuffle(pairs)
    return pairs


def _sampling_schedule(
    events: int,
    crop_starts: tuple[int, ...],
    *,
    seed: int,
    batch_size: int,
    updates: int,
) -> np.ndarray:
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    rows: list[np.ndarray] = []
    epoch = 0
    while sum(len(row) for row in rows) < updates * batch_size:
        rows.append(_balanced_pairs(events, crop_starts, seed + epoch))
        epoch += 1
    return np.concatenate(rows)[: updates * batch_size].reshape(updates, batch_size, 2)


@torch.no_grad()
def _posterior_reference(
    model: VQ2InformedDreamer,
    arrays: dict[str, np.ndarray],
    *,
    crop_starts: tuple[int, ...],
    microbatch_size: int,
    device: torch.device,
) -> dict[int, RSSMState]:
    wanted = {start - 1 for start in crop_starts if start > 0}
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        index: {field: [] for field in RSSMState._fields} for index in wanted
    }
    model.eval()
    for offset in range(0, len(arrays["mask"]), microbatch_size):
        events = np.arange(offset, min(len(arrays["mask"]), offset + microbatch_size))
        legal, action = _legal_action_batch(
            arrays, events, np.zeros(len(events), dtype=np.int64),
            sequence_length=max(FIXED_CROP_STARTS), device=device,
        )
        state = _initial_state(arrays, events, device=device)
        for step in range(legal.shape[1]):
            state, _prior = model.rssm.observe_step(
                state, action[:, step], legal[:, step], deterministic_latent=True
            )
            if step in wanted:
                for field in RSSMState._fields:
                    collected[step][field].append(getattr(state, field).float().cpu())
    return {
        step + 1: RSSMState(
            *(torch.cat(collected[step][field]) for field in RSSMState._fields)
        )
        for step in sorted(wanted)
    }


def _pair_initial_state(
    arrays: dict[str, np.ndarray],
    reference: dict[int, RSSMState],
    event: np.ndarray,
    start: np.ndarray,
    *,
    device: torch.device,
) -> RSSMState:
    rows: dict[str, list[torch.Tensor]] = {field: [] for field in RSSMState._fields}
    for chosen_event, chosen_start in zip(event, start, strict=True):
        if int(chosen_start) == 0:
            state = _initial_state(
                arrays, np.asarray([chosen_event], dtype=np.int64), device=torch.device("cpu")
            )
        else:
            source = reference[int(chosen_start)]
            state = RSSMState(*(value[int(chosen_event) : int(chosen_event) + 1] for value in source))
        for field in RSSMState._fields:
            rows[field].append(getattr(state, field))
    return RSSMState(*(torch.cat(rows[field]).to(device) for field in RSSMState._fields))


def _prior_kl(
    model: VQ2InformedDreamer,
    legal: torch.Tensor,
    action: torch.Tensor,
    state: RSSMState,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for step in range(legal.shape[1]):
        state, prior = model.rssm.observe_step(
            state, action[:, step], legal[:, step], deterministic_latent=True
        )
        rows.append(model._categorical_kl(state.logits.detach(), prior))
    return torch.stack(rows, 1).mean()


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
        rows.append(
            model.deterministic_actor_action(
                model.actor_distribution(state.features)
            ).float()
        )
    return torch.stack(rows, 1)


def _sampling_summary(schedule: np.ndarray, events: int) -> dict[str, object]:
    flat = schedule.reshape(-1, 2)
    pair_ids = flat[:, 0] * len(FIXED_CROP_STARTS) + np.searchsorted(
        np.asarray(FIXED_CROP_STARTS), flat[:, 1]
    )
    pair_counts = np.bincount(pair_ids, minlength=events * len(FIXED_CROP_STARTS))
    event_counts = np.bincount(flat[:, 0], minlength=events)
    return {
        "draws": int(len(flat)),
        "unique_pairs": int((pair_counts > 0).sum()),
        "unseen_pairs": int((pair_counts == 0).sum()),
        "pair_count_minimum": int(pair_counts.min()),
        "pair_count_maximum": int(pair_counts.max()),
        "event_count_minimum": int(event_counts.min()),
        "event_count_maximum": int(event_counts.max()),
    }


@torch.no_grad()
def _project_prior_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    source: dict[str, torch.Tensor],
    maximum_delta: float,
) -> int:
    if maximum_delta <= 0.0:
        raise ValueError("maximum delta must be positive")
    projected = 0
    for name, parameter in named_parameters:
        origin = source[name].to(parameter.device)
        delta = parameter - origin
        projected += int((delta.abs() > maximum_delta).sum().item())
        parameter.copy_(origin + delta.clamp(-maximum_delta, maximum_delta))
    return projected


def continue_prior(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("prior calibration must not overwrite its parent")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n711_report": _sha256(args.n711_report),
        "n714_report": _sha256(args.n714_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    if args.smoke_report is not None:
        source_before["smoke_report"] = _sha256(args.smoke_report)
    expected = {
        "checkpoint": N711_CHECKPOINT_SHA256,
        "n711_report": N711_REPORT_SHA256,
        "n714_report": N714_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if args.smoke_report is not None:
        expected["smoke_report"] = N715_REPORT_SHA256
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("prior calibration source hash mismatch")
    n711_report = json.loads(args.n711_report.read_text(encoding="utf-8"))
    n714_report = json.loads(args.n714_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    smoke_report = (
        None
        if args.smoke_report is None
        else json.loads(args.smoke_report.read_text(encoding="utf-8"))
    )
    if (
        n711_report.get("package_admitted") is not True
        or n714_report.get("process_passed") is not True
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("test_rows_scored") != 0
        or (
            smoke_report is not None
            and (
                smoke_report.get("contract") != CONTINUATION_SCHEMA
                or smoke_report.get("process_passed") is not True
                or smoke_report.get("updates") != 1
                or smoke_report.get("output_checkpoint_sha256")
                != "0dfc5c46771f7b3c668cfbe181ea279d9a02fc311dbee06cc9cf1c5be061918a"
            )
        )
    ):
        raise RuntimeError("prior calibration predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or auxiliary.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("N711 fixed ridge auxiliary is missing")
    if _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("N711 fixed ridge auxiliary hash mismatch")
    model = _model_from_payload(payload, device)
    model_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior_parameters = list(model.rssm.prior.parameters())
    named_prior_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("rssm.prior.")
    ]
    for parameter in prior_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        prior_parameters, lr=args.learning_rate, weight_decay=0.0
    )
    reference = _posterior_reference(
        model, arrays, crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size, device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]), FIXED_CROP_STARTS, seed=args.seed,
        batch_size=args.batch_size, updates=args.updates,
    )
    first = schedule[0]
    first_event, first_start = first[:, 0], first[:, 1]
    first_legal, first_action = _legal_action_batch(
        arrays, first_event, first_start,
        sequence_length=args.sequence_length, device=device,
    )
    first_initial = _pair_initial_state(
        arrays, reference, first_event, first_start, device=device
    )
    with torch.no_grad():
        initial_kl = float(_prior_kl(model, first_legal, first_action, first_initial).cpu())
        initial_actions = _posterior_actions(
            model, first_legal, first_action, first_initial
        ).cpu()
    last_kl = math.nan
    last_gradient = math.nan
    projected_parameter_values = 0
    for update, batch in enumerate(schedule, 1):
        event, start = batch[:, 0], batch[:, 1]
        legal, action = _legal_action_batch(
            arrays, event, start, sequence_length=args.sequence_length, device=device
        )
        initial = _pair_initial_state(arrays, reference, event, start, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = _prior_kl(model, legal, action, initial)
        loss.backward()
        gradient = _finite_clip_grad_norm(prior_parameters, 10.0, label="RSSM-prior")
        optimizer.step()
        if args.project_parameter_delta:
            projected_parameter_values += _project_prior_parameters(
                named_prior_parameters, model_before, args.maximum_parameter_delta
            )
        last_kl = float(loss.detach().cpu())
        last_gradient = float(gradient.detach().cpu())
        if args.progress_every and update % args.progress_every == 0:
            print(json.dumps({"update": update, "prior_kl": last_kl}, sort_keys=True), flush=True)
    with torch.no_grad():
        final_kl = float(_prior_kl(model, first_legal, first_action, first_initial).cpu())
        final_actions = _posterior_actions(
            model, first_legal, first_action, first_initial
        ).cpu()
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
        "n711_report": _sha256(args.n711_report),
        "n714_report": _sha256(args.n714_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    if args.smoke_report is not None:
        source_after["smoke_report"] = _sha256(args.smoke_report)
    sampling_summary = _sampling_summary(schedule, len(arrays["mask"]))
    full_schedule = args.updates == 357
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "only_prior_tensors_changed": bool(changed) and forbidden == [],
        "posterior_actor_actions_bit_exact": bool(torch.equal(initial_actions, final_actions)),
        "first_batch_kl_decreased": final_kl < initial_kl,
        "finite_loss_and_gradient": math.isfinite(last_kl) and math.isfinite(last_gradient),
        "maximum_parameter_delta_bounded": maximum_delta <= args.maximum_parameter_delta,
        "fixed_crop_contract": FIXED_CROP_STARTS == (0, 32, 64, 96, 128, 160, 192),
        "fixed_ridge_unchanged": _fit_sha256(_fit_from_auxiliary(auxiliary)) == N710_FIT_SHA256,
        "no_validation_or_test_path": True,
        "smoke_contract_exact_when_required": (
            args.updates == 1 or smoke_report is not None
        ),
        "projection_contract_exact": (
            (args.updates == 1 and not args.project_parameter_delta)
            or (full_schedule and args.project_parameter_delta)
        ),
        "full_schedule_balanced_when_required": (
            not full_schedule
            or (
                sampling_summary["draws"] == 5712
                and sampling_summary["unseen_pairs"] == 0
                and sampling_summary["pair_count_minimum"] == 3
                and sampling_summary["pair_count_maximum"] == 3
                and sampling_summary["event_count_minimum"] == 21
                and sampling_summary["event_count_maximum"] == 21
            )
        ),
    }
    process_passed = all(process_gates.values())
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "crop_starts": list(FIXED_CROP_STARTS),
        "sampling": sampling_summary,
        "initial_first_batch_kl": initial_kl,
        "last_preupdate_batch_kl": last_kl,
        "final_first_batch_kl": final_kl,
        "last_gradient_norm": last_gradient,
        "changed_model_keys": changed,
        "forbidden_model_keys": forbidden,
        "maximum_parameter_delta": maximum_delta,
        "projection_enabled": args.project_parameter_delta,
        "projected_parameter_values": projected_parameter_values,
        "posterior_actor_action_rmse": float(action_difference.square().mean().sqrt()),
        "posterior_actor_action_max_abs": float(action_difference.abs().max()),
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "representation_updates": 0,
        "reward_updates": 0,
        "prior_updates": args.updates,
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
        raise RuntimeError(f"prior calibration gates failed: {[k for k,v in process_gates.items() if not v]}")
    output = dict(payload)
    output["model"] = child_state
    output["offline_event_prior_optimizer"] = optimizer.state_dict()
    output["offline_event_prior_updates"] = int(payload.get("offline_event_prior_updates", 0)) + args.updates
    output["offline_event_prior_parent_sha256"] = source_before["checkpoint"]
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
    parser.add_argument("--n711-report", type=Path, required=True)
    parser.add_argument("--n714-report", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--maximum-parameter-delta", type=float, default=0.005)
    parser.add_argument("--project-parameter-delta", action="store_true")
    parser.add_argument("--seed", type=int, default=715)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("prior calibration refuses to overwrite outputs")
    if args.updates not in (1, 357) or args.learning_rate <= 0.0:
        parser.error("updates must be 1 or 357 and learning rate must be positive")
    if args.batch_size != 16 or args.sequence_length != 128 or args.seed != 715:
        parser.error("prior calibration sampler contract is fixed")
    if args.updates == 357 and (
        args.smoke_report is None or not args.project_parameter_delta
    ):
        parser.error("357 updates require the admitted smoke report and projection")
    if args.updates == 1 and args.project_parameter_delta:
        parser.error("one-update smoke must measure its unprojected step")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_prior(parse_args()), indent=2, sort_keys=True))
