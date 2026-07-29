#!/usr/bin/env python3
"""Read-only full-train aggregate multi-horizon prior surface."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_frozen_ridge_readout import _fit_sha256
from scripts.audit_vq2_prior_multihorizon_surface import (
    HORIZONS,
    _eligible,
    _float_rows,
    _observe,
    _restore,
    _selection_key,
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
from scripts.continue_vq2_prior_progress_offline import (
    MATERIALIZATION_SHA256,
    TRAIN_SHA256,
    _ridge_tensors,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _legal_action_batch,
    _model_from_payload,
)
from scripts.package_vq2_invariant_donor_readout import N710_FIT_SHA256, _fit_from_auxiliary
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_prior_multihorizon_progress_full_train_aggregate_surface_v1"
N720_CHECKPOINT_SHA256 = (
    "4d7cc2b64f4760a15186163cdac5068b58459c7fa7bc0f3186e864d809760897"
)
N723_REPORT_SHA256 = "a176d5fa4305491ff1961c91aef06a40491ad4658893a59065d52aa9a72916c2"
N724_REPORT_SHA256 = "4b3f03a3efcdebc2cc681538f0239648f8b84a06fefa2f8eb9a580d857fc0f80"
N726_REPORT_SHA256 = "b2b69f465da3c3093c0dd981efb8f017031319738bff1104d1297a9014d5ff9f"
LEARNING_RATES = (1e-9, 3e-9, 1e-8, 3e-8, 1e-7)


def _mean_rows(
    rows: list[dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("aggregate surface requires rows")
    result: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        result[key] = {
            "loss": float(np.mean([row[key]["loss"] for row in rows])),
            "progress_mae": float(
                np.mean([row[key]["progress_mae"] for row in rows])
            ),
            "progress_rmse": float(
                np.sqrt(np.mean([row[key]["progress_rmse"] ** 2 for row in rows]))
            ),
            "progress_bias": float(
                np.mean([row[key]["progress_bias"] for row in rows])
            ),
            "kl": float(np.mean([row[key]["kl"] for row in rows])),
            "action_rmse": float(
                np.sqrt(np.mean([row[key]["action_rmse"] ** 2 for row in rows]))
            ),
            "action_max_abs": float(
                max(row[key]["action_max_abs"] for row in rows)
            ),
        }
    return result


@torch.no_grad()
def _evaluate_schedule(
    model,
    arrays: dict[str, np.ndarray],
    reference,
    schedule: np.ndarray,
    ridge,
    *,
    device: torch.device,
    progress_every: int = 0,
    phase: str = "evaluation",
) -> dict[str, dict[str, float]]:
    rows: list[dict[str, dict[str, float]]] = []
    for batch_index, batch in enumerate(schedule, 1):
        event, start = batch[:, 0], batch[:, 1]
        legal, action = _legal_action_batch(
            arrays, event, start, sequence_length=128, device=device
        )
        initial = _pair_initial_state(arrays, reference, event, start, device=device)
        posterior = _observe(model, legal, action, initial)
        _objective, metrics = _surface_metrics(
            model, posterior, action, ridge, burn_in=32, stride=8
        )
        rows.append(_float_rows(metrics))
        if progress_every and batch_index % progress_every == 0:
            print(
                json.dumps(
                    {"phase": phase, "batch": batch_index, "batches": len(schedule)},
                    sort_keys=True,
                ),
                flush=True,
            )
    return _mean_rows(rows)


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n723_report": _sha256(args.n723_report),
        "n724_report": _sha256(args.n724_report),
        "n726_report": _sha256(args.n726_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    expected = {
        "checkpoint": N720_CHECKPOINT_SHA256,
        "n723_report": N723_REPORT_SHA256,
        "n724_report": N724_REPORT_SHA256,
        "n726_report": N726_REPORT_SHA256,
        "train_dataset": TRAIN_SHA256,
        "materialization_report": MATERIALIZATION_SHA256,
    }
    if any(source_before[name] != digest for name, digest in expected.items()):
        raise RuntimeError("aggregate multi-horizon source hash mismatch")
    n723 = json.loads(args.n723_report.read_text(encoding="utf-8"))
    n724 = json.loads(args.n724_report.read_text(encoding="utf-8"))
    n726 = json.loads(args.n726_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        n723.get("process_passed") is not True
        or n724.get("process_passed") is not True
        or n724.get("selection_informative") is not True
        or n726.get("process_passed") is not True
        or n726.get("prior_admitted") is not False
        or n726.get("improved_all_horizons_vs_n720") is not False
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
    ):
        raise RuntimeError("aggregate multi-horizon predecessor contract mismatch")
    arrays = _load_train(args.train_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    auxiliary = payload.get("frozen_invariant_ridge_aux")
    if not isinstance(auxiliary, dict) or _fit_sha256(_fit_from_auxiliary(auxiliary)) != N710_FIT_SHA256:
        raise RuntimeError("aggregate multi-horizon fixed ridge mismatch")
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
    ridge = _ridge_tensors(auxiliary, device)
    reference = _posterior_reference(
        model,
        arrays,
        crop_starts=FIXED_CROP_STARTS,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    schedule = _sampling_schedule(
        len(arrays["mask"]),
        FIXED_CROP_STARTS,
        seed=727,
        batch_size=16,
        updates=119,
    )
    sampling = _sampling_summary(schedule, len(arrays["mask"]))
    model.zero_grad(set_to_none=True)
    source_rows: list[dict[str, dict[str, float]]] = []
    for batch_index, batch in enumerate(schedule, 1):
        event, start = batch[:, 0], batch[:, 1]
        legal, action = _legal_action_batch(
            arrays, event, start, sequence_length=128, device=device
        )
        initial = _pair_initial_state(arrays, reference, event, start, device=device)
        posterior = _observe(model, legal, action, initial)
        objective, metrics = _surface_metrics(
            model, posterior, action, ridge, burn_in=32, stride=8
        )
        (objective / len(schedule)).backward()
        source_rows.append(_float_rows(metrics))
        if args.progress_every and batch_index % args.progress_every == 0:
            print(
                json.dumps(
                    {"phase": "gradient", "batch": batch_index, "batches": len(schedule)},
                    sort_keys=True,
                ),
                flush=True,
            )
    before = _mean_rows(source_rows)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _name, parameter in named_prior], 1.0
    )
    gradients = {name: parameter.grad.detach().clone() for name, parameter in named_prior}
    rows: list[dict[str, object]] = []
    for rate_index, learning_rate in enumerate(LEARNING_RATES, 1):
        _virtual_adam_step(named_prior, model_before, gradients, learning_rate)
        after = _evaluate_schedule(
            model,
            arrays,
            reference,
            schedule,
            ridge,
            device=device,
            progress_every=args.progress_every,
            phase=f"rate_{learning_rate:g}",
        )
        eligible = _eligible(before, after)
        rows.append(
            {
                "learning_rate": learning_rate,
                "before": before,
                "after": after,
                "eligible": eligible,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "rate_complete",
                    "rate_index": rate_index,
                    "rates": len(LEARNING_RATES),
                    "learning_rate": learning_rate,
                    "eligible": eligible,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _restore(named_prior, model_before)
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=_selection_key) if eligible_rows else None
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), model_before[name])
    ]
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n723_report": _sha256(args.n723_report),
        "n724_report": _sha256(args.n724_report),
        "n726_report": _sha256(args.n726_report),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "sources_exact": all(source_after[name] == digest for name, digest in expected.items()),
        "model_restored_exact": changed == [],
        "full_train_schedule_exact": (
            sampling["draws"] == 1904
            and sampling["unseen_pairs"] == 0
            and sampling["pair_count_minimum"] == 1
            and sampling["pair_count_maximum"] == 1
            and sampling["event_count_minimum"] == 7
            and sampling["event_count_maximum"] == 7
        ),
        "fixed_horizons_exact": HORIZONS == (1, 2, 4, 8, 16),
        "five_virtual_rates": len(rows) == 5 and LEARNING_RATES == (
            1e-9,
            3e-9,
            1e-8,
            3e-8,
            1e-7,
        ),
        "finite_surface": all(
            math.isfinite(float(value))
            for row in rows
            for section in ("before", "after")
            for horizon in row[section].values()
            for value in horizon.values()
        )
        and math.isfinite(float(gradient_norm.cpu())),
        "no_validation_or_test_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "sampling": sampling,
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
        raise RuntimeError("aggregate multi-horizon process gates failed")
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
    parser.add_argument("--n723-report", type=Path, required=True)
    parser.add_argument("--n724-report", type=Path, required=True)
    parser.add_argument("--n726-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("aggregate multi-horizon surface refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
