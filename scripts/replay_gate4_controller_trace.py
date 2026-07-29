#!/usr/bin/env python3
"""Replay retained official observations through a policy callable.

The official smoke runner retains policy observations at a lower diagnostic
rate than the fixed recurrent clock. This tool holds each retained observation
until the next sample and advances the callable on the same fixed schedule.
It records controller-plan transitions so stale-track and failsafe behavior can
be compared without another simulator flight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_callable(path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location("_gate4_trace_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load policy source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    reset = getattr(module, "reset", None)
    snapshot = getattr(module, "controller_snapshot", None)
    if not callable(function) or not callable(reset) or not callable(snapshot):
        raise RuntimeError("policy must expose infer, reset, and controller_snapshot")
    return function, reset, snapshot


def _mpc_snapshot(snapshot: dict) -> dict:
    for key in (
        "n237_gate4_fresh_projected_intercept",
        "n236_gate4_staged_mpc",
        "n235_gate4_search_mpc",
        "n234_gate4_reacquiring_mpc",
        "n232_gate4_visual_mpc",
    ):
        value = snapshot.get(key)
        if isinstance(value, dict):
            return value
    nested = snapshot.get("n232_gate4_visual_mpc")
    if isinstance(nested, dict):
        return nested
    raise KeyError("controller snapshot has no supported Gate-4 MPC section")


def replay(args: argparse.Namespace) -> dict:
    payload = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    manifest = payload.get("policy_trace", {}).get("deployment_manifest", {})
    deployment_paths = {
        "PUFFER_POLICY_CHECKPOINT_PATH": "checkpoint",
        "PUFFER_POLICY_PREFIX_CHECKPOINT_PATH": "checkpoint",
        "PUFFER_POLICY_GATE4_CHECKPOINT_PATH": "gate4_checkpoint",
        "PUFFER_POLICY_GATE5_CHECKPOINT_PATH": "gate5_checkpoint",
        "PUFFER_POLICY_GATE6_CHECKPOINT_PATH": "gate6_checkpoint",
    }
    for environment_name, manifest_name in deployment_paths.items():
        path = manifest.get(manifest_name, {}).get("path")
        if path:
            os.environ[environment_name] = str(path)
    os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
    os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
    os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
    os.environ["PUFFER_GATE3_COUNTER_ENTRY_M"] = "-1.2"
    os.environ["PUFFER_POLICY_GATE4_DYNAMICS_PATH"] = str(args.model.resolve())
    infer, reset, snapshot_fn = _load_callable(
        args.policy.resolve(), args.function
    )
    reset()
    samples = list(payload.get("policy_trace", {}).get("samples", []))
    if not samples:
        raise RuntimeError("trace contains no retained policy samples")

    calls = 0
    last_plan_marker = None
    last_failsafe_count = 0
    plan_reasons: Counter[str] = Counter()
    events: list[dict] = []
    replay_action = [0.0, 0.0, 0.0, 0.0]
    for sample in samples:
        elapsed_s = float(sample["elapsed_s"])
        target_calls = int(math.floor(elapsed_s * args.state_hz + 1e-9)) + 1
        observation = sample["observation"]
        while calls < target_calls:
            replay_action = [float(value) for value in infer(observation)]
            calls += 1
            full_snapshot = snapshot_fn()
            mpc_snapshot = _mpc_snapshot(full_snapshot)
            last_plan = mpc_snapshot.get("last_plan")
            marker = mpc_snapshot.get("last_plan_s")
            failsafe_count = int(mpc_snapshot.get("failsafe_plans", 0))
            reason = None if last_plan is None else str(last_plan.get("reason"))
            changed = marker != last_plan_marker
            no_state_failsafe = failsafe_count > last_failsafe_count and not changed
            if changed or no_state_failsafe:
                event_reason = "no_admitted_state" if no_state_failsafe else reason
                plan_reasons[event_reason or "unknown"] += 1
                events.append(
                    {
                        "call": calls,
                        "logical_time_s": round((calls - 1) / args.state_hz, 6),
                        "trace_elapsed_s": elapsed_s,
                        "official_active_gate_index": sample.get(
                            "official_active_gate_index"
                        ),
                        "reason": event_reason,
                        "action": replay_action,
                        "tracker": mpc_snapshot.get("tracker"),
                    }
                )
            last_plan_marker = marker
            last_failsafe_count = failsafe_count

    final_snapshot = snapshot_fn()
    return {
        "trace": str(args.trace.resolve()),
        "policy": str(args.policy.resolve()),
        "model": str(args.model.resolve()),
        "state_hz": args.state_hz,
        "retained_samples": len(samples),
        "replayed_calls": calls,
        "plan_reason_counts": dict(sorted(plan_reasons.items())),
        "maximum_abs_replay_action": max(
            abs(float(value))
            for event in events
            for value in event.get("action", [])
        ),
        "events": events,
        "final_controller_snapshot": final_snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--function", default="infer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-hz", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.state_hz) or args.state_hz <= 0.0:
        raise ValueError("state-hz must be finite and positive")
    result = replay(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
