#!/usr/bin/env python3
"""Aggregate native checkpoint eval + SITL replay regression into one readiness report."""

from __future__ import annotations

import argparse
import json
import os


def load_json(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must decode to a JSON object")
    return payload


def write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build native->SITL transfer readiness report")
    parser.add_argument("--native-eval-json", required=True)
    parser.add_argument("--sitl-regression-json", required=True)
    parser.add_argument("--output-json", default=os.path.join("logs", "sitl", "native_transfer_readiness.json"))
    args = parser.parse_args()

    native = load_json(args.native_eval_json)
    sitl = load_json(args.sitl_regression_json)
    native_metrics = native.get("metrics", {})
    sitl_baseline = sitl.get("baseline", {})
    sitl_policy = sitl.get("policy", {})
    comparison = sitl.get("comparison", {})

    report = {
        "native_eval_json": os.path.abspath(args.native_eval_json),
        "sitl_regression_json": os.path.abspath(args.sitl_regression_json),
        "native": {
            "success_rate": native_metrics.get("env/success_rate"),
            "crash_rate": native_metrics.get("env/crash"),
            "gates_passed": native_metrics.get("env/gates_passed"),
        },
        "sitl": {
            "baseline_valid_rate": sitl_baseline.get("valid_rate"),
            "policy_valid_rate": sitl_policy.get("valid_rate"),
            "policy_non_inferior_to_baseline": comparison.get("policy_non_inferior_to_baseline"),
        },
        "transfer_ready": bool(
            comparison.get("policy_non_inferior_to_baseline")
            and sitl_policy.get("valid_rate", 0.0) >= sitl_baseline.get("valid_rate", 0.0)
        ),
    }
    write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
