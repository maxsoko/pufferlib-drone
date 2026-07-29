#!/usr/bin/env python3
"""Verify native rollout trace parity against full-history callable replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from capture_native_policy_trace import verify_trace_replay
except ModuleNotFoundError:
    from scripts.capture_native_policy_trace import verify_trace_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--action-atol", type=float, default=5e-5)
    parser.add_argument("--state-atol", type=float, default=5e-5)
    args = parser.parse_args()
    report = verify_trace_replay(
        args.checkpoint,
        args.trace,
        action_atol=args.action_atol,
        state_atol=args.state_atol,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
