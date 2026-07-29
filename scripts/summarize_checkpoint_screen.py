#!/usr/bin/env python3
"""Build a deterministic aggregate for a directory of checkpoint evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(metrics: dict[str, float], name: str, default: float = 0.0) -> float:
    return float(metrics.get(f"env/{name}", default))


def summarize(
    input_dir: Path,
    *,
    min_success_rate: float,
    max_crash_rate: float,
    max_timeout_rate: float,
) -> dict:
    rows = []
    for evaluation_path in sorted(input_dir.glob("*.json")):
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metadata = evaluation["metadata"]
        metrics = evaluation["metrics"]
        checkpoint = Path(metadata["weights"])
        csv_path = evaluation_path.with_suffix(".csv")
        try:
            step = int(evaluation_path.stem)
        except ValueError:
            step = -1
        row = {
            "step": step,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluation": str(evaluation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "csv": str(csv_path) if csv_path.exists() else None,
            "csv_sha256": sha256_file(csv_path) if csv_path.exists() else None,
            "episodes": int(round(metric(metrics, "n"))),
            "success_rate": metric(metrics, "success_rate"),
            "gates_passed": metric(metrics, "gates_passed"),
            "crash_rate": metric(metrics, "crash"),
            "timeout_rate": metric(metrics, "timeout"),
            "missed_gate_rate": metric(metrics, "missed_gate"),
            "terminal_crossing_radial": metric(
                metrics, "avg_terminal_crossing_radial", float("inf")
            ),
            "terminal_crossing_right": metric(
                metrics, "avg_terminal_crossing_right", float("inf")
            ),
            "terminal_crossing_vertical": metric(
                metrics, "avg_terminal_crossing_vertical", float("inf")
            ),
        }
        row["promotion_passed"] = bool(
            row["success_rate"] >= min_success_rate
            and row["crash_rate"] <= max_crash_rate
            and row["timeout_rate"] <= max_timeout_rate
        )
        rows.append(row)

    if not rows:
        raise ValueError(f"no evaluation JSON files found under {input_dir}")

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["success_rate"],
            -row["gates_passed"],
            row["crash_rate"],
            row["timeout_rate"],
            row["terminal_crossing_radial"],
            row["step"],
        ),
    )
    return {
        "input_dir": str(input_dir),
        "criteria": {
            "min_success_rate": min_success_rate,
            "max_crash_rate": max_crash_rate,
            "max_timeout_rate": max_timeout_rate,
        },
        "evaluations": len(rows),
        "promotion_passed": any(row["promotion_passed"] for row in rows),
        "qualifying_steps": [
            row["step"] for row in ranked if row["promotion_passed"]
        ],
        "best": ranked[0],
        "rows": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--min-success-rate", type=float, default=0.9)
    parser.add_argument("--max-crash-rate", type=float, default=0.0)
    parser.add_argument("--max-timeout-rate", type=float, default=0.0)
    args = parser.parse_args()
    result = summarize(
        args.input_dir,
        min_success_rate=args.min_success_rate,
        max_crash_rate=args.max_crash_rate,
        max_timeout_rate=args.max_timeout_rate,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["promotion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
