#!/usr/bin/env python3
"""Rotating-seed coordinate search for a recurrent full-policy action head.

The search keeps the recurrent policy and runtime controller unchanged. It
first ranks roll-row scales, then ranks thrust-row scales around the best roll
candidate. Screens use at least two episodes per 1,024 vector environments so
the fixed first-reset seed population cannot produce an optimistic promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval_drone_race_checkpoint.py"
SCALER = ROOT / "scripts" / "scale_policy_decoder_action.py"


def scale_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def candidate_score(metrics: dict[str, float], crash_limit: float) -> tuple[float, ...]:
    crash = float(metrics.get("env/crash", 1.0))
    valid = 1.0 if crash <= crash_limit else 0.0
    return (
        valid,
        float(metrics.get("env/success_rate", 0.0)),
        float(metrics.get("env/gates_passed", 0.0)),
        -crash,
        -float(
            metrics.get(
                "env/terminal_crossing_radial",
                metrics.get("env/avg_terminal_crossing_radial", math.inf),
            )
        ),
    )


def passes(metrics: dict[str, float], success_threshold: float, crash_limit: float) -> bool:
    return (
        float(metrics.get("env/success_rate", 0.0)) >= success_threshold
        and float(metrics.get("env/crash", 1.0)) <= crash_limit
    )


def run_quiet(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def scale_checkpoint(
    *, source: Path, output: Path, action_index: int, scale: float, input_dim: int
) -> None:
    if output.is_file():
        return
    run_quiet([
        sys.executable,
        str(SCALER),
        str(source),
        str(output),
        "--action-index",
        str(action_index),
        "--scale",
        str(scale),
        "--input-dim",
        str(input_dim),
    ])


def evaluate(
    *,
    checkpoint: Path,
    report: Path,
    radius: float,
    geometry_scale: float,
    episodes: int,
    env_name: str,
    max_rollouts: int,
) -> dict[str, float]:
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if (
            int(metadata.get("eval_episodes_target", -1)) == episodes
            and Path(metadata.get("weights", "")).resolve() == checkpoint.resolve()
        ):
            return payload["metrics"]
    csv_path = report.with_suffix(".csv")
    run_quiet([
        sys.executable,
        str(EVALUATOR),
        str(checkpoint),
        "--env-name",
        env_name,
        "--eval-episodes",
        str(episodes),
        "--require-exact-episodes",
        "--horizon",
        "32",
        "--max-rollouts",
        str(max_rollouts),
        "--json-path",
        str(report),
        "--csv-path",
        str(csv_path),
        "--label",
        report.stem,
        "--env.num-gates",
        "4",
        "--env.course-geometry-scale",
        f"{geometry_scale:.9g}",
        "--env.course-geometry-scale-randomize",
        "0",
        "--env.gate-radius",
        f"{radius:.6g}",
        "--env.sitl-gate-obs-dropout-from-index",
        "1",
        "--env.sitl-gate-obs-dropout-range-m",
        "0.00",
        "--env.gate-position-domain-randomize",
        "0",
    ])
    return json.loads(report.read_text(encoding="utf-8"))["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--geometry-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roll-scales", type=float, nargs="+", default=(0.90, 0.95, 1.0, 1.05, 1.10)
    )
    parser.add_argument(
        "--thrust-scales", type=float, nargs="+", default=(0.90, 0.95, 1.0, 1.05, 1.10)
    )
    parser.add_argument("--screen-episodes", type=int, default=2048)
    parser.add_argument("--promotion-episodes", type=int, default=4096)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    parser.add_argument("--env-name", default="drone_race_full_policy_stage_d_gate4")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--max-rollouts", type=int, default=192)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.radius <= 0.0:
        parser.error("--radius must be positive")
    if not 0.0 <= args.geometry_scale <= 1.0:
        parser.error("--geometry-scale must be in [0, 1]")
    if args.screen_episodes < 2048 or args.screen_episodes % 1024:
        parser.error("--screen-episodes must be a multiple of 1024 and at least 2048")
    if args.promotion_episodes < 4096 or args.promotion_episodes % 1024:
        parser.error("--promotion-episodes must be a multiple of 1024 and at least 4096")
    for name, values in (("roll", args.roll_scales), ("thrust", args.thrust_scales)):
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            parser.error(f"--{name}-scales must contain positive finite values")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.checkpoint.resolve()
    candidates: list[dict] = []

    def screen(checkpoint: Path, label: str, **parameters: float) -> dict:
        report = output_dir / f"screen_{label}.json"
        metrics = evaluate(
            checkpoint=checkpoint,
            report=report,
            radius=args.radius,
            geometry_scale=args.geometry_scale,
            episodes=args.screen_episodes,
            env_name=args.env_name,
            max_rollouts=args.max_rollouts,
        )
        candidate = {
            "checkpoint": str(checkpoint),
            "report": str(report),
            "parameters": parameters,
            "metrics": metrics,
        }
        candidates.append(candidate)
        print(
            f"{label}: success={metrics.get('env/success_rate', 0.0):.6f} "
            f"crash={metrics.get('env/crash', 0.0):.6f} "
            f"gates={metrics.get('env/gates_passed', 0.0):.6f}",
            flush=True,
        )
        return candidate

    roll_candidates: list[dict] = []
    for roll_scale in args.roll_scales:
        label = f"roll_{scale_tag(roll_scale)}"
        checkpoint = output_dir / f"{label}.bin"
        scale_checkpoint(
            source=source,
            output=checkpoint,
            action_index=1,
            scale=roll_scale,
            input_dim=args.input_dim,
        )
        roll_candidates.append(screen(checkpoint, label, roll_scale=roll_scale))
    best_roll = max(
        roll_candidates,
        key=lambda item: candidate_score(item["metrics"], args.crash_limit),
    )

    thrust_candidates: list[dict] = []
    for thrust_scale in args.thrust_scales:
        roll_scale = float(best_roll["parameters"]["roll_scale"])
        label = f"roll_{scale_tag(roll_scale)}_thrust_{scale_tag(thrust_scale)}"
        checkpoint = output_dir / f"{label}.bin"
        scale_checkpoint(
            source=Path(best_roll["checkpoint"]),
            output=checkpoint,
            action_index=2,
            scale=thrust_scale,
            input_dim=args.input_dim,
        )
        thrust_candidates.append(
            screen(
                checkpoint,
                label,
                roll_scale=roll_scale,
                thrust_scale=thrust_scale,
            )
        )

    best = max(
        candidates,
        key=lambda item: candidate_score(item["metrics"], args.crash_limit),
    )
    promotion = None
    promoted = False
    if passes(best["metrics"], args.success_threshold, args.crash_limit):
        promotion_report = output_dir / "promotion_best.json"
        promotion_metrics = evaluate(
            checkpoint=Path(best["checkpoint"]),
            report=promotion_report,
            radius=args.radius,
            geometry_scale=args.geometry_scale,
            episodes=args.promotion_episodes,
            env_name=args.env_name,
            max_rollouts=args.max_rollouts,
        )
        promoted = passes(
            promotion_metrics, args.success_threshold, args.crash_limit
        )
        promotion = {
            "checkpoint": best["checkpoint"],
            "report": str(promotion_report),
            "metrics": promotion_metrics,
            "accepted": promoted,
        }

    manifest = {
        "source": str(source),
        "radius": args.radius,
        "geometry_scale": args.geometry_scale,
        "screen_episodes": args.screen_episodes,
        "promotion_episodes": args.promotion_episodes,
        "success_threshold": args.success_threshold,
        "crash_limit": args.crash_limit,
        "candidates": candidates,
        "best_screen": best,
        "promotion": promotion,
    }
    manifest_path = output_dir / "decoder_scale_search.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest={manifest_path}")
    if promotion is None:
        print("No screen candidate met the promotion threshold.")
        return 3
    print(
        f"promotion success={promotion['metrics'].get('env/success_rate', 0.0):.6f} "
        f"crash={promotion['metrics'].get('env/crash', 0.0):.6f} "
        f"accepted={promoted} checkpoint={promotion['checkpoint']}"
    )
    return 0 if promoted else 3


if __name__ == "__main__":
    raise SystemExit(main())
