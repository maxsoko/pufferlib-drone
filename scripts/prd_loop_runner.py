#!/usr/bin/env python3
"""PRD-driven loop orchestrator (milestone 4 focus).

Reads PRD priorities and runs one actionable tick per invocation:
  - Track A (70%): multi-gate official sim validation when streams are live
  - Track B: paused by default; must be explicitly requested

State: logs/prd/state.json

Usage:
  python scripts/prd_loop_runner.py
  python scripts/prd_loop_runner.py --tag prd_loop_tick
  python scripts/prd_loop_runner.py --run-track-b-train  # explicit support work
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "logs" / "prd" / "state.json"
VENV_PY = ROOT / ".venv-win" / "Scripts" / "python.exe"


def _now() -> float:
    return time.time()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "active_milestone": 4,
        "track_a": {"multigate_passed": False, "attempts": []},
        "track_b": {"env_name": None, "training_status": None, "attempts": []},
        "updated_unix": _now(),
    }


def save_state(state: dict) -> None:
    state["updated_unix"] = _now()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    print(f"==> {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_multigate_validation(
    tag: str,
    *,
    smoke_duration_s: int = 90,
    target_gate_count: int = 2,
    policy_callable: str = "",
    policy_activate_gate_index: int = 1,
) -> dict:
    """Guidance-plus-gate FSM run targeting official gate 2+."""
    py = str(VENV_PY if VENV_PY.exists() else sys.executable)
    summary = ROOT / "logs" / "prd" / f"multigate_validation_{tag}.json"
    debug_frames = ROOT / "logs" / "sitl" / f"course_fsm_{tag}_frames"
    cmd = [
            py,
            str(ROOT / "scripts" / "run_official_gate1_validation.py"),
            "--host",
            "0.0.0.0",
            "--mavlink-port",
            "14550",
            "--camera-port",
            "5600",
            "--endpoint",
            "udpin:0.0.0.0:14550",
            "--camera-host",
            "0.0.0.0",
            "--probe-duration",
            "5",
            "--smoke-duration",
            str(smoke_duration_s),
            "--command-hz",
            "60",
            "--post-reset-sleep-s",
            "5",
            "--send-sim-reset",
            "--require-official-race-progress",
            "--acceptance-config",
            str(ROOT / "config" / "sitl_multigate_acceptance.json"),
            "--control-mode",
            "course-fsm",
            "--target-gate-count",
            str(target_gate_count),
            "--min-gate-passes",
            str(target_gate_count),
            "--debug-frame-dir",
            str(debug_frames),
            "--attitude-servo-hover-thrust",
            "0.27",
            "--attitude-servo-min-thrust",
            "0.18",
            "--attitude-servo-max-thrust",
            "0.42",
            "--summary-json-path",
            str(summary),
            "--probe-json-path",
            str(ROOT / "logs" / "prd" / f"stream_probe_{tag}.json"),
            "--smoke-json-path",
            str(ROOT / "logs" / "sitl" / f"competition_smoke_multigate_{tag}.json"),
            "--smoke-csv-path",
            str(ROOT / "logs" / "sitl" / f"competition_smoke_multigate_{tag}.csv"),
        ]
    if policy_callable:
        cmd.extend(
            [
                "--policy-callable",
                policy_callable,
                "--course-policy-activate-gate-index",
                str(policy_activate_gate_index),
            ]
        )
    proc = run_cmd(cmd, timeout=180)
    result = {
        "tag": tag,
        "unix": _now(),
        "exit_code": proc.returncode,
        "summary_path": str(summary),
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
    }
    if summary.exists():
        summary_data = json.loads(summary.read_text(encoding="utf-8"))
        result["status"] = summary_data.get("status")
        smoke = summary_data.get("smoke") or {}
        result["official_active_gate_index"] = smoke.get("official_active_gate_index")
        result["acceptance_passed"] = smoke.get("acceptance_passed")
        result["multigate_passed"] = int(smoke.get("official_active_gate_index") or 0) >= 2
    return result


def wsl_training_running() -> bool:
    proc = subprocess.run(
        [
            "wsl",
            "bash",
            "-c",
            "pgrep -af 'pufferlib.pufferl train drone_race_sitl_plant' || true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if "pufferlib.pufferl train" in ln]
    return len(lines) > 0


def start_track_b_hover_training() -> dict:
    if wsl_training_running():
        return {"status": "skipped", "reason": "training_already_running"}
    sync = subprocess.run(
        ["wsl", "bash", str(ROOT / "scripts" / "sync_wsl_ext4_repo.sh"), "~/pufferlib-drone"],
        capture_output=True,
        text=True,
        check=False,
    )
    train = subprocess.Popen(
        [
            "wsl",
            "bash",
            "-c",
            "export ENV_NAME=drone_race_sitl_plant_gate1_hover FRESH_START=1; "
            "bash ~/pufferlib-drone/scripts/train_gate1_wsl.sh",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "started",
        "pid": train.pid,
        "env_name": "drone_race_sitl_plant_gate1_hover",
        "sync_exit": sync.returncode,
    }


def tick(
    *,
    tag: str,
    run_track_b: bool,
    smoke_duration_s: int = 90,
    target_gate_count: int = 2,
    policy_callable: str = "",
    policy_activate_gate_index: int = 1,
) -> dict:
    state = load_state()
    report: dict = {"tag": tag, "unix": _now(), "actions": []}

    # The validation runner already performs the required stream probe. Avoid
    # probing twice before every attempt; it wastes five seconds and can consume
    # the first post-reset camera/telemetry window.
    mg = run_multigate_validation(
        tag,
        smoke_duration_s=smoke_duration_s,
        target_gate_count=target_gate_count,
        policy_callable=policy_callable,
        policy_activate_gate_index=policy_activate_gate_index,
    )
    report["actions"].append({"track_a_multigate": mg})
    state["track_a"]["attempts"].append(mg)
    if mg.get("multigate_passed"):
        state["track_a"]["multigate_passed"] = True

    if run_track_b:
        if not wsl_training_running():
            tb = start_track_b_hover_training()
        else:
            tb = {"status": "skipped", "reason": "training_already_running"}
        report["actions"].append({"track_b_train": tb})
        state["track_b"]["env_name"] = "drone_race_sitl_plant_gate1_hover"
        state["track_b"]["training_status"] = tb.get("status")
        state["track_b"]["attempts"].append(tb)

    save_state(state)
    report["state_path"] = str(STATE_PATH)
    out = ROOT / "logs" / "prd" / f"tick_{tag}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PRD loop single tick")
    parser.add_argument("--tag", default=f"prd_{int(_now())}")
    parser.add_argument(
        "--run-track-b-train",
        action="store_true",
        help="Explicitly start support-track native training (paused by default)",
    )
    parser.add_argument("--smoke-duration", type=int, default=90)
    parser.add_argument("--target-gate-count", type=int, default=2)
    parser.add_argument(
        "--policy-callable",
        default="",
        help="Optional post-FSM policy callable; gate 1 and braking remain deterministic",
    )
    parser.add_argument("--policy-activate-gate-index", type=int, default=1)
    args = parser.parse_args()

    if args.smoke_duration <= 0:
        parser.error("--smoke-duration must be positive")
    if args.target_gate_count <= 0:
        parser.error("--target-gate-count must be positive")
    if args.policy_activate_gate_index < 1:
        parser.error("--policy-activate-gate-index must be at least 1")

    report = tick(
        tag=args.tag,
        run_track_b=args.run_track_b_train,
        smoke_duration_s=args.smoke_duration,
        target_gate_count=args.target_gate_count,
        policy_callable=args.policy_callable,
        policy_activate_gate_index=args.policy_activate_gate_index,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
