#!/usr/bin/env python3
"""Track B orchestrator: PufferLib sim replication for official AI-GP transfer.

Phases:
  B1 — Transfer gate: checkpoint + policy-attitude in official sim (gate index >= 1)
  B2 — Course map: extract landmarks -> native custom gate layout
  B3 — Native training: only after B1 passes
  B4 — BC dataset: collect expert trajectories for optional student policy

State is persisted in logs/track_b/state.json between loop iterations.

Usage:
  python scripts/track_b_runner.py
  python scripts/track_b_runner.py --phase B1 --run-transfer
  python scripts/track_b_runner.py --phase B2 --smoke-glob "logs/sitl/*smoke*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "logs" / "track_b" / "state.json"
DEFAULT_CHECKPOINT_GLOB = "checkpoints/drone_race_sitl_plant*/**/*.bin"
GATE1_INI = "config/drone_race_sitl_plant_gate1.ini"
ROBUST_INI = "config/drone_race_sitl_plant_robust.ini"


def _now() -> float:
    return time.time()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "phase": "B1",
        "b1_transfer_passed": False,
        "b1_attempts": [],
        "b2_course_map_path": None,
        "b3_native_solved": False,
        "b4_bc_dataset_path": None,
        "checkpoint_path": None,
        "plant_model_path": "config/sitl_plant_defaults.json",
        "updated_unix": _now(),
    }


def save_state(state: dict) -> None:
    state["updated_unix"] = _now()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def latest_checkpoint(pattern: str = DEFAULT_CHECKPOINT_GLOB) -> str | None:
    candidates = [Path(p) for p in glob.glob(str(ROOT / pattern), recursive=True)]
    if not candidates:
        solved = ROOT / "logs" / "drone_race_sitl_plant_solved_checkpoint.txt"
        if solved.exists():
            path = solved.read_text(encoding="utf-8").strip()
            if path and Path(path).exists():
                return path
        gate1_solved = ROOT / "logs" / "drone_race_sitl_plant_gate1_solved_checkpoint.txt"
        if gate1_solved.exists():
            path = gate1_solved.read_text(encoding="utf-8").strip()
            if path and Path(path).exists():
                return path
        return None
    best = max(candidates, key=lambda p: (int(p.parent.name), int(p.stem)))
    return str(best.resolve())


def run_cmd(cmd: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print(f"==> {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=str(ROOT), env=merged, check=False, capture_output=True, text=True)


def refine_plant_from_smokes(smoke_glob: str) -> dict:
    smokes = sorted(glob.glob(smoke_glob, recursive=True))
    if not smokes:
        return {"status": "skipped", "reason": "no smoke JSON for plant identification"}

    best = max(smokes, key=lambda p: os.path.getmtime(p))
    out_path = ROOT / "logs" / "track_b" / "plant_model.json"
    proc = run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "identify_sitl_plant.py"),
            "--smoke-json",
            best,
            "--output",
            str(out_path),
            "--apply-ini",
            str(ROOT / "config" / "drone_race_sitl_plant.ini"),
        ]
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "smoke_json": best,
        "plant_model_path": str(out_path),
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def run_transfer_validation(
    checkpoint: str,
    *,
    tag: str = "track_b_b1",
    n_repeats: int = 1,
) -> dict:
    env = {"PUFFER_POLICY_CHECKPOINT_PATH": checkpoint}
    proc = run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_official_policy_validation.py"),
            "--n-repeats",
            str(n_repeats),
            "--tag",
            tag,
            "--control-mode",
            "policy-attitude",
            "--policy-callable",
            "scripts/policy_callable_checkpoint.py:infer",
            "--send-sim-reset",
            "--post-reset-sleep-s",
            "5",
            "--require-official-race-progress",
            "--min-official-gate-index",
            "1",
            "--smoke-duration",
            "45",
        ],
        env=env,
    )
    summary_path = ROOT / "logs" / "sitl" / f"official_policy_validation_summary_{tag}.json"
    payload: dict = {
        "exit_code": proc.returncode,
        "checkpoint_path": checkpoint,
        "stdout": proc.stdout[-3000:],
        "stderr": proc.stderr[-3000:],
    }
    if summary_path.exists():
        payload.update(json.loads(summary_path.read_text(encoding="utf-8")))
    payload["gate_progress_passed"] = bool(payload.get("passes", 0) >= 1)
    return payload


def run_b2_course_map(smoke_glob: str, *, min_official_gates: int = 1) -> dict:
    proc = run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_official_course_map.py"),
            "--glob",
            smoke_glob,
            "--min-official-gates",
            str(min_official_gates),
        ]
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "course_map_path": str(ROOT / "config" / "official_course_r1.json"),
        "ini_path": str(ROOT / "config" / "drone_race_sitl_plant_official.ini"),
    }


def run_b4_bc_collect(smoke_glob: str) -> dict:
    out = ROOT / "logs" / "track_b" / "bc_dataset.jsonl"
    proc = run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_bc_dataset.py"),
            "--glob",
            smoke_glob,
            "--out",
            str(out),
        ]
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "dataset_path": str(out),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def suggest_wsl_train_command(env_name: str = "drone_race_sitl_plant_gate1") -> str:
    return (
        f"REPO_ROOT=~/pufferlib-drone ENV_NAME={env_name} "
        f"bash scripts/train_sitl_plant_until_solved.sh"
    )


def run_phase(state: dict, args: argparse.Namespace) -> dict:
    report: dict = {"phase": state.get("phase", "B1"), "actions": []}
    smoke_glob = args.smoke_glob or "logs/sitl/*smoke*.json"

    checkpoint = args.checkpoint or state.get("checkpoint_path") or latest_checkpoint()
    if checkpoint:
        state["checkpoint_path"] = checkpoint
        report["checkpoint_path"] = checkpoint
    else:
        report["actions"].append(
            {
                "action": "train_gate1_checkpoint",
                "status": "needed",
                "command": suggest_wsl_train_command("drone_race_sitl_plant_gate1"),
                "ini": GATE1_INI,
            }
        )

    phase = args.phase or state.get("phase", "B1")

    if phase in {"B1", "ALL"}:
        if args.refine_plant:
            plant = refine_plant_from_smokes(smoke_glob)
            report["actions"].append({"action": "refine_plant", **plant})
            if plant.get("plant_model_path"):
                state["plant_model_path"] = plant["plant_model_path"]

        if args.run_transfer:
            if not checkpoint:
                report["actions"].append(
                    {"action": "transfer_validation", "status": "blocked", "reason": "no checkpoint"}
                )
            else:
                attempt = run_transfer_validation(
                    checkpoint,
                    tag=args.transfer_tag,
                    n_repeats=args.n_repeats,
                )
                state["b1_attempts"].append(
                    {"unix": _now(), "tag": args.transfer_tag, **attempt}
                )
                if attempt.get("gate_progress_passed"):
                    state["b1_transfer_passed"] = True
                    state["phase"] = "B2"
                report["actions"].append({"action": "transfer_validation", **attempt})
        elif not state.get("b1_transfer_passed"):
            report["actions"].append(
                {
                    "action": "transfer_validation",
                    "status": "pending",
                    "hint": (
                        "Set PUFFER_POLICY_CHECKPOINT_PATH, start official sim RACE, then "
                        "python scripts/track_b_runner.py --run-transfer"
                    ),
                }
            )

    if phase in {"B2", "ALL"} and (state.get("b1_transfer_passed") or args.force_b2):
        b2 = run_b2_course_map(smoke_glob, min_official_gates=args.min_official_gates)
        report["actions"].append({"action": "course_map", **b2})
        if b2.get("status") == "ok":
            state["b2_course_map_path"] = b2["course_map_path"]
            if state.get("b1_transfer_passed"):
                state["phase"] = "B3"

    if phase in {"B3", "ALL"}:
        if state.get("b1_transfer_passed"):
            report["actions"].append(
                {
                    "action": "native_train",
                    "status": "ready",
                    "env_name": "drone_race_sitl_plant_official"
                    if state.get("b2_course_map_path")
                    else "drone_race_sitl_plant_gate1",
                    "command": suggest_wsl_train_command(
                        "drone_race_sitl_plant_official"
                        if state.get("b2_course_map_path")
                        else "drone_race_sitl_plant_gate1"
                    ),
                    "robust_ini": ROBUST_INI,
                }
            )
        else:
            report["actions"].append(
                {
                    "action": "native_train",
                    "status": "blocked",
                    "reason": "B1 transfer gate not passed",
                    "gate1_command": suggest_wsl_train_command("drone_race_sitl_plant_gate1"),
                }
            )

    if phase in {"B4", "ALL"}:
        b4 = run_b4_bc_collect(smoke_glob)
        report["actions"].append({"action": "bc_collect", **b4})
        if b4.get("status") == "ok":
            state["b4_bc_dataset_path"] = b4["dataset_path"]

    report["state"] = state
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["B1", "B2", "B3", "B4", "ALL"], default="ALL")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--smoke-glob", default="logs/sitl/*smoke*.json")
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--refine-plant", action="store_true")
    parser.add_argument("--force-b2", action="store_true")
    parser.add_argument("--min-official-gates", type=int, default=1)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--transfer-tag", default="track_b_b1")
    parser.add_argument("--json-out", default=str(ROOT / "logs" / "track_b" / "last_report.json"))
    args = parser.parse_args()

    state = load_state()
    report = run_phase(state, args)
    save_state(state)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.run_transfer and not report.get("state", {}).get("b1_transfer_passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
