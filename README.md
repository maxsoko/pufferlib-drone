# PufferLib Drone Racing (AI Grand Prix 2026)

Autonomous drone racing stack built on PufferLib for the AI Grand Prix challenge.

**Goal**: Navigate a drone through ordered gates as fast as possible with fully autonomous control.

## Challenge Focus

Your autonomy stack must handle three pillars:

1. **Gate recognition** from provided sensor and visual feeds.
2. **Drone control** (speed, orientation, thrust/rates) with reliability under dynamics limits.
3. **Path planning and navigation** through all gates in strict sequence.

Primary scoreboard metric is **fastest valid completion time**.

## Current Scope

This repo currently uses a race-first simulation program in `PufferLib`:

- Race environment: `pufferlib/environments/drone_race/`
- Race configs: `pufferlib/config/drone_race*.ini`
- Eval scripts: `scripts/eval_drone_race.py`, `scripts/eval_drone_race_levels.py`
- Visualizer: `scripts/visualize_drone_race.py`
- Program PRD: `PRD.md`

`drone_hover` remains useful for control pretraining and regression checks.

## Architecture

```text
PufferLib/
├── pufferlib/environments/drone_race/
│   ├── environment.py      # race dynamics, rewards, gate validity, episode lifecycle
│   ├── adapters.py         # perception adapter layer
│   ├── planner.py          # gate setpoint planning + fallback behavior
│   ├── gate_progress.py    # shared gate crossing geometry helper
│   ├── contracts.py        # perception/planner contract structures
│   └── torch.py            # race policy definitions
├── pufferlib/config/
│   ├── drone_race.ini
│   ├── drone_race_curriculum.ini
│   ├── drone_race_curriculum_tier1.ini
│   ├── drone_race_curriculum_tier2.ini
│   ├── drone_race_curriculum_tier3.ini
│   └── drone_race_levels.ini
├── scripts/
│   ├── eval_drone_race.py
│   ├── eval_drone_race_levels.py
│   └── visualize_drone_race.py
├── tests/
│   ├── test_drone_race_*.py
│   ├── test_eval_drone_race*.py
│   └── test_policy_warmstart.py
└── PRD.md
```

## Curriculum Strategy

Training backbone:

- `drone_race_curriculum_tier1`
- `drone_race_curriculum_tier2`
- `drone_race_curriculum_tier3`

Governance gates (L1-L9) are defined in `pufferlib/config/drone_race_levels.ini` and evaluated with `scripts/eval_drone_race_levels.py`.

Promotion policy: pass current level thresholds before promotion; otherwise run remediation and retry.

## Quick Start

### 1) Install

```bash
cd PufferLib
pip install -e .
```

Optional native extension build:

```bash
python setup.py build_ext --inplace
```

### 2) Train Race Policy (Tier 1)

```bash
python -m pufferlib.pufferl train drone_race_curriculum_tier1 --csv-log-dir experiments
```

### 3) Warmstart from Hover Pretraining (Recommended)

Use hover checkpoint as representation/control bootstrap for race:

```bash
python -m pufferlib.pufferl train drone_race_curriculum_tier1 \
  --warmstart-model-path experiments/<best_hover_model>.pt \
  --warmstart-prefixes encoder \
  --csv-log-dir experiments
```

Notes:

- Use `--warmstart-model-path` for cross-task transfer (hover -> race).
- Use `--load-model-path` for same-task resume/eval.

### 4) Evaluate a Checkpoint

```bash
python scripts/eval_drone_race.py \
  --mode policy \
  --model-path experiments/<race_model>.pt \
  --suite drone_race_v1_quick20 \
  --deterministic
```

### 5) Evaluate L1-L9 Level Gates

```bash
python scripts/eval_drone_race_levels.py \
  --mode policy \
  --model-path experiments/<race_model>.pt \
  --levels-config pufferlib/config/drone_race_levels.ini
```

### 6) Visualize a Rollout

Install optional visualization deps first:

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate
```

Run scripted baseline:

```bash
python scripts/visualize_drone_race.py --mode scripted
```

Run trained policy:

```bash
python scripts/visualize_drone_race.py --mode policy --model-path experiments/<race_model>.pt --deterministic
```

## Visualizer Controls

- Mouse drag: orbit camera
- Scroll: zoom
- `W/A/S/D`: pan
- `Space`: pause/resume
- `R`: reset
- `T`: toggle trails
- `+/-`: simulation speed
- `Q` or `Esc`: quit

## Program Status (2026-02-20)

Implemented:

- Modular race stack (perception adapter + planner + control policy path).
- Shared tested gate crossing helper (`gate_progress.py`).
- Fixed-seed checkpoint evaluation and L1-L9 governance evaluation.
- Hover -> race warmstart support in `pufferlib.pufferl`.
- Interactive OpenGL race visualizer.

In progress:

- Tier promotion runs and metric logging on GPU rig.
- Extended remediation and submission-style deterministic harness workflow.

## Tests

Focused race + warmstart tests:

```bash
python -m pytest tests/test_eval_drone_race_levels.py tests/test_drone_race_gate_progress.py tests/test_policy_warmstart.py -q
```

Broader race suite:

```bash
python -m pytest tests/test_drone_race_configs.py tests/test_drone_race_env.py tests/test_drone_race_planner.py tests/test_drone_race_adapters.py tests/test_eval_drone_race.py -q
```

## Docs

- Source-of-truth PRD: `PRD.md`
- Legacy PRD archive: `docs/prd_archive_legacy_2026-02-20.md`
- GPU checklist: `docs/gpu-training-checklist.md`
- PufferLib leverage checklist: `docs/use-more-of-pufferlib-checklist.md`

## Ralph Loop

For iterative PRD-driven Codex execution:

```bash
./scripts/ralph-loop.sh
```

Set `RALPH_UNSAFE=1` only if you explicitly intend bypass mode.
