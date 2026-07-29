# Goal: Fast hybrid RL → official AI-GP course completion

> **Status: SUPERSEDED (2026-07-13).** This file preserves the earlier hybrid
> milestone for history only. Do not execute it as the active goal. The current
> architecture is a single recurrent full-flight policy; use
> `docs/full_policy_execution_prompt.md` and `docs/course_completion_strategy.md`.

## Completion criteria

### Native gate-2 curriculum
- Stage 1: `success_rate >= 0.90`, `crash <= 0.10`.
- Stage 2: `success_rate >= 0.80`, `crash <= 0.15` on 4096 deterministic episodes.
- Checkpoint recorded in `logs/drone_race_sitl_plant_gate2_fast_solved_checkpoint.txt`.

### Official sim (AI-GP v1.0.3385)
- FSM passes gate 1 and brakes; checkpoint activates at official index 1.
- `official_active_gate_index >= 2`, then extend the curriculum gate by gate.
- MAVLink in-place reset `31000` between attempts
- Optional promotion: 30/30 passes per `config/sitl_competition_acceptance.json`

## Known fixes (2026-07-03)

- **Heap crash at epoch ~3:** `drone_race` `my_log` writes 35 keys into `create_dict(32)` — overflow corrupts heap. Fixed via `ENV_LOG_DICT_CAPACITY=64` in `src/vecenv.h`.
- **Segfault on exit with `cudagraphs=-1`:** `close_impl` destroyed uninitialized cuda graphs. Guarded with `rollout_captured` check.
- **Windows C: disk:** keep >1GB free; full disk corrupted `vecenv.h` during edit.


**Important:** run only ONE training process on WSL at a time. Concurrent `pufferl` runs cause heap corruption.

```bash
# WSL ext4 (recommended)
bash scripts/sync_wsl_ext4_repo.sh ~/pufferlib-drone
REPO_ROOT=~/pufferlib-drone bash scripts/train_sitl_native_retry.sh
REPO_ROOT=~/pufferlib-drone bash scripts/train_gate2_rl_fast_cycle.sh
```

```powershell
# Windows official sim (simulator process remains running)
.\scripts\run_windows_course_fsm.ps1 `
  -PolicyCheckpoint "path\to\solved.bin" `
  -PolicyActivateGateIndex 1 `
  -SmokeDuration 30 -TargetGateCount 2 -Tag hybrid_gate2
```

## Current evidence

- Corrected native mode-2 CUDA smoke: approximately 170–235K SPS.
- Hover-centered action smoke: crash reduced from 1.0 to about 0.24 after 1M steps; success remains 0, so this checkpoint is not promotable.
