# PRD: Drone Gate Navigation Challenge (Source of Truth)

Last updated: 2026-04-23
Owner: `PufferLib/drone_race`

## Source of Truth Policy
- This file is the active product and execution PRD for the race program.
- Legacy hover-era and historical experiment logs are archived at:
  - `docs/prd_archive_legacy_2026-02-20.md`
- If conflicts exist between this file and archived material, this file wins.

## Challenge Goal
Build a fully autonomous drone stack that passes all gates in strict order and minimizes valid completion time in virtual competition environments.

## Competition-Aligned Requirements
- Gate recognition from provided sensor/visual streams.
- Precision drone control (thrust/orientation/rates) with reliability-first safety behavior.
- Efficient planning/navigation under realistic vehicle limits.
- No manual control and no hardware advantages.

## Program-Level Success Metrics
- `valid_run_rate`: percentage of runs that complete all gates in order.
- `median_valid_time`: median completion time over valid runs.
- `ordered_gate_pass_rate`: percentage of accepted in-sequence gate crossings.
- `crash_rate`: fraction of runs that terminate via crash/unsafe state.

## Architecture Decision (Active)
Use a modular race stack:
1. Perception adapters (`privileged_state`, camera/sensor adapters).
2. Planner producing bounded setpoints and fallback behavior.
3. Controller path (target: dedicated module with saturation/fallback).
4. PPO training and deterministic fixed-seed evaluation in PufferLib.

## Scope Boundaries
In scope:
- `pufferlib/environments/drone_race/*`
- `pufferlib/config/drone_race*.ini`
- `scripts/eval_drone_race*.py`
- `tests/test_drone_race*.py`

Out of scope for active race implementation:
- Expanding `drone_hover` features unless needed for shared-core bug fixes.
- Hardware flight implementation.

## Training Backbone and Governance Model
Decision:
- Keep tier configs as the training backbone.
- Add L1-L9 as promotion governance gates over evaluation outputs.

### Tier Backbone
- Tier 1: `drone_race_curriculum_tier1`
- Tier 2: `drone_race_curriculum_tier2`
- Tier 3: `drone_race_curriculum_tier3`

### L1-L9 Governance
- L-levels do not replace tier configs.
- L-levels decide promotion, remediation, and checkpoint acceptance.
- Remediation policy: max 3 attempts per level, targeted remediation between attempts.

## L1-L9 Definitions (Operational)
| Level | Intent | Primary Metric | Gate |
|---|---|---|---|
| L1 | Stable launch/hold behavior | `valid_run_rate` (easy setup) | >= 0.90 |
| L2 | Reach first gate reliably | `first_gate_pass_rate` | >= 0.90 |
| L3 | Pass one gate reliably | `first_gate_pass_rate` | >= 0.95 |
| L4 | Two-gate sequencing reliability | `two_gate_pass_rate` | >= 0.90 |
| L5 | Sequence robustness (turning pressure) | `success_rate` (tier2) | >= 0.80 |
| L6 | Full lap reliability | `success_rate` (tier3) | >= 0.60 |
| L7 | Speed optimization after reliability | `median_completion_time_valid` | <= target time |
| L8 | Consistency | `max_consecutive_successes` | >= 10 |
| L9 | Generalization | `success_rate` on strict suite | >= 0.50 |

Notes:
- L7 target time remains configurable per track and evaluation suite.
- L9 generalization currently uses strict fixed-seed suite until multi-track interfaces land.

## Execution Strategy
1. Reliability first on Tier 1 -> Tier 2 -> Tier 3.
2. Promote only if current L-gate passes.
3. On failure, run targeted remediation and retry.
4. Optimize speed only after sustained valid completion.

## Strong and Reusable Patterns (Adopted)
From cross-project analysis, we adopt these patterns:
- Strict level gating with explicit thresholds and budgets.
- Gate-geometry helper as a single tested source for crossing validity.
- Deterministic submission-style local harness behavior.
- Domain-randomization/noise profiles as progressive robustness knobs.

## Extended Implementation Plan

### Phase A: Governance Layer
- [ ] Add level spec config: `pufferlib/config/drone_race_levels.ini`.
- [ ] Add level evaluator script: `scripts/eval_drone_race_levels.py`.
- [ ] Emit level report artifacts (JSON + CSV) per checkpoint.

### Phase B: Gate Geometry Hardening
- [ ] Add shared helper: `pufferlib/environments/drone_race/gate_progress.py`.
- [ ] Route env crossing checks through helper.
- [ ] Add dedicated gate-progress tests.

### Phase C: Remediation Loop
- [ ] Add iterative trainer script that enforces L-gates and retry policy.
- [ ] Implement targeted remediation hooks (seed subsets + teacher/scripted warmup).

### Phase D: Deterministic Submission Harness
- [ ] Add `submission/entrypoint.py` for deterministic inference path.
- [ ] Add `submission/local_eval.py` for local controlled evaluation.

### Phase E: Robustness Profiles
- [ ] Add clean/mild/medium/harsh race noise presets in config.
- [ ] Wire profile progression to level policy.

## Immediate Sprint (Current)
- [ ] Refactor PRD to source-of-truth structure (this update).
- [ ] Implement Phase A and Phase B with focused tests.
- [ ] Keep tier configs as training backbone and L-levels as governance.

## Promotion and Remediation Policy
- Promotion requires passing all required metrics for the current level.
- If failed:
  1. Run diagnostics over fixed-seed suite.
  2. Apply targeted remediation.
  3. Retry.
- Max attempts per level: 3.

## Experiment Logging Requirements
For each training/eval block, record:
- Checkpoint path and config used.
- Suites evaluated and seed ranges.
- L-level pass/fail outcomes and blocking metric.
- Commands run and status.

## Risks and Controls
- Risk: speed tuning before reliability leads to invalid-run collapse.
  - Control: enforce L6/L8 gates before aggressive speed optimization.
- Risk: interface drift in gate-crossing logic.
  - Control: shared helper + dedicated tests.
- Risk: overfitting to privileged-state perception.
  - Control: staged camera/noise profile adoption and fallback diagnostics.

## Hover Legacy Snapshot (Reference Only)
- Role in active program:
  - `drone_hover` remains a regression/safety baseline and optional warmstart source.
  - Active optimization surface remains `drone_race`.
- Canonical historical record:
  - `docs/prd_archive_legacy_2026-02-20.md` (full hover R&D timeline and run log).
- Best robust checkpoint lineage (historical):
  - `experiments/drone_hover_robust_176940098863.pt`
  - `experiments/drone_hover_robust_177049920214.pt` (post curriculum-counter persistence fix).
- Recorded robust evaluation outcomes (fixed seeds, deterministic):
  - Wind `0.3`, gains `kp_xy=3.8`, `kd_xy=3.5`: `100/100` success, mean hover `30.000s`.
  - Wind `0.4`, gains `kp_xy=3.8`, `kd_xy=3.5`: `90/100` success, mean hover `27.380s`.
  - Wind `0.4`, gains `kp_xy=4.2`, `kd_xy=3.9`: `94/100` success, mean hover `28.299s`.
- Operational caveat:
  - Best historical performance depends on PD-assisted residual control (`pd_assist=True`, `pd_assist_scale=0.02`, `residual_penalty=0.5`).
  - No-assist hover policies under randomized settings underperformed.
- Current workspace note:
  - Hover CSV/log artifacts are present under `experiments/`, but historical `.pt` checkpoints may be absent locally and may need regeneration.
- Race handoff usage:
  - If a hover checkpoint is regenerated/restored, use `--warmstart-model-path` with `--warmstart-prefixes encoder` for hover -> race transfer.

## Current Program Status
- Race env/planner/perception-adapter baseline implemented.
- Tier configs in place and updated for MP/GPU + recurrent track.
- Next required implementation: L-gate evaluator and shared gate-progress helper.
