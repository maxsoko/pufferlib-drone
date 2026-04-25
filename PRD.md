# PRD: Drone Gate Navigation Challenge (Source of Truth)

Last updated: 2026-04-24
Owner: `PufferLib/native-v4-drone-port`

## Source of Truth Policy
- This file is the active product and execution PRD for the race program.
- Legacy hover-era and historical experiment logs remain preserved in git history on the stabilized `3.0` checkpoint branch.
- If conflicts exist between this file and archived material, this file wins.

## Challenge Goal
Build a fully autonomous drone stack that passes all gates in strict order and minimizes valid completion time in virtual competition environments.

## Competition-Aligned Requirements
- Gate recognition from provided sensor/visual streams.
- Precision drone control (thrust/orientation/rates) with reliability-first safety behavior.
- Efficient planning/navigation under realistic vehicle limits.
- No manual control and no hardware advantages.

## Official Virtual Qualifier Spec Alignment
Reference: `docs/260318_Technical_Spec_0001.pdf` (`VADR-TS-001`, issue `00.01`).

Decision:
- Native PufferLib v4 remains the high-throughput training backend.
- Native `_C` execution is the required path for drone training and evaluation; `--slowly`/PyTorch backend runs are not acceptance criteria.
- Final competition integration must communicate through MAVLink v2 over UDP via a MAVSDK-compatible SITL bridge.
- Training interfaces must converge toward the official observable interface instead of relying permanently on privileged absolute state.

Spec constraints:
- Physics target rate: `120 Hz`.
- Recommended command rate: `50-120 Hz`.
- Minimum heartbeat rate: `2 Hz`.
- Round 1 maximum run duration: `8 minutes` (`480s`).
- Simulator exposes local Cartesian / navigation reference data, not GPS or global position.
- Vision is forward-facing first-person camera; detailed stream parameters are pending a separate spec.
- Supported inbound control messages include `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`.
- Relevant telemetry includes attitude/orientation, linear velocity, system status flags, and simulator navigation reference data.
- Course geometry, physics parameters, and environmental conditions are deterministic.

## Program-Level Success Metrics
- `valid_run_rate`: percentage of runs that complete all gates in order.
- `median_valid_time`: median completion time over valid runs.
- `ordered_gate_pass_rate`: percentage of accepted in-sequence gate crossings.
- `crash_rate`: fraction of runs that terminate via crash/unsafe state.

## Architecture Decision (Active)
Use a native v4 training stack with an explicit competition adapter boundary:
1. Native training envs: `ocean/drone` for hover/control prior and `ocean/drone_race` for race progression.
2. Perception interface: telemetry + vision inputs only for competition-facing policy paths.
3. Planner/controller interface: emit local setpoints or attitude targets compatible with MAVLink command messages.
4. SITL bridge: translate model/controller outputs to MAVSDK-compatible UDP MAVLink commands.
5. Deterministic evaluation: keep fixed-seed native smoke/eval harnesses for regression before SITL testing.
6. Edge inference: keep the default policy small enough for deterministic onboard inference and add quantized PufferNet runtimes beside the float path.

## Edge Inference and Quantization Track
Decision:
- Start with quantized, mixed-precision `PufferNet` inference for the existing `DefaultEncoder + MinGRU + DefaultDecoder` policy.
- Do not introduce Transformer policies until native state-vector control is reliable and camera/vision requirements justify the added complexity.
- Treat quantization as a closed-loop control problem, not just action-MSE compression.

Initial target:
- Observations are normalized/clamped in native envs (`25` race features, `23` hover features), making them suitable for fixed-point encoding.
- Use Q8 weights with per-output-channel scales for linear layers.
- Use `int32` accumulators for low-bit MACs.
- Keep recurrent state and nonlinear gate math mixed precision initially, then tighten to Q15/Q8 only after closed-loop parity is measured.
- Keep final actions as float or high-resolution fixed point until actuator command tolerances are known.

Validation metrics:
- Float vs Q8 inference latency.
- Float vs Q8 action drift.
- Hover score / gate success rate.
- Crash and invalid-run rates.
- Worst-case inference latency under recurrent state.

Implementation surface:
- `src/puffernet.h`: canonical float C inference path.
- `src/puffernet_q8.h`: Q8/mixed-precision inference path.
- `tests/bench_puffernet_edge.c`: deterministic float-vs-Q8 latency and drift benchmark.

## Scope Boundaries
In scope:
- `ocean/drone/*` for native hover/control-prior benchmarking.
- `ocean/drone_race/*` for native race dynamics and reward/metric logging.
- `config/drone.ini` and `config/drone_race.ini`.
- `scripts/eval_drone_hover_native.py` and `scripts/eval_drone_race_native.py`.
- Future MAVLink/SITL adapter code and submission harness.

Out of scope for active race implementation:
- Rebuilding the old v3 Python `pufferlib.environments.drone_race` runtime.
- Hardware flight implementation.

## Training Backbone and Governance Model
Decision:
- Use native v4 `drone` hover/control training as the stability prior.
- Use native v4 `drone_race` race training after hover/control smoke metrics are stable.
- Add L1-L9 as promotion governance gates over native evaluation outputs and later SITL outputs.

### Tier Backbone
- H0: native `drone` hover/control prior.
- R1: native `drone_race` short-course reliability.
- R2: native `drone_race` full-duration reliability.
- S1: SITL/MAVLink adapter smoke with telemetry-only control.
- S2: SITL/MAVLink + vision/perception integration.

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
1. Reconfirm native hover/control stability first.
2. Train native race only after hover/control smoke remains non-crashing and learnable.
3. Keep native race timing aligned with `120 Hz` and `480s` qualifier duration.
4. Add MAVLink/SITL bridge before treating any result as competition-representative.
5. Optimize speed only after sustained valid completion.

## Strong and Reusable Patterns (Adopted)
From cross-project analysis, we adopt these patterns:
- Strict level gating with explicit thresholds and budgets.
- Gate-geometry helper as a single tested source for crossing validity.
- Deterministic submission-style local harness behavior.
- Domain-randomization/noise profiles as progressive robustness knobs.

## Extended Implementation Plan

### Phase A: Native v4 Baseline
- [x] Import upstream PufferLib v4 core on `native-v4-drone-port`.
- [x] Add native `ocean/drone_race` C environment and binding.
- [x] Add native hover/race smoke eval helpers.
- [x] Validate native CPU build and native eval smoke for local correctness.
- [x] Add Linux CUDA/NCCL native validation runner (`scripts/validate_native_drone_gpu.sh`).
- [ ] Validate CUDA native build on a CUDA/NCCL-capable machine.

### Phase B: Qualifier Timing and Metrics
- [x] Align native `drone_race` config with `120 Hz` physics target and `480s` max run.
- [x] Add dedicated native gate-crossing regression tests or standalone C/Python parity harness.
- [x] Replace race kinematic shortcut with representative quadrotor motor/RK4 dynamics.
- [ ] Emit native eval report artifacts (JSON + CSV) per checkpoint.

### Phase C: MAVLink/SITL Adapter
- [x] Add MAVLink client scaffold for UDP MAVLink v2.
- [x] Maintain heartbeat at `>=2 Hz`.
- [x] Send `SET_POSITION_TARGET_LOCAL_NED` and/or `SET_ATTITUDE_TARGET` at `50-120 Hz`.
- [ ] Parse attitude, orientation, velocity, status, and simulator navigation reference telemetry.
- [ ] Define a model/controller output contract that maps to MAVLink setpoints.

### Phase D: Vision/Perception Path
- [ ] Add forward-camera ingestion once the separate vision stream specification is available.
- [ ] Keep privileged/native state available only for training diagnostics and teacher signals.
- [ ] Train/evaluate telemetry-only and telemetry+vision variants separately.

### Phase E: Submission Harness
- [ ] Add deterministic submission entrypoint for SITL.
- [ ] Add local SITL eval runner with fixed course/start conditions.
- [ ] Record no-human-interaction compliance assumptions.

### Phase F: Edge Inference
- [x] Benchmark native float `forward_puffernet` latency on local CPU.
- [x] Add Q8/mixed-precision PufferNet runtime.
- [x] Add float-vs-Q8 latency and action-drift benchmark.
- [x] Add exporter for trained float weights to packed quantized weights.
- [x] Add closed-loop hover comparison harness for trained `.bin` checkpoints.
- [ ] Add fake quantization during native RL fine-tuning.
- [ ] Promote Q8 only on closed-loop hover/race metrics, not action drift alone.

## Immediate Sprint (Current)
- [x] Preserve v3 hover/race checkpoint on `3.0`.
- [x] Create native v4 port branch.
- [x] Add native race and hover smoke/eval path.
- [x] Align native race timing to official qualifier constraints.
- [x] Add Q8 PufferNet edge benchmark and runtime scaffold.
- [x] Implement MAVLink/SITL adapter scaffold.
- [x] Add native gate-crossing parity/regression tests.
- [ ] Run native CUDA/NCCL validation on Linux GPU hardware.

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
- Risk: native training overfits privileged state and fails the official observable interface.
  - Control: keep MAVLink telemetry/vision adapter as an explicit promotion gate.
- Risk: v4 native CUDA build cannot be validated on local macOS.
  - Control: use native CPU build/eval only for local correctness; run CUDA/NCCL native training on Linux GPU before trusting throughput claims.
- Risk: timing mismatch versus qualifier simulator.
  - Control: use `120 Hz` / `480s` defaults in native race config and SITL tests.
- Risk: old v3 hover checkpoint is not weight-compatible with v4 native models.
  - Control: use v3 hover results as teacher/reference metrics, not as direct v4 checkpoint input.
- Risk: quantized inference passes open-loop action checks but destabilizes closed-loop flight.
  - Control: require hover/race closed-loop parity, crash-rate checks, and worst-case latency checks before using Q8 for competition-facing control.

## Hover Legacy Snapshot (Reference Only)
- Role in active program:
  - The old v3 hover work remains the strongest evidence of stable control behavior.
  - Native v4 `drone` hover is now the control-prior benchmark before native `drone_race`.
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
  - v3 `.pt` checkpoints are not assumed compatible with v4 native `.bin`/model layouts.
  - Use old hover behavior as teacher/reference data or benchmark, not a direct weight load path.

## Current Program Status
- Branch `3.0` contains the stabilized v3 checkpoint and historical Python race/hover work.
- Branch `native-v4-drone-port` contains the native v4 migration.
- Native `drone` hover CPU build and native eval smoke pass locally.
- Native `drone_race` now uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate actions.
- Native `drone_race` CPU build, eval smoke, and gate-crossing/physics/timing regressions pass locally.
- Q8 PufferNet benchmark and closed-loop hover comparison harness compile locally; real checkpoint comparison is pending a native `.bin` checkpoint.
- MAVLink/SITL scaffold dry-run passes locally.
- `--slowly`/PyTorch backend runs are considered debug-only and are not part of the accepted training path.
- Native CUDA build is blocked locally by missing Linux CUDA/NVCC; run `scripts/validate_native_drone_gpu.sh` on Linux GPU hardware.
- Next required implementation: native CUDA/NCCL training validation, telemetry parsing in the SITL adapter, and fake quantization during RL fine-tuning.
