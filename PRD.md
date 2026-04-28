# PRD: Drone Gate Navigation Challenge (Source of Truth)

Last updated: 2026-04-28
Owner: `PufferLib/native-v4-drone-port`

## Source of Truth Policy

- This file is the active product and execution PRD for the race program.
- Legacy hover-era and historical experiment logs remain preserved in git history on the stabilized `3.0` checkpoint branch.
- If conflicts exist between this file and archived material, this file wins.

## Challenge Goal

Build a fully autonomous drone stack that passes all gates in strict order and minimizes valid completion time in virtual competition environments.

## Current Completion Snapshot

Completed:

- `native-v4-drone-port` is now the GitHub default branch and active source of truth.
- Native v4 baseline exists for `drone` and `drone_race`, including C bindings, CPU smoke/eval paths, local regression checks, and Linux CUDA/NCCL validation on Vast.ai.
- `drone_race` uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate kinematics.
- Race and hover observations are aligned at `23` features, enabling hover-to-race native checkpoint warm starts.
- Staged native race curriculum exists for H1, R1, R3, and R4, with promotion checks that stop on failed thresholds.
- Reproducible checkpoint eval artifacts are supported via `scripts/eval_drone_race_checkpoint.py` with paired JSON/CSV outputs.
- Useful native checkpoints exist through H1 and R1; R3 has a reproducible partial-success checkpoint but is not promotion-ready.
- Q8/mixed-precision PufferNet inference scaffolding, export, latency/drift benchmarking, and closed-loop hover comparison harnesses exist.
- MAVLink/SITL adapter scaffold and local dry-run path exist.

Not complete / blocking:

- R3 three-gate stability is still below promotion thresholds; latest best reproducible R3 eval is `success_rate=0.8629`, `crash=0.1371`, `gates_passed=2.6843`.
- R4/full-course race promotion is blocked until R3 reliability improves.
- Current native policies still train on privileged native state; qualifier-facing policy/control must move toward official telemetry plus camera observations.
- MAVLink telemetry parsing and the final model/controller output contract are not implemented.
- Local SITL evaluation with fixed course/start conditions is not implemented.
- Vision ingestion is not implemented and is waiting on the detailed camera stream spec.
- Q8/fake-quant training is not promoted; closed-loop race parity and action drift remain unresolved.

Blocking work breakdown:

- R3 native race reliability:
  - Current state: best reproducible default-native R3 eval is `success_rate=0.8629`, `crash=0.1371`, `gates_passed=2.6843`.
  - Next action: run targeted R3 remediation from the strong R1 checkpoint and/or best reproducible R3 checkpoint, changing one curriculum variable at a time.
  - Candidate knobs: crash-height shaping, control penalty, continuation length, and fixed-seed eval coverage. Recent probes improved success but crash remains the blocker.
  - Immediate next probes:
    - Start from `checkpoints/drone_race/1777415131824/0000000039387136.bin`.
    - Test slightly more forgiving `crash_height` first; goal is to separate low-altitude clipping from genuine navigation failure.
    - If crash drops without losing success, tighten `crash_height` back toward the current course setting.
    - If crash remains high, inspect terminal-state distributions before further reward tuning.
  - Acceptance signal: deterministic JSON/CSV eval reaches the R3 promotion threshold with `success_rate >= 0.85` and `crash <= 0.10`.

- R4/full-course promotion:
  - Current state: blocked; direct 4-gate and weak 3-gate continuations have regressed.
  - Next action: do not run R4 as a promotion attempt until R3 passes deterministic eval. Use R4 only for diagnostic probes if explicitly labeled as non-promotable.
  - Acceptance signal: R4 resumes only from a promoted R3 checkpoint, with checkpoint lineage and eval artifacts recorded.

- Privileged native-state dependency:
  - Current state: native `drone_race` is still a training environment, not a qualifier-equivalent interface.
  - Next action: define a telemetry-shaped observation contract that maps native state into the fields available through MAVLink/SITL telemetry, then train/evaluate telemetry-only variants before adding camera inputs.
  - Acceptance signal: telemetry-only native/SITL policy path runs without privileged absolute state dependencies.

- MAVLink telemetry and controller contract:
  - Current state: heartbeat and setpoint scaffolding exist, but inbound telemetry parsing and final action mapping are incomplete.
  - Next action: parse attitude/orientation, local velocity, status flags, and simulator navigation reference data; define whether the model outputs local position/velocity/yaw setpoints, attitude targets, or a controller-facing intermediate action.
  - Acceptance signal: SITL bridge can log telemetry, emit accepted commands at `50-120 Hz`, and produce a deterministic run report with command-rate/dropout metrics.

- Local SITL eval runner:
  - Current state: dry-run scaffolding exists, but no fixed-course evaluator gates promotion.
  - Next action: add a local entrypoint that starts or connects to the simulator, applies fixed course/start conditions, runs a policy/controller, and writes JSON/CSV results.
  - Acceptance signal: one-command SITL smoke/eval produces success, ordered gate passes, completion time, crash/invalid-run status, command rates, and telemetry dropout fields.

- Vision ingestion:
  - Current state: waiting on the detailed camera stream spec.
  - Next action: keep the native/state-vector policy path moving while reserving a perception boundary for forward-camera frames. Avoid coupling core control training to guessed camera parameters.
  - Acceptance signal: once stream details are available, add a camera adapter and separate telemetry-only vs telemetry+vision eval tracks.

- Q8/fake-quant promotion:
  - Current state: Q8 export/runtime and benchmarks exist, but action drift and closed-loop race parity are not promotion-ready.
  - Next action: defer fake-quant RL until FP32 R3/R4 behavior is reliable; then add QAT or post-training calibration against closed-loop hover/race metrics.
  - Acceptance signal: Q8 policy matches FP32 closed-loop success/crash metrics within agreed tolerance and meets worst-case latency requirements.

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
- Native `drone_race` success is a training milestone only; it is not a qualifier-equivalent result until the policy/controller runs through the SITL/MAVLink interface.

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

Current alignment assessment:

- Status: correct backbone, not qualifier-ready yet.
- Aligned:
  - Native v4 `_C` is the required high-throughput training/eval backend.
  - `drone_race` now uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate kinematics.
  - Native race timing is set around the official `120 Hz` physics target and `480s` maximum run.
  - MAVLink v2/UDP bridge scaffolding exists for heartbeat and setpoint commands at the required rates.
  - The program is reliability-first: speed optimization is blocked until valid-run stability improves.
- Not yet compliant:
  - Current policies still train on native privileged state vectors; competition-facing policies must converge to telemetry plus forward-camera observations.
  - Telemetry parsing for attitude/orientation, velocity, status, and navigation reference data is not implemented.
  - The policy/controller output contract to official MAVLink setpoints is not finalized.
  - Vision ingestion is waiting on the separate camera stream spec and has not been integrated.
  - Current native race policy is not yet stable enough for full-course promotion.

Qualifier acceptance boundary:

- A run is qualifier-representative only if it:
  - communicates with the simulator over MAVLink v2/UDP through the SITL bridge;
  - maintains heartbeat at `>=2 Hz`;
  - emits accepted setpoint/control messages at `50-120 Hz`;
  - consumes official telemetry/observable streams rather than native privileged state;
  - respects the `480s` Round 1 maximum duration;
  - records deterministic report artifacts with success, ordered gate passes, completion time, crash/invalid-run status, command rates, and any telemetry dropouts.
- Native `_C` race metrics remain necessary for training throughput and regression, but are not sufficient for competition acceptance.

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

- Observations are normalized/clamped in native envs (`23` race features, `23` hover features), making them suitable for fixed-point encoding and hover-to-race checkpoint warm starts.
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


| Level | Intent                                 | Primary Metric                 | Gate           |
| ----- | -------------------------------------- | ------------------------------ | -------------- |
| L1    | Stable launch/hold behavior            | `valid_run_rate` (easy setup)  | >= 0.90        |
| L2    | Reach first gate reliably              | `first_gate_pass_rate`         | >= 0.90        |
| L3    | Pass one gate reliably                 | `first_gate_pass_rate`         | >= 0.95        |
| L4    | Two-gate sequencing reliability        | `two_gate_pass_rate`           | >= 0.90        |
| L5    | Sequence robustness (turning pressure) | `success_rate` (tier2)         | >= 0.80        |
| L6    | Full lap reliability                   | `success_rate` (tier3)         | >= 0.60        |
| L7    | Speed optimization after reliability   | `median_completion_time_valid` | <= target time |
| L8    | Consistency                            | `max_consecutive_successes`    | >= 10          |
| L9    | Generalization                         | `success_rate` on strict suite | >= 0.50        |


Notes:

- L7 target time remains configurable per track and evaluation suite.
- L9 generalization currently uses strict fixed-seed suite until multi-track interfaces land.

## Execution Strategy

1. Reconfirm native hover/control stability first.
2. Train native race only after hover/control smoke remains non-crashing and learnable.
3. Keep native race timing aligned with `120 Hz` and `480s` qualifier duration.
4. Add MAVLink/SITL bridge before treating any result as competition-representative.
5. Optimize speed only after sustained valid completion.

Current priority order:

1. Improve 3-gate native race stability before any 4-gate/full-course promotion.
2. Implement SITL telemetry parsing and define the model/controller output contract to `SET_POSITION_TARGET_LOCAL_NED` and/or `SET_ATTITUDE_TARGET`.
3. Add a local SITL eval runner with fixed course/start conditions.
4. Add vision ingestion when the official camera stream spec is available.
5. Continue Q8/fake-quant work only after FP32 closed-loop behavior is reliable.

## Strong and Reusable Patterns (Adopted)

From cross-project analysis, we adopt these patterns:

- Strict level gating with explicit thresholds and budgets.
- Gate-geometry helper as a single tested source for crossing validity.
- Deterministic submission-style local harness behavior.
- Domain-randomization/noise profiles as progressive robustness knobs.

## Extended Implementation Plan

### Phase A: Native v4 Baseline

- Import upstream PufferLib v4 core on `native-v4-drone-port`.
- Add native `ocean/drone_race` C environment and binding.
- Add native hover/race smoke eval helpers.
- Validate native CPU build and native eval smoke for local correctness.
- Add Linux CUDA/NCCL native validation runner (`scripts/validate_native_drone_gpu.sh`).
- Validate CUDA native build on a CUDA/NCCL-capable machine.

### Phase B: Qualifier Timing and Metrics

- Align native `drone_race` config with `120 Hz` physics target and `480s` max run.
- Add dedicated native gate-crossing regression tests or standalone C/Python parity harness.
- Replace race kinematic shortcut with representative quadrotor motor/RK4 dynamics.
- Add hover-compatible race observation contract and staged first-gate to full-race curriculum runner.
- Emit native eval report artifacts (JSON + CSV) per checkpoint.
- Improve 3-gate race reliability to promotion threshold.
- Promote to 4-gate/full-course curriculum only after R3 passes deterministic eval.

### Phase C: MAVLink/SITL Adapter

- Add MAVLink client scaffold for UDP MAVLink v2.
- Maintain heartbeat at `>=2 Hz` in scaffold/dry-run.
- Send `SET_POSITION_TARGET_LOCAL_NED` and/or `SET_ATTITUDE_TARGET` at `50-120 Hz` in scaffold/dry-run.
- Parse attitude, orientation, velocity, status, and simulator navigation reference telemetry.
- Define a model/controller output contract that maps to MAVLink setpoints.

### Phase D: Vision/Perception Path

- Add forward-camera ingestion once the separate vision stream specification is available.
- Keep privileged/native state available only for training diagnostics and teacher signals.
- Train/evaluate telemetry-only and telemetry+vision variants separately.

### Phase E: Submission Harness

- Add deterministic submission entrypoint for SITL.
- Add local SITL eval runner with fixed course/start conditions.
- Record no-human-interaction compliance assumptions.

### Phase F: Edge Inference

- Benchmark native float `forward_puffernet` latency on local CPU.
- Add Q8/mixed-precision PufferNet runtime.
- Add float-vs-Q8 latency and action-drift benchmark.
- Add exporter for trained float weights to packed quantized weights.
- Add closed-loop hover comparison harness for trained `.bin` checkpoints.
- Add fake quantization during native RL fine-tuning.
- Promote Q8 only on closed-loop hover/race metrics, not action drift alone.

## Immediate Sprint (Current)

- Preserve v3 hover/race checkpoint on `3.0`.
- Create native v4 port branch.
- Make `native-v4-drone-port` the GitHub default branch.
- Add native race and hover smoke/eval path.
- Align native race timing to official qualifier constraints.
- Add Q8 PufferNet edge benchmark and runtime scaffold.
- Implement MAVLink/SITL adapter scaffold.
- Add native gate-crossing parity/regression tests.
- Run native CUDA/NCCL validation on Linux GPU hardware.
- Add staged `drone_race` curriculum after full race collapsed at `crash=1.000`.
- Add promotion checks and an explicit 3-gate bridge to the race curriculum runner.
- Add deterministic native checkpoint eval reports.
- Stabilize R3 to promotion threshold.
- Resume R4 only after R3 passes deterministic eval.
- Implement SITL telemetry parsing and controller output contract.
- Add local SITL eval runner.

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
- Native checkpoint eval artifacts should be produced with `scripts/eval_drone_race_checkpoint.py` and saved as paired JSON/CSV files under `logs/`.

## Current Vast GPU Run Log

- Vast workspace path: `/root/pufferlib-drone`.
- Active branch/commit used for these runs: `native-v4-drone-port` at `9ec46207` (`add drone race curriculum warm start`).
- 2026-04-28 Vast restart:
  - New RTX 4090 instance: `35774236`.
  - Direct SSH used by the local workstation: `root@89.221.67.131 -p 16604`.
  - Remote workspace restored to `/root/pufferlib-drone`.
  - Local backup mirror created on 2026-04-28: `/Users/anon/code/rl/puffer_ai/vast_backup_20260428`.
- Important checkpoint compatibility note: native training checkpoints from the curriculum are default native precision checkpoints. Do not evaluate or continue them from a `_C` build made with `./build.sh drone_race --float`; rebuild with `./build.sh drone_race` first.
- Before shutting down an ephemeral Vast instance, copy at least `checkpoints/`, `logs/`, and any changed source/docs to persistent storage or the local workstation. Do not rely on the instance root disk surviving destroy/relaunch.
- Local backup mirror created on 2026-04-25: `/Users/anon/code/rl/puffer_ai/vast_backup_20260425` (`checkpoints/` and `logs/`, about `44M`).
- Minimum backup from local machine:
  ```bash
  mkdir -p vast_backup
  rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p 26520" \
    root@ssh1.vast.ai:/root/pufferlib-drone/checkpoints/ vast_backup/checkpoints/
  rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p 26520" \
    root@ssh1.vast.ai:/root/pufferlib-drone/logs/ vast_backup/logs/
  ```
- If using a mounted persistent volume on the Vast box, mirror in-place before shutdown:
  ```bash
  rsync -av checkpoints/ /persistent/checkpoints/
  rsync -av logs/ /persistent/logs/
  ```

### Native Race Curriculum Checkpoints

- Native hover/control prior:
  - `checkpoints/drone/1777138104363/0000000039976960.bin`
- Solved H1 one-gate race bridge:
  - `checkpoints/drone_race/1777139155996/0000000049938432.bin`
  - Metrics: `success_rate=0.9995`, `crash=0.0005`, `gates_passed=0.9995`.
- Strong R1 two-gate bridge; use this as the canonical restart point for 3-gate experiments:
  - `checkpoints/drone_race/1777139257574/0000000052494336.bin`
  - Metrics: `success_rate=0.9823`, `crash=0.0177`, `gates_passed=1.9812`.
- Best current 3-gate stabilization checkpoint:
  - `checkpoints/drone_race/1777139626939/0000000052494336.bin`
  - Metrics: `success_rate=0.7990`, `crash=0.2006`, `gates_passed=2.6586`.
  - Not promotion-ready; continue 3-gate stability work before returning to 4 gates.
- Best reproducible 2026-04-28 default-native eval from restored R3 run:
  - `checkpoints/drone_race/1777139626939/0000000039387136.bin`
  - Eval artifact: `logs/drone_race/scan_r3_0000000039387136_1000.json`.
  - Metrics: `success_rate=0.7521`, `crash=0.2479`, `gates_passed=2.5122`.
  - The old dashboard-final R3 metrics remain useful context, but deterministic JSON/CSV eval should govern future promotion.
- Best current R3 checkpoint after 2026-04-28 remediation probes:
  - `checkpoints/drone_race/1777415131824/0000000039387136.bin`
  - Eval artifact: `logs/drone_race/scan_r3_ctrl12_0000000039387136_1000.json`.
  - Metrics: `success_rate=0.8629`, `crash=0.1371`, `gates_passed=2.6843`.
  - Not promotion-ready because crash remains above the `<= 0.10` threshold, but this is the current best R3 lineage candidate.
- Latest attempted continuation regressed and should not be used for warm start:
  - `checkpoints/drone_race/1777161616339/0000000078708736.bin`: 3 gates, `success_rate=0.3055`, `crash=0.6944`, `gates_passed=1.6427`.
- `scripts/train_drone_race_curriculum.sh` now includes H1, R1, R3, and R4 stages with promotion thresholds. If a stage misses its success/crash threshold, the script exits instead of automatically continuing from a regressed checkpoint.
- Do not continue from these regressed runs unless explicitly investigating failure modes:
  - `checkpoints/drone_race/1777139303583/0000000078708736.bin`: 4 gates, `success_rate=0.5402`, `crash=0.4566`.
  - `checkpoints/drone_race/1777139353818/0000000099942400.bin`: 4 gates, `success_rate=0.4035`, `crash=0.5898`.
  - `checkpoints/drone_race/1777139537143/0000000099942400.bin`: 3 gates, `success_rate=0.4637`, `crash=0.5363`.
  - `checkpoints/drone_race/1777411909778/0000000049938432.bin`: invalid 2026-04-28 continuation made from a `--float` `_C` build; ignore for lineage.
  - `checkpoints/drone_race/1777413297251/0000000099942400.bin`: default-native R3 continuation regressed to `success_rate=0.2017`, `crash=0.7976`.
  - `checkpoints/drone_race/1777413469422/0000000104923136.bin`: relaxed R3 retry regressed to `success_rate=0.4377`, `crash=0.5623`.
  - `checkpoints/drone_race/1777414869351/0000000069992448.bin`: increased `invalid_penalty=55` collapsed to `success_rate=0.0117`, `crash=0.9883`.
  - `checkpoints/drone_race/1777414942471/0000000069992448.bin`: lower `learning_rate=0.00002` final run regressed, though its early checkpoint became an intermediate improvement.
  - `checkpoints/drone_race/1777415036469/0000000069992448.bin`: zero-entropy continuation final run regressed, though its early checkpoint improved R3 before the control-penalty probe.

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
- Native CUDA/NCCL validation passes on Vast.ai, and native hover training produced usable `.bin` checkpoints.
- Full native `drone_race` from scratch collapsed at `crash=1.000`; the current path is staged hover-warm-started race curriculum.
- Q8 PufferNet benchmark and closed-loop hover comparison harness compile locally; initial trained-hover Q8 comparison shows useful score retention but action drift remains too high for promotion.
- MAVLink/SITL scaffold dry-run passes locally.
- `--slowly`/PyTorch backend runs are considered debug-only and are not part of the accepted training path.
- Native CUDA build remains unavailable on local macOS; use Linux GPU hardware for accepted throughput/training validation.
- Next required implementation: improve 3-gate race curriculum stability on GPU, add telemetry parsing in the SITL adapter, and add fake quantization during RL fine-tuning.

