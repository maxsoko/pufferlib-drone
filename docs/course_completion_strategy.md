# Course Completion Strategy

Status: **superseded 2026-07-16**. This is the preserved single-policy strategy
and must not be used as current execution direction. The active strategy is
`docs/competitive_lap_execution_prompt.md`, with durable initialization context
in `AGENTS.md` and product decisions in `PRD.md`.

## Historical Decision (Do Not Execute)

Use one recurrent full-flight policy:

1. Deterministic lifecycle code starts/reuses the simulator, arms, maintains
   heartbeat and command rate, aborts collisions, records evidence, and resets
   the race with MAVLink command `31000`.
2. The recurrent PufferLib policy owns pitch, roll, yaw, and thrust from the
   timed start through the finish. There is no gate-1 FSM handoff and no
   per-gate command arbitration.
3. Native training supplies high-throughput parallel exploration. The official
   simulator is a promotion/evidence environment, not the training inner loop.
4. The course FSM remains a diagnostic teacher and fallback baseline only. It
   may initialize a policy, but it may not fly any part of a promoted timed run.
5. Official `active_gate_index` remains the only gate-pass authority.

Do not resume manual per-gate gain sweeps. Do not add SLAM, Q8, or speed tuning
before a reliable full-course completion.

## Geometry and observation contract

The native environment contains the calibrated absolute course map internally
for physics, gate crossing, resets, rewards, and parallel training. Absolute
vehicle position, velocity, future-gate coordinates, and other native-only state
are not policy observations.

The deployed policy receives only values available through the official
interface: active-gate camera-relative pose and motion, HIGHRES_IMU-derived
attitude/rates, observable race phase, previous policy action, and elapsed race
fraction.

The official gate inner width is `1.5 m`. Native `gate_radius` is the radial
center-crossing tolerance, so target geometry is `0.75 m` on every gate. A
native radius of `1.5 m` is relaxed curriculum geometry and cannot support
promotion.

## Current evidence

- Gate 1 has been passed repeatedly through the official interface, proving
  camera ingestion, actuation, arming, heartbeat, and official race-status
  handling.
- The corrected native attitude plant uses absolute pitch/roll/yaw setpoints
  and hover-centered thrust. Normalized thrust zero maps to `0.27`, bounded to
  the live `0.18-0.42` band.
- Recurrent state is preserved across rollout chunks and gates, reset only for
  terminated agents, and restored from captured rollout-start state for PPO.
- A relaxed-radius candidate completed `3967/4096` deterministic native episodes
  with zero crashes, but its first official v3385 run collided with the left
  post of gate 0. Its closest camera-relative right displacement was about
  `0.7567 m`, exposing the former width/radius error.
- The retained true-aperture lineage passes gates 0-2 at `0.75 m`; the current
  retained-parent curriculum is tightening only gate 3. Its authoritative state
  is `logs/drone_race_full_policy_stage_d_gate4/radius_curriculum_true_aperture_g3_state.json`.
- No current checkpoint has passed exact 4096-episode promotion at `0.75 m` on
  all four gates. Therefore no further official attempt is justified yet.

## Active implementation

- `config/drone_race_full_policy_stage_d_gate4.ini`: full-start v3385-calibrated
  native environment and official-observable policy contract.
- `scripts/train_full_policy_radius_curriculum.py`: retained-parent aperture
  annealing with deterministic screens, bounded PPO children, automatic local
  step refinement, and atomic resume state.
- `scripts/eval_drone_race_checkpoint.py`: deterministic paired JSON/CSV
  development and promotion evaluation.
- `scripts/finish_full_policy_native_curriculum.py`: waits for the retained
  Gate-3 state to reach 0.75 m and runs only the exact 4096-episode all-gate
  0.75 m promotion; it never starts a trainer or the official simulator.
- `scripts/collect_full_policy_teacher.c`: native DAgger/teacher data generator;
  teacher geometry and actions are training-only and never deployed.
- `scripts/drone_sitl_competition_smoke.py --control-mode policy-attitude`:
  official full-policy runner and evidence recorder.
- `scripts/run_windows_full_policy.ps1`: reuses the healthy v3385 simulator and
  delegates normal race restarts to MAVLink `31000`.
- `docs/full_policy_execution_prompt.md`: historically authoritative copyable
  goal and closed-branch continuation record; do not execute it now.

## Verified runtime facts

Do not spend iterations re-deriving these:

- The supported simulator is AI-GP Simulator `v1.0.3385` at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`.
- The camera stream is 640x360 and the rendered camera is level.
- `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and `TRACK_INFO` are unavailable
  to the current official runner.
- `SET_POSITION_TARGET_LOCAL_NED` does not actuate.
- `SET_ATTITUDE_TARGET` plus the gyro-closed attitude controller is the working
  control path.
- MAVLink command `31000` resets the race in place. Simulator process restart is
  recovery-only, never the normal retry mechanism.
- Only official race status counts as gate progress.

## Promotion ladder

1. Development screen: at least 512 deterministic episodes, success `>=0.90`,
   crash `<=0.10`, and ordered completion at the current curriculum geometry.
2. Target geometry: all four per-gate radii exactly `0.75 m`, one policy from
   the true start, and no recurrent reset between gates.
3. Native promotion: exactly 4096 deterministic episodes (`n == 4096`), success
   `>=0.90`, crash `<=0.10`, no NaNs/invalid actions, and official-observable
   inputs only.
4. Official course proof: one valid ordered completion under policy control from
   arming, with command rate `>=50 Hz` and `<100 Hz`, heartbeat `>=2 Hz`, zero
   telemetry drain-limit hits, and immediate collision abort.
5. Reliability: at least 8/10 valid collision-free full-course runs.
6. Speed: reduce median valid time while preserving at least 8/10 validity.

## Iteration policy

- Run only one PufferLib trainer at a time.
- For a near-passing aperture parent, train in bounded 0.5M-step corrections
  and deterministically evaluate every child. Allow at most four sequential
  corrections before retaining the last passing parent and refining the step.
- Use training-only `--exploration-log-std -4.0` when the default Gaussian
  variance knocks agents off solved 0.75 m prefix gates. This changes sampling,
  not the deployed deterministic mean policy or its observation contract.
- Keep `--trainer-eval-episodes 0`: the trainer otherwise spends up to half of
  a cycle on redundant stochastic post-training rollouts after saving the
  checkpoint. The runner's separate deterministic screen remains authoritative.
- Resume trusts the atomic state's last accepted parent/radius and immediately
  continues its persisted pending schedule; it rejects changed environment
  overrides instead of spending another screen on already-proven geometry.
- Every trained-child state record includes checkpoint steps, wall time, and
  effective end-to-end SPS in addition to deterministic promotion metrics.
- Screen a tighter aperture before training; do not train when the retained
  parent already passes it.
- Never replace a passing parent with a failed child. After both bounded
  children fail, halve the local radius step and retry from the retained parent
  down to the configured minimum step.
- Fail fast after two catastrophic evaluations (`success<=0.001` and
  `crash>=0.99`); inspect the contract or curriculum before more training.
- Do not run an official attempt until a checkpoint passes native promotion.
  Never retry a known-failing unchanged checkpoint.
- Feed official failures back into native observation/plant/curriculum fidelity;
  do not patch the live runner with per-gate flight logic.
- Optimize completion time only after reliable full-course completion.
