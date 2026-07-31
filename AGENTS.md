# AI Grand Prix Workspace Context

This file applies to the entire repository. Read it before planning or running
AI Grand Prix work.

## Objective

Produce a competition-legal, fully autonomous, **competitive valid lap time**
in the locally available `AI-GP Virtual Qualifier R2` event. Preserve the
promoted VQ1 result as frozen evidence, but treat VQ2's camera/IMU-only runtime
as the active development target. A valid finish is necessary but is not the
endpoint; optimize elapsed race time while preserving reliable, ordered gate
completion and collision-free operation.

The official rules require autonomy, not an end-to-end neural policy. A
deterministic perception/planning/control stack or a hybrid stack is allowed.
Prefer the simplest inspectable controller that solves the official simulator.
Use learning only where measured evidence shows it is needed.

## Installed Official Simulator

- Current authoritative VQ2-capable build supplied by the user on 2026-07-26:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`
- WSL view:
  `/mnt/c/Users/anon/Desktop/AI-GP Simulator v1.0.3391/AIGP_3391/FlightSim.exe`
- VQ2 package identity: launcher SHA-256 `0d3217fa...`, shipping executable
  SHA-256 `68dfd80d...`, and `4574381151`-byte PAK SHA-256 `5d424b4e...`.
  The UI exposes `AI-GP Virtual Qualifier R1`, `AI-GP Virtual Qualifier R2 -
  Submission`, and `AI-GP Virtual Qualifier R2 - Training`. Use Training for
  all development and probing. Submission is not authorized until a promoted,
  submission-ready controller and explicit submission decision exist.
- Current VQ2 SDK is
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\PyAIPilotExample-v4`.
  It is byte-identical to the VQ1 SDK, so runtime packet evidence supersedes
  its stale/general comments.
- Current authoritative VQ1 build supplied by the user on 2026-07-26:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391-VQ1\AIGP_VQ1_3391\FlightSim.exe`
- WSL view:
  `/mnt/c/Users/anon/Desktop/AI-GP Simulator v1.0.3391-VQ1/AIGP_VQ1_3391/FlightSim.exe`
- Current bundled example/SDK:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391-VQ1\PyAIPilotExample-v4`
- The user supplied the official re-release notice stating that this legacy
  VQ1 build has full access to telemetry data interfaces, sensor outputs,
  downstream telemetry, and state estimation. This build supersedes v3385 for
  active VQ1 development. Preserve v3385 artifacts as historical evidence, but
  do not silently execute new attempts against v3385 after the v3391 migration.
- Verify the actual v3391 wire contract before designing the successor. The
  bundled `PyAIPilotExample-v4/mavlink_rx.py` still contains older comments
  claiming `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and gate fields are
  disabled/nulled, which conflicts with the supplied re-release notice. Treat
  runtime packet evidence as authoritative and record rates and non-null fields.
- The v3391 example continues to define reset as learned-target MAVLink
  `COMMAND_LONG` command `31000`, confirmation `0`, parameters 1--7 all zero.
  Its `controller.py` also documents revision-3390 type-mask extension bit `16`
  for physical-rad/s body rates. Legacy behavior remains available without the
  bit; do not change existing controller scaling until measured on v3391.
- Historical simulator used through 2026-07-19:
- Exact executable:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`
- WSL view:
  `/mnt/c/Users/anon/Desktop/AI-GP Simulator v1.0.3385/AIGP_3385/FlightSim.exe`
- Bundled example:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\PyAIPilotExample-v2`
- Controller repository on Windows:
  `C:\Users\anon\code\pufferlib-drone`
- Windows controller Python:
  `C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe`
- Historical UI state through 2026-07-16 exposed only VQ1. The separately
  supplied VQ2-capable package above supersedes that constraint for active
  development; do not use the VQ1-only install for new VQ2 work.
- The user confirmed on 2026-07-16 that official VQ1/R1 contains **six ordered
  gates**. Official `active_gate_index=3` means three gates have passed and
  Gate 4 is active; it is not the final gate. A full-lap validation must require
  all six passes and a nonnegative `race_finish_time_ns`. Use target/min index
  `4` only for a bounded Gate-4 milestone and target/min `6` for course proof.
  Historical native curricula with four gates are incomplete proxies and are
  never authoritative for the official course count.
- Finish-lock decision, 2026-07-16: the official FAQ says teams may attempt runs
  throughout the qualification window, reports thousands of completed simulator
  runs used to refine autonomy stacks, ranks the fastest valid times, and then
  says the time is locked when that run passes all gates. Treat this as a
  per-attempt timer lock, not an irreversible first team submission. Source:
  https://www.theaigrandprix.com/. No public VQ1 leaderboard/cutoff was found.
  A deliberate local first finish is authorized, with immediate official-finish
  stop and subsequent reset proof.

Use `scripts/start_windows_aigp_race.ps1 -DiscoverOnly` to discover an install.
When a path is supplied explicitly, use the exact executable above; do not ask
the user to provide it again.

## Official Interface and Verified Runtime

Public contract: `docs/260508_Technical_Spec_0002.pdf` (`VADR-TS-002`, issue
`00.02`). Local operational source: `docs/windows_official_sitl_runbook.md`.
MAVLink site-wide relevance audit and repository mapping:
`docs/mavlink_site_context_2026-07-18.md`. Read it before changing framing,
target IDs, heartbeat, reset delivery, TIMESYNC, telemetry validity, or
`SET_ATTITUDE_TARGET` semantics.

- Simulator: Windows 11, deterministic rigid-body physics at `120 Hz`.
- MAVLink client: UDP `14550`; camera client: UDP `5600`.
- Camera: JPEG chunks with header `<IHHIIQ`, `640x360`, nominal `30 Hz`,
  `fx=fy=320`, `cx=320`, `cy=180`.
- The v1.0.3385/VQ1 rendered camera was measured level. VQ2's UI displays
  `cam 20°`, but passive UDP imagery measures optical uptilt
  `1.920944634732011 deg`; use that value for the active VQ2 runtime while
  keeping tilt configurable. The stationary IMU mount pitch near `17.8°` is a
  sensor-body calibration and must not be substituted for camera tilt.
- Maintain heartbeat `>=2 Hz`; send commands at an engineering target of
  `60-90 Hz` and always `<100 Hz`.
- In VQ2 Training, command-free runtime evidence publishes `HIGHRES_IMU`,
  heartbeat, race status, actuator status, and camera. `ATTITUDE`,
  `LOCAL_POSITION_NED`, `ODOMETRY`, and track transfer are absent. Collision is
  event-driven and was not exercised by the stationary inventory; retain it as
  the immediate-abort authority when observed.
- `SET_POSITION_TARGET_LOCAL_NED` does not actuate v1.0.3385.
- The working path is `SET_ATTITUDE_TARGET` body-rate/thrust control with the
  repository's gyro-closed controller.
- Official race status `active_gate_index` is the only gate-pass authority.
  Vision-only pass events are diagnostic.
- Simulator reset is MAVLink `COMMAND_LONG` with command ID `31000`,
  confirmation `0`, and parameters 1--7 all `0`, addressed to the learned
  simulator system/component IDs. On v1.0.3385, send a normal disarm first;
  then count the reset only after `sim_boot_time_ms` rolls backward by more
  than `1000 ms` and a new nonnegative `race_start_boot_time_ms` appears.
  Keep a healthy simulator process alive; process restart is recovery-only.
- Abort immediately on collision. Never continue a damaged run to collect a
  misleading time.

No human interaction is allowed during a submitted timed flight. Lifecycle
automation before/after the timed run is permitted. Never use disabled,
privileged, or simulator-internal coordinates in the deployed controller.

## Current Strategy

### VQ2 PufferLib Gate-1 promotion — N294/N295, 2026-07-26

- The user explicitly requires a PufferLib flight policy because the prior
  classical lineage did not solve the official vision-only runtime. Classical
  state feedback is permitted only as a native training label generator; the
  deployed controller is the single recurrent 32-input PufferLib policy.
- VQ2 camera duplicate emission is fixed by deduplicating complete frames on
  `(frame_id, sim_time_ns)`. A six-frame passive calibration measures UDP
  optical uptilt `1.920944634732011 deg`; Gate 1 is approximately
  `[10.78, 0.084, -0.56] m` in the policy body camera convention. The fixed
  IMU mount angle is not camera tilt or vehicle attitude.
- Direct N104 transfer, narrow N285 feature adaptation, randomized PPO N286,
  and exploratory-aperture PPO N287 are rejected without live control. N104
  and every N286 deterministic checkpoint were `0/128` at the measured
  `0.75 m` Gate-1 aperture. N287's apparent training gains depended on sampled
  actions; its deployable mean remained `0/128` below a `1.60 m` aperture.
- N288--N290 use native teacher-labeled scheduled-sampling DAgger while keeping
  privileged state out of the student observation and runtime. N288 is
  `512/512` exact but only `275/512` perturbed; N289 is `338/512` perturbed;
  N290 is `367/512` perturbed. N291's `75%` student rung regressed and is
  rejected.
- N294 is the frozen interpolation `N289 + 0.60*(N290-N289)`. Checkpoint
  SHA-256 is
  `a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6`.
  It passes deterministic exact Gate 1 `512/512` with `0.309751 m` mean
  crossing error and perturbed exact-512 `422/512` (`82.421875%`), with zero
  crash and timeout in both screens.
- Full-rate Windows shadow 003 passes: `520/520` recurrent steps replay within
  `4.3229561e-7`, inference is `52.0 Hz`, no-op cadence is `63.401 Hz`, all
  eight deployment hashes match, and reset/arm/setpoint/disarm counts are zero.
  It sent only the required `21` heartbeats. Shadow/parity SHA-256 values are
  `4b9837de...`/`c632a708...`. Shadow 002 was control-safe but its sparse
  logging was insufficient for recurrent parity and is superseded by 003.
- N295 ran exactly once under `vq2_n295_gate1_bounded_001` and passed Gate 1.
  Official index/finish are `1/-1`; official Gate-1 race time is
  `3.228605508 s`, and the authoritative index stop fired at `3.250 s`.
  The run delivered `207` PufferLib-policy setpoints at `63.995 Hz` with zero
  command-rate violation, collision, invalid state, telemetry dropout, or
  drain-limit hit. Its one reset was detected, all eight deployment hashes
  match, and one exit disarm was sent. JSON SHA-256 is
  `5f46b74bd5636ba282c9868cff7f7d17b409965e39987314479cd0e2f5728091`.
- A five-second receive-only follow-up sent zero packets and proves disarm with
  `base_mode=65`, system status `3`; SHA-256 is
  `9b295ccbbad88214dce8da82b23a8f5ceb88f56c5174405c0f024af445c5baaf`.
  It also observes the simulator's post-disarm terminal ground contact
  (`0.0464` impact) after the accepted index stop. This is not an in-run N295
  collision, but the reset/inactive post-stop state must not be reused as live
  evidence. Do not retry N295 unchanged.
- Freeze further VQ2 simulator commands until the N295 trace is added to an
  offline Gate-2 continuation diagnosis and a separately tagged bounded
  PufferLib-policy experiment is source-locked and preregistered. VQ2
  Submission remains unauthorized.

### VQ2 PufferLib Gate-2 offline continuation — N296--N313, 2026-07-27

- The deployment constraint is explicit: Gate 2 and every later gate must be
  flown by a recurrent PufferLib policy. Classical/native state feedback is
  training-only teacher/reward infrastructure and must be absent from
  deterministic screens, shadow, and FlightSim control.
- N295's legal camera/IMU reconstruction places the next gate approximately
  `14.74 m` forward, `8.70 m` right, and `1.095 m` above the Gate-1 transition,
  with native entry speed about `4.677 m/s`. Unchanged N294 is `0/128` here.
  The vehicle must brake to roughly `1.3 m/s`, bank right, hold altitude, and
  accelerate again; a fixed-pitch teacher is physically inadequate.
- N296 shared-state BC and zero-state N297--N303 DAgger variants all remain
  `0` deterministic Gate-2 completions. N304 PPO also has no policy-only
  completion. Do not promote or live-test any of them.
- Default-off training-only extensions add dynamic pitch speed control,
  world-frame vertical thrust feedback, and explicit yaw stabilization. They
  do not alter the 32-value student observation or deployed checkpoint. The
  corrected teacher is `128/128` exact with `0.263289 m` mean crossing error
  at a `1.30 m/s` target, and `244/512` under broad perturbation, zero crash.
- N306 assisted trace SHA `30d8ec6e...` produced `59,397` legal records
  (`teacher.bin` SHA `59951931...`). High-convergence distillation, reweighted
  DAgger N307, policy-state DAgger N308, aggregate N309, and full-assistance
  PPO N310 all remain `0` teacher-free completions and are rejected.
- Assistance annealing is only an offline bridge. N311 step `188416`, SHA-256
  `13561d7613d2907bb80189c0e1cb31e3da4deae6856b88c033037d98a41d5529`,
  scores `128/128` with `0.50` assistance and `119/128` with `0.25`, but
  crashes unassisted. N312 has `0/33` unassisted checkpoints, all crashes.
  N313 peaks at `116/128` with `0.25` assistance, below N311, and is rejected.
- No post-N295 FlightSim command was sent. Keep VQ2 Training frozen until a
  Puffer checkpoint completes Gate 2 with teacher blend exactly zero on exact
  and perturbed screens, then passes composite replay and zero-command Windows
  shadow. Submission remains unauthorized.

### VQ2 PufferLib Gate-2 continuation — N314--N336, 2026-07-27

- Runtime remains PufferLib-only. A legal composite may keep N294 for Gate 1
  and warm a second recurrent Puffer policy on the same public observations,
  then select the second policy at official index `1`; both action sources must
  be Puffer checkpoints. Classical feedback may label offline records or shape
  training reward, but cannot emit, blend, override, or arbitrate an action in
  an admission screen, shadow, or FlightSim.
- N317 source-locks the `195` public N295 observation/action records ahead of
  each of `128` native Gate-2 teacher episodes. The resulting `84,357`-record
  dataset SHA-256 is
  `30b97b22cde3848380da9d22464bc5d7da37eedc642fce58035de5f3252782c0`.
  The command-free evaluator warms a candidate's recurrent state on that exact
  official prefix, restores the final legal previous action, and pins teacher
  blend to zero for admission. A positive blend is accepted only with explicit
  training-dataset output and marks the result non-admissible.
- N314--N322 assistance, distillation, prefixed BC/DAgger, decoder ridge, and
  recurrent-fit branches produce no safe unassisted completion. N323
  `policy_1e5_e1.bin` (SHA `dff5bc7c...`) is the retained zero-crash local
  parent, but remains `0/128` with `4.753 m` closest approach. N324--N327 show
  a sharp safety/completion boundary: moving toward the required pitch/thrust
  response creates completions only with unacceptable crashes.
- N328 is the first teacher-free warmed Puffer Gate-2 completion. Checkpoint
  `policy_p0p50_t0p15.bin` (SHA `1fe9c426...`) completes `7/128` at the
  official `0.75 m` aperture but crashes `89/128`; it is not deployable. The
  wider-curriculum parent `policy_p0p50_t0p05.bin` (SHA `532f0dfa...`) scores
  `32/128` at radius `2.0 m`, crashes `71/128`, and misses `25/128`.
- N329--N335 small decoder updates, rolling PPO, assisted recapture, recurrent
  interpolation, added roll, phase-feature training, and wide-aperture PPO all
  fail to improve that safety/completion frontier. N336 fixes PPO rollout
  integrity with one complete episode per agent and terminal-safe padding, but
  its best same-seed deterministic child scores `31/128` successes and
  `72/128` crashes at radius `2.0 m`, versus the unchanged parent's `32/128`
  and `71/128`; reject N336.
- No post-N295 simulator command has been sent. No Gate-2 checkpoint is
  authorized for live use. Continue offline Puffer training; require reliable
  blend-zero exact and perturbed Gate-2 completion, composite replay, and a
  zero-command Windows shadow before preregistering a bounded Training run.
  VQ2 Submission remains unauthorized.

### VQ2 corrected Gate-2 Puffer frontier — N337--N388, 2026-07-27

- Runtime control remains exclusively recurrent PufferLib. N294 emits every
  Gate-1 action; a separately warmed recurrent Puffer checkpoint may emit the
  complete four-action vector from official index `1`. Classical state
  feedback is training-only reward/teacher infrastructure and never emits,
  blends, clips, schedules, or overrides a deployment action.
- N337--N342 whole-Puffer schedules, a high-thrust checkpoint, static decoder
  surfaces, action-row PPO, and decoder ES do not promote. A subsequent audit
  found that the native custom start still applied legacy reset-position noise
  (`+-0.1 m` horizontal, `+-0.05 m` vertical), so every earlier result called
  “exact” was actually stochastic. Default behavior is preserved, but the
  measured-transition evaluator now explicitly sets both reset noise terms to
  zero; all N343+ exact claims use that corrected contract.
- Under the corrected exact contract, N328 checkpoint
  `policy_p0p50_t0p10.bin` passes `16/16`. N348 source-locks a centered decoder
  line child (SHA `b754b736...`) with `0.306 m` exact crossing error. Tiny
  start-state jitter exposes a legal-observation recurrent bifurcation after
  gate detection dropout; N353's source-locked public trace contains only
  observations and Puffer actions. Recurrent reset and the legal dropout
  predictor branches fail and are rejected.
- A default-off native training reward now penalizes gate image-plane
  misalignment from active index `1`. It uses native geometry only to compute
  an offline reward; the policy still observes the unchanged legal 32-value
  ABI, teacher blend is zero in every admission screen, and every plant action
  is the Puffer output. N357--N363 progressively improve tiny-start robustness
  from `55/128` to `120/128`; N364 regresses and is rejected. N365 reaches
  `125/128`, and N366 reaches `128/128` with zero crash. N366 checkpoint SHA is
  `a53d767e073a...`; its larger audit is `509/512`, zero crash, at perturbation
  scale `0.001`.
- N367 proves N366 is `128/128` for separate gate, plant, and combined
  start-plus-gate scale-`0.001` probes. At start scale `0.002` it falls to
  `96/128` with one crash. N368--N380 therefore insert `0.00125` and `0.0015`
  rungs, lower PPO exploration, and eventually open the full secondary Puffer
  encoder/MinGRU/decoder at trust-region learning rates. The retained offline
  frontier is N380:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n380_gate2_full_policy_reward_bracket/teacher100_align20/update_0001.bin`,
  SHA-256
  `6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a`.
  It passes the corrected exact transition with `0.264729 m` crossing error.
  N386 audits `509/512` at start scale `0.00125` (`2` delayed low crashes,
  `1` miss) and `498/512` at `0.0015` (`8` delayed low crashes, `6` misses).
  Normal completion is around `11.3 s`; rare crashes occur after a missed
  approach around `15.1 s`, but timeout cannot substitute for completion.
- Start-jitter diagnostics are now default-preserving and can isolate elapsed,
  position, velocity, attitude, and body-rate groups. Each group alone passes
  `256/256`; body rates plus attitude are the dominant interaction. N384--N385
  targeted and large-batch recurrent updates, and N387 lower-noise full-start
  updates, do not beat N380 and are rejected.
- N388 adds the minimal deployment callable
  `scripts/policy_callable_vq2_gate2_composite.py`. Both recurrent checkpoints
  advance on every public observation; official progress selects one complete
  Puffer action vector, with no blend or analytic fallback. Callable SHA-256 is
  `1998a3370df919bf0eaf16e57d3928de41850fefac9bc0ea026d0ba783d653bc`.
  Source-locked replay passes all `195` N295 prefix samples with maximum action
  error `4.805624485015869e-7`, and the index-1 selector matches the standalone
  N380 action exactly. Report:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n388_gate2_composite_replay/report.json`.
  Actual Windows controller Python passes the three runtime-only composite
  tests and reproduces the same errors without PyTorch; Linux/Windows report
  SHA-256 values are `6ae43d5d...` and `e8e2ee11...`.
- N389's `0.10` and N390's `0.01` assisted label probes each score `0/256`:
  even tiny classical blending destroys the Puffer trajectory. Their datasets
  are quarantined and must not be fitted. N391--N393 checkpoint/row line
  searches, N394--N395 whole-Puffer late schedules, and the N396 legal motion
  predictor do not improve completion; N396 is `0/256` at every gain.
- N397 attempted the first composite shadow from N295's inactive post-disarm
  ground state and is rejected without unchanged retry. It saw no camera or
  inference ticks and inherited collision/invalid state, but its lifecycle is
  clean: reset/arm/setpoint/disarm all zero, one heartbeat only. JSON SHA is
  `3f5d4b8d...`.
- The healthy v3391 process was preserved and recovered through Pause -> Back
  To Main Menu. Screenshots explicitly distinguish the three event rows; only
  `AI-GP Virtual Qualifier R2 - Training` was selected. Receive-only telemetry
  then proved index `0`, finish `-1`, `base_mode=193`, and system status `4`.
- N398 `vq2_n398_gate2_composite_shadow_002` passes in active Training. It
  records `527/527` samples at `52.7 Hz`, Linux replay maximum action error
  `4.0513782501028217e-7`, all nine manifest hashes exact, `113` camera frames/
  detections, and `1,804` telemetry messages. No-op cadence is `64.002 Hz`
  (`640` counts), zero violations/setpoints; reset/arm/control/disarm are all
  zero. Shadow/parity SHA-256 values are `13120ab1...`/`66b059e4...`.
- Preregister exactly one bounded Training attempt tagged
  `vq2_n399_gate2_bounded_001`. Wrapper SHA-256 is
  `db2e603ff5dbcaa265350816c63c2c44fc7daf0effe95e3c6648621cb14edb4a`.
  Reuse the active v3391 Training process, issue one normal detected command
  `31000` reset, request `80 Hz`, run at most `14 s`, require target/min/stop
  official index `2`, abort on collision/invalid/dropout/rate failure, and
  require exactly one exit disarm plus passive disarm proof. Use N294 for all
  index-0 actions and N380 for all index-1 actions; both recurrent states run
  continuously. Any failure rejects N399 and forbids an unchanged retry.
  VQ2 Submission remains unauthorized.
- N399 ran exactly once and is rejected; never retry it unchanged. It passed
  Gate 1 at official `3.194960355 s`, then collided with ID `1002`, impact
  `5.930058`, at official index `1` around `5.845 s` race time. Finish stayed
  `-1`. Transport is accepted: `372` whole-Puffer setpoints at `63.999 Hz`,
  zero rate violation, `1,657` telemetry messages, `67` frames/detections, and
  one exit disarm. The attempt report SHA-256 is `28a63a4...`.
- The command-free N399 follow-up sent zero heartbeats, TIMESYNC replies,
  metadata requests, lifecycle commands, and setpoints. It proves passive
  disarm at `base_mode=65`, system status `3`; SHA-256 is `c84347fb...`.
- N400 reproduces N380's old native exact screen as `1/1` success with
  `0.264745 m` radial error, but N401 proves why that screen did not predict
  N399. All `340` live decisions (`188` N294 then `152` N380) replay within
  `6.556510925292969e-7`, so neither checkpoint execution nor transport
  diverged. At the official phase transition, the camera-derived raw pose
  jumped from `[14.736, 8.608, -1.233]` to `[8.488, 4.978, -0.364] m` in
  `31 ms` (`234.8 m/s` apparent) while bearing changed only `0.00174 rad`.
  This is aperture/range aliasing, not vehicle motion. The old native screen
  also warmed on N295's `12 s` elapsed normalization, then began Gate 2 on a
  `20 s` native clock, whereas N399 used one continuous `14 s` clock.
  N401 analysis/report SHA-256 values are `a78c1e73...`/`59813343...`.
- Freeze live commands. The next candidate must remain whole-output recurrent
  PufferLib, warm on N399's actual Gate-1 prefix, use one deployment-matched
  elapsed clock, and train/screen against source-locked camera range and gate
  association perturbations. Classical/native state may shape offline reward
  or training only and may never emit, mix, clip, or override a runtime action.
  No new bounded attempt is authorized yet; Submission remains forbidden.

### Historical VQ2 command-free migration — superseded by N294/N295

- The exact new build is live in `AI-GP Virtual Qualifier R2 - Training`.
  `VQ2 Submission` was observed in the event list but never selected. The
  training vehicle remains stationary at official index `0`; no heartbeat,
  TIMESYNC reply, metadata request, reset, arm/disarm, or setpoint was sent.
- Raw 20-second evidence saw `3215` MAVLink-v2 datagrams and `39822` valid
  TS-002 camera chunks, with zero v1 packets or camera-header failures. Raw
  artifact SHA-256 is `73cb40ea...`.
- The decoded 12-second inventory saw HEARTBEAT `119` (`9.917 Hz`), HIGHRES_IMU
  `655` (`54.583 Hz`), ACTUATOR_OUTPUT_STATUS `1127` (`93.917 Hz`), and
  ENCAPSULATED_DATA/race status `48` (`4 Hz`). It saw zero ATTITUDE,
  LOCAL_POSITION_NED, ODOMETRY, track handshakes/transfers, TIMESYNC, malformed
  messages, or dropouts. Race status remained index `0`, finish `-1`.
- Camera is `640x360`. The transport completed `720` JPEG instances in `12 s`,
  consisting of duplicate pairs for approximately `360` unique frames or
  `30 Hz`; consumers must deduplicate on `(frame_id, sim_time_ns)`. There were
  zero malformed chunks, chunk loss, frame stalls, or reconstruction failures.
  Inventory/frame SHA-256 values are `7e68308c...`/`8a0e9ea3...`.
- Probe root/Windows SHA-256 is `fe6943a4...`; files are byte-identical and
  both compile. Detailed authority is
  `docs/vq2_interface_inventory_2026-07-26.md`.
- N283 is VQ1-only because its governor consumes public vehicle/gate pose. Do
  not run it in VQ2. Freeze all VQ1 artifacts and do not select VQ2 Submission.
  The next development path is command-free/offline: measure VQ2 camera
  extrinsics, define a camera/HIGHRES_IMU/previous-action/official-progress ABI,
  and validate passive perception/state reconstruction in Training. Do not send
  a VQ2 reset, arm/disarm, heartbeat, TIMESYNC response, or control setpoint
  until that source-hashed passive gate is complete and a bounded Training run
  is separately preregistered.

### Frozen VQ1 promotion — N283, 2026-07-26

- The v3391 migration, telemetry proof, first valid lap, and first repeatable
  optimization are complete. N270 is the promoted controller: frozen N259
  inference plus the public-telemetry governor at crossing/max/slowdown/gain/
  cap `7.5/8.0/10.0/1.5/4.0`. N267--N269 are `3/3` valid; the best official
  time is N268 at `26.591306686 s`.
- N271 source-locks N270, all three official traces, N259, the runner, and the
  corrected config. Its independent three-live-model sweep selects crossing/
  max/gain `9.0/9.0/2.0`, retaining slowdown/cap `10.0/4.0` and every safety
  guard. Predicted finishes are `23.273484/23.243388/24.080792 s`; worst
  modeled crossing error is `0.432355 m` under the frozen `0.45 m` limit.
  This is offline evidence only.
- N272 passed the one authorized bounded run at official index `3`, with Gate
  1/2/3 times `4.435974121/7.801361083/11.806352615 s`, null collision/invalid
  reason, `59.965 Hz`, one reset, six transferred gates, zero telemetry faults,
  and final stop/disarm. Artifact SHA-256 is `8563b545...`.
- N273 is rejected without unchanged retry. It passed Gates 1--3, then the
  legacy `1.5 m` plane-miss guard fired at Gate 4 before an official status
  update. N274 reconstructs a center-safe `0.267310 m` Gate-4 plane crossing;
  the latest status timestamp was still `21.473 ms` before that crossing.
- N274 preserves the `1.5 m` miss distance and adds only a fail-closed `350 ms`
  status-acknowledgment bound: abort when a post-crossing official status still
  reports the same gate or when the bound expires. Default zero-grace behavior
  is legacy-exact. Corrected runner SHA-256 is `643cdec8...`.
- N275 passed its single bounded attempt at official index `4`, with Gate
  1--4 times `4.444323539/7.826340675/11.826963424/17.182216644 s`, finish
  `-1`, null collision/invalid reason, and `60.049230 Hz`. Its one reset, six
  transferred gates, zero telemetry faults, final stop, and disarm all pass.
  The new guard was exercised (`4` deferrals, maximum status lag `245 ms`) and
  accepted the post-crossing Gate-4 update without weakening the `1.5 m` miss
  threshold. Artifact SHA-256 is `89fab65c...`.
- N276 passed its single full-lap attempt at official index `6` in
  `23.848192214 s`, improving N270's best by `2.743114472 s`. All gate times
  are `4.422417163/7.812987327/11.818965911/17.180858612/20.544542312/
  23.848192214 s`; collision/invalid reason are null, wire rate is
  `59.992485 Hz`, telemetry faults are zero, and reset/transfer/final-stop/
  disarm proofs pass. Artifact SHA-256 is `6618847f...`.
- N277 is rejected without unchanged retry. It repeated N276's fast Gate-1--3
  prefix, then the runner misclassified equal integer-millisecond crossing and
  status timestamps (`20457/20457 ms`) as a post-crossing same-gate
  acknowledgment. It aborted at index `3`; collision is null and transport is
  clean. Artifact SHA-256 is `9e5e35f5...`.
- N278 source-locks that failure. The reconstructed vehicle center is inside
  Gate 4's transferred aperture, and the abort occurred `211 ms` after crossing
  with `139 ms` of grace remaining. The guard now requires a strictly newer
  status timestamp (`>`); distance/timeout behavior is unchanged. Root/Windows
  runner SHA-256 is `de4995a1...`; focused tests pass `19/19` and Windows
  compile/direct checks pass.
- N279 passed its single bounded run at official index `4`, with Gate 1--4
  times `4.423978328/7.786879062/11.807009696/17.171604156 s`, finish `-1`,
  null collision/invalid reason, and `60.036769 Hz`. Reset, six-gate transfer,
  telemetry health, final stop, and disarm all pass. Artifact SHA-256 is
  `e068a685...`.
- N280 passed all six official gates in `23.832376480 s`, improving N270's best
  by `2.758930206 s`. Collision/invalid reason are null, wire rate is
  `60.041667 Hz`, telemetry faults are zero, and reset/transfer/final-stop/
  disarm proofs pass. Artifact SHA-256 is `5b926383...`.
- N281 and N282 passed confirmations in `23.879062652/23.867099761 s`. Together
  with N280 at `23.832376480 s`, the strict-ack controller is `3/3` official,
  collision-free, valid, telemetry-clean, and below `24.5 s`. Finish spread is
  only `0.046686172 s`; mean is `23.859512964 s`.
- N283 is promoted and supersedes N270. It freezes N259, runner SHA-256
  `de4995a1...`, config SHA-256 `9eec139d...`, and governor crossing/max/
  slowdown/gain/cap `9/9/10/2/4`, miss distance `1.5 m`, strict post-crossing
  acknowledgment, and grace `0.35 s`. Best official time is N280 at
  `23.832376480 s`, `2.758930206 s` faster than N270's best.
- Promotion aggregate SHA-256 is `898c6671...`; it regenerates exactly and its
  focused suite passes `20/20`. No additional confirmation or simulator action
  is authorized. Further optimization requires new offline evidence, bounded
  validation, and unique tags; N259 remains immutable.

### v3391 telemetry supersession — 2026-07-26

- The vision-only N241 Gate-4 identification continuation is paused. The
  telemetry-enabled VQ1 v3391 release may eliminate the unobservable-state
  problem that motivated N203--N241.
- First migrate the running session from v3385 to the exact v3391 executable,
  open VQ1, and perform a command-free packet inventory. Explicitly verify
  `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, track-transfer contents,
  `HIGHRES_IMU`, race status, collision, actuator output, camera, and reset
  rollback. Do not infer availability from the announcement or stale SDK
  comments alone.
- If v3391 publishes non-null vehicle pose plus six gate poses, prefer the
  simplest inspectable solution: ordered-gate trajectory generation and a
  deterministic position/velocity or cascaded attitude controller. Use the
  official `active_gate_index` for sequencing and require index `6` plus a
  nonnegative finish time. Only resume learned visual-state reconstruction if
  the live telemetry probe shows those state feeds are actually unavailable.

### Current promotion snapshot — 2026-07-17

- The old N104 next-action text below is superseded. N104 Candidate 002 was run
  once, passed Gate 1, then collided before Gate 2 at official index `1` and is
  rejected without an unchanged retry. The separately reproduced historical
  recurrent parent remains authoritative for a clean official Gates-1-through-
  3 prefix (checkpoint SHA-256 `ea03965f...`).
- N142 remains the retained deployable late tail. The phase-reset composite uses
  N112 for Gate 4, N118 for Gate 5, and N115A plus a coordinate-free visible
  controller for Gate 6. It uses only the 32-value camera/IMU/action/progress
  observation; no native state, course spline, or gate coordinates are
  deployed. The radius-`0.75 m` Gate-4-through-Gate-6 validation passes
  `128/128` common and `128/128` disjoint, with zero crash/timeout.
- Frozen checkpoint SHA-256 values are Gate 4/N112
  `9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca`,
  Gate 5/N118
  `e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30`,
  and Gate 6/N115A
  `4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354`.
- N142 Candidate 003 was executed exactly once and is rejected without an
  unchanged retry. It passed Gates 1-2, then collided with frame ID `1001` at
  active Gate 3/official index `2` around `10.266 s`; Gate 4 never activated
  and finish stayed `-1`. The last clean Gate-3 pose was approximately
  `2.945 m` forward, `2.167 m` left, and `0.225 m` down. Exit disarm succeeded;
  passive status records `base_mode=65`, index `2`, finish `-1`. This is the
  known severe-left Gate-3 failure family, not evidence about N142's late tail.
- N143 changes only Gate-3 roll at official index `2`, promoting the frozen
  Candidate-128/129 observable lateral-intercept rules. It leaves recurrent
  state and pitch/thrust/yaw untouched. An adapter-consistency replay covers
  all ten Candidate-129 traces: `237` Gate-3 samples, all adaptive/early/counter
  modes exercised, zero violations, and the source batch retains its historical
  `8/10` accepted runs. Report SHA-256 is `4ae028aa...`.
- Linux and actual Windows Python replay the identical `15,981`-record N142
  Gate-4-through-Gate-6 dataset with maximum action error
  `6.616115570068359e-6`; therefore the N143 Gate-3 change leaves all late-tail
  actions invariant. The updated composite callable SHA-256 is
  `d0119c0d9fa099cbe08bdbd7072e7089f9961250183b3fa358cc01ce668cf9a4`;
  the SHA-pinned wrapper is `e241e759...`. The focused promotion suite passes
  `65/65`, and both Windows PowerShell launchers parse successfully.
- Use `scripts/run_windows_six_gate_composite.ps1`. `BoundedGate4` explicitly
  uses target/min/stop official index `4`; `FullLap` uses target/min `6` and
  remains unauthorized. Zero-command N143 Shadow 003 passed with `223/223`
  phase-0 ticks replayed within `2.11e-7`, all 11 manifest artifacts exact,
  healthy streams, and reset/arm/setpoint/disarm counts all zero. Its JSON/
  parity SHA-256 values are `d4be9256...`/`7e5225af...`.
- A preflight evidence audit then fixed the effective-command-rate metric: for
  fixed-rate flights it now measures the first-to-last wire setpoint interval,
  excluding reset/calibration lead time. Candidate 004 then falsified the
  initial reporting-only hypothesis: its actual publisher delivered only
  `42.128 Hz` over `7.406 s`, despite zero too-fast violations.
  Because runner/adapter manifest hashes changed, default zero-command N143
  Shadow 004 was required and passed: `242/242` ticks within `2.41e-7`, all 11
  current artifacts exact, healthy streams, finish `-1`, and all four lifecycle/
  control counts zero. JSON/parity hashes are `a31c92ee...`/`fd122b48...`.
- Candidate 004 ran once and is rejected without an unchanged retry. It passed
  Gate 1, then collided with ID `1001`, impact `9.2843`, at active Gate 2/index
  `1` around `7.844 s`; finish stayed `-1` and Gate 3/N143 never activated.
  Exit disarm succeeded; passive status is `base_mode=65`, finish `-1`.
- N143 transport repair now uses a `1 ms` GIL handoff, Windows timer/priority
  setup, hot deadline waiting, and compliant catch-up no faster than `95.2 Hz`.
  Shadow mode measures the same scheduler under live camera/inference load via
  a no-op cadence sink that sends zero MAVLink setpoints. Both smoke acceptance
  and `verify_live_policy_shadow.py` require cadence in `[50,100) Hz`, zero
  violations, and zero probe setpoints. Verification passes `92/92` focused
  Linux tests and `38/38` actual-Windows smoke tests. Runner SHA is `ec95ab6e...`.
  Zero-command Shadow 005 exercised the cadence successfully at `54.569 Hz`
  across `9.969 s`, zero violations/setpoints, but parity failed because the
  post-collision simulator state was inactive (`race_start=-1`) and all `96`
  frames contained no gate. Lifecycle/control counts stayed zero. Recover the
  R1 UI with `start_windows_aigp_race.ps1 -NoRelaunch -SkipLoginClick`, then
  zero-command Shadow 006 passed in active R1: cadence `52.518 Hz` over
  `10.187 s`, zero violations/probe setpoints, `42/42` visible ticks within
  `1.58e-7`, all 11 hashes exact, and lifecycle/control zeros. Evidence hashes
  are `b274a99e...`/`2e38c1df...`.
- Candidate 005 ran exactly once and is rejected without an unchanged retry.
  The repaired publisher cleared its live requirement at `50.233 Hz` over
  `5.375 s` (`271` wire commands, zero too-fast violations), but the vehicle
  never passed Gate 1. It collided with frame ID `1001`, impact `7.5096`, at
  active index `0` after about `5.766 s`; finish stayed `-1`. The last clean
  Gate-1 pose was approximately `4.444 m` forward, `1.451 m` left, and
  `0.076 m` up. Exit disarm and passive `base_mode=65` status passed.
- Do not apply the complete Gate-3 adapter at Gate 1: fresh projection over
  twelve Gate-1-passing traces shows its counter-bank would alter `185/397`
  archived samples. The evidence-separated N144 hypothesis is narrower: reuse
  only the positive adaptive/early/close roll floors at official index `0`,
  never the Gate-3 counter-bank, and keep Gate 2/index `1` exact. Before code
  activation or another simulator command, formalize this projection with
  source hashes and require zero action changes across all `397` passing-trace
  samples while Candidate 005 exercises its adaptive/close correction. That
  verifier now passes: 12 accepted sources, `397` samples unchanged exactly,
  both legacy/current ABIs covered, and exactly six Candidate-005 changes
  (`2` adaptive, `4` close) with all non-roll channels exact. It also records
  the rejected full adapter's `185` accepted-trace counter changes. Report
  SHA-256 is `71eb295c...`. N144 now activates only this narrow Gate-1 branch;
  callable SHA-256 is `f4364b03...` and pinned wrapper SHA-256 is `2fd967a8...`.
  Linux passes `127/127` deployment tests, actual Windows passes `13/13` plus
  both PowerShell parses, and both runtimes replay all `15,981` late-tail
  actions with unchanged maximum error `6.616115570068359e-6`. Candidate 005's
  inactive post-collision state ignored a diagnostic MAVLink-`31000` reset
  (`race_start_boot_time_ms=-1`), so recovery required the documented
  recovery-only process relaunch and window-relative login/R1 selection. R1 was
  then active with index `0` and finish `-1`. Zero-command
  `n144_six_gate_composite_shadow_007` passed: all `35/35` visible Gate-1
  samples replay within `1.3715932369235719e-7`, all 11 manifest artifacts
  match, cadence is `50.643 Hz` over `10.031 s` (`509` no-op commands), and
  cadence violations plus reset/arm/MAVLink-setpoint/disarm counts are exactly
  zero. Shadow/parity SHA-256 values are `066e29d8...`/`bfb7eef6...`.
  This authorizes exactly one bounded tag
  `n144_six_gate_composite_gate4_bounded_006`: reuse the responsive simulator,
  issue one normal policy-ready MAVLink-`31000` reset, request `60 Hz`, run at
  most `30 s`, and require/stop at official index `4`. Do not select `FullLap`
  and do not retry Candidate 006 unchanged. Candidate 006 has now run exactly
  once and is rejected. The normal `31000` reset was detected, but the vehicle
  made zero official passes and collided with ID `1001`, impact `9.0863`, at
  active index `0` after about `4.734 s`; finish stayed `-1`. The N144 floor
  reached normalized roll `0.9`/decoded `0.45 rad` during the failed approach,
  so the positive-floor transfer is not retained. Actual wire delivery was
  also only `46.204 Hz` over `4.610 s` (`214` commands), below the `50 Hz`
  floor. Exit and passive disarm passed (`base_mode=65`, system status `3`,
  index `0`, finish `-1`). Freeze live attempts while the Candidate-006 trace
  is added to an offline source-hashed separation/transport diagnosis; do not
  restore N143 or tune another Gate-1 floor from this single run without that
  evidence. That diagnosis is now complete. Candidate 006 contains five N144
  roll changes from the recurrent base; one `0.9` bank was activated only
  `1.49e-8 m` across the old `-0.3 m` threshold at unrounded precision. N145
  therefore requires direct observed right displacement `<-0.8 m` before any
  positive floor. Source-hashed projection changes none of `397` accepted
  samples, preserves all six Candidate-005 severe-miss corrections, and changes
  none of Candidate 006; Linux/Windows report hashes are `d73b0cb0...`/
  `28e02251...`. Callable SHA-256 is `9eaab396...`. The cadence diagnosis keeps
  the proven `1 ms` GIL handoff and hard `10.5 ms` minimum wire interval. An
  actual-Windows contention benchmark rejects `70` and `90 Hz` requests but
  passes `80 Hz` in `5/5` trials at minimum `52.000`, average `55.691`, maximum
  `57.860 Hz`, zero fast-rate violations. N145 changes only the six-gate wrapper
  request to `80 Hz`; the shared runner default remains `60 Hz`. Full-policy
  wrapper SHA-256 is `6dd35edc...`; N145 pinned wrapper is `c53859cb...`.
  Linux passes `58/58` focused tests and actual Windows `18/18` plus both parser
  checks. Both replay all `15,981` late-tail records with unchanged maximum
  error `6.616115570068359e-6`. Passive tag
  `n145_six_gate_composite_shadow_008` was attempted with no lifecycle/control
  command, but the post-Candidate-006 terminal state produced zero stationary
  IMU calibration samples despite `2,183` MAVLink messages and `100`
  heartbeats. A diagnostic MAVLink `COMMAND_LONG` command `31000` was sent,
  but v3385 ignored it in the post-collision `THROTTLE DOWN` state: boot/race
  telemetry did not reset and camera/IMU state did not become policy-ready.
  This is a state-precondition failure, not an N145 shadow result. The normal
  between-attempt reset remains command `31000`; never infer success merely
  from sending it.
- Preserve the responsive simulator process when terminal-state recovery is
  needed. PID `24476` was returned through Pause -> Back To Main Menu, the sole
  VQ1/R1 event was reselected, and passive telemetry proved `race_started=true`,
  active index `0`, finish `-1`, base mode `193`, and system status `4`.
  Zero-command tag `n145_six_gate_composite_shadow_009` then passed at requested
  `80 Hz`: all `32/32` inference samples replay, `28` are visible Gate-1
  samples, maximum action error is `1.4837913513143786e-7`, all 11 deployment
  artifacts match, and no-op cadence is `53.620 Hz` (`538` counts over
  `10.015 s`) with zero cadence violations/probe MAVLink setpoints. Reset,
  arm, control-setpoint, and disarm counts are all zero; active official status
  stayed index `0`, finish `-1`, with zero collision/dropout/drain-limit hit.
  Shadow/parity SHA-256 values are `f580dd0105f9d1c1ff630166bdd88cdaef6488f1db652da61b7602c7eb597e02`
  and `ca70caf55f1d22b066f95d8752e51c8ba4f761d868bf95f14b0af58b94e78b30`.
- The single next simulator interaction is exactly one bounded Candidate 007
  under tag `n145_six_gate_composite_gate4_bounded_007`: reuse the responsive
  v3385 process, issue one normal policy-ready MAVLink command `31000` reset,
  request `80 Hz`, run at most `30 s`, use repeats/min-valid `1/1`, and set
  target/min/authoritative stop index `4`. Accept only an official index-4 stop
  with finish `-1`, actual wire rate in `[50,100) Hz`, zero too-fast violation,
  all frozen hashes, no collision/invalid/dropout/drain issue, one exit disarm,
  and passive disarm proof. Any failure rejects Candidate 007 and forbids an
  unchanged retry. `FullLap` remains unauthorized.
- Candidate 007 ran exactly once and is rejected; do not retry it unchanged.
  Its one normal MAVLink command `31000` reset was sent and detected
  (`pre=2667002 ms`, `post=3405 ms`, race start `3300 ms`) with all 60
  stationary calibration samples. It made zero official passes and reached the
  `30.047 s` bound at Gate 1/index `0`, finish `-1`, without a collision or
  invalid run. The closest accepted Gate-1 pose was only `8.771574 m` at
  `4.094 s`, body vector `[8.771574, 1.973604, -2.508122]`; the detector then
  lost the real gate and the final detection age was `17.266 s`.
- Candidate-007 transport is accepted: `1,644` setpoints over `29.875 s`,
  effective `54.996 Hz`, zero rate violation; `7,559` telemetry messages,
  `3,823` HIGHRES_IMU messages, `237` processed frames, `59` detections, zero
  collision/dropout/drain-limit hit. One exit disarm was sent and passive
  follow-up proves `base_mode=65`, system status `3`, index `0`, finish `-1`.
  Aggregate/per-attempt/smoke/CSV/passive SHA-256 values are `9d70c26d...`,
  `8b37842a...`, `5e24adb7...`, `a5cfd802...`, and `91dd8838...`.
- Freeze simulator attempts. Add the complete Candidate-007 trace to a
  source-hashed offline replay against the recurrent base, N145 projection,
  Candidate 005/006, and all retained Gate-1 passes. Establish whether the
  severe-direct branch changed this flight and isolate the first trajectory
  divergence before any Candidate 008 or other live control. `FullLap` remains
  unauthorized.
- Detailed authoritative evidence and all closed branches are in
  `docs/competitive_lap_run_ledger_2026-07-16.md`. Do not retry Candidate 003,
  select `FullLap`, repeat Candidate 004, or repeat Candidate 005 unchanged.

The retained best official prefix is a recurrent `policy-attitude` controller,
not the newer deterministic `course-fsm` lineage. The user correctly recalled
that a previous controller passed Gate 2. Its artifacts exist only in the
Windows worktree log copy:

- tag: `full_policy_rl_stage_c_v6c_gate2_001`;
- evidence:
  `C:\Users\anon\code\pufferlib-drone\logs\sitl\official_policy_validation_summary_full_policy_rl_stage_c_v6c_gate2_001.json`;
- checkpoint:
  `C:\Users\anon\code\pufferlib-drone\checkpoints\drone_race_full_policy_stage_c\1783952962041\0000000001998848.bin`;
- checkpoint SHA-256:
  `25b8b886e803387dcf26a9876a2ef3370c959966ef7b74fdb2884bd71c6e7a19`;
- result: acceptance passed, official `active_gate_index=2`, two ordered gate
  passes, no collision/invalid state/dropout/rate violation, and no finish.

This baseline was reproduced unchanged on 2026-07-16 under tag
`competitive_r1_reproduce_stage_c_v6c_gate2_083`: official index `2`,
acceptance passed, no collision/invalid state/dropout/rate violation, effective
command rate `56.333 Hz`, and no finish.

The strongest historical next prefix is
`full_policy_gate3_latest_r135_dropout_001` with checkpoint
`checkpoints\drone_race_full_policy_stage_d_gate3_bc\v6c_latest_obsmatch_r135_dropout_e10.bin`
(SHA-256
`ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760`).
It reached official index `3` at `9.109 s`, then collided around `19.3 s`.
Reproduce it first in a bounded 12-second, target-3 prefix run; do not mistake
the later collision for failure to solve Gates 1-3.

That bounded reproduction passed on 2026-07-16 under tag
`competitive_r1_reproduce_gate3_r135_dropout_084`: official index `3`,
acceptance passed, no collision/invalid state/dropout/rate violation,
`55.833 Hz`, and no finish. This is the retained clean three-gate prefix. No
historical artifact in either log tree reached official index `4` or recorded a
finish; Gate 4 is the first unsolved official segment of six, not the finish.

Course-count correction after Candidate 206: the callable historically named
`policy_callable_final_gate_handoff.py` is a Gates-4-through-6 post-prefix
controller, despite its filename. Runs through Candidate 206 used target/min
index `3`, so they are only Gate-4-entry diagnostics. They never established a
Gate-4 pass and their missing finish timestamp says nothing about Gates 5-6.
Before another live run, use a six-phase denominator while preserving the
checkpoint's restored previous-yaw input, reset post-prefix association/pulse/
delay state on every official index transition, reset delayed edge preference
per gate, and set the next milestone target/min index to `4`. Do not reuse one
continuous “final phase” state across Gates 4-6.

The correction is implemented and verified: focused root suite `102 passed`;
Windows compile/direct CLI checks pass; callable root/Windows SHA-256 is
`00f5d0e3a6288170612a1c3a7e2217239bbb2a4cd9ff6bdbedd9f52ec2e50bb1`.
Candidate 207 then ran the exact safe Candidate-205 controller in five bounded
attempts under denominator `6` and target/min/stop index `4`. Attempts 1/3/5
reached index `3` cleanly; attempts 2/4 failed at Gate 3; none reached index `4`
or finished. Close the manual pulse/re-anchor/cooldown bracket. The user
explicitly selected one learned recurrent policy over all six gates as the next
mechanism. Preserve the exact deployable camera/IMU/action prefix, add legal
normalized official progress `active_gate_index / 6`, randomize unknown
Gate-5/6 geometry in training, and never deploy native coordinates. A native
six-gate finish is a screening result only; official telemetry remains final.

The six-gate native training path is implemented and tracked through N104 in
`docs/competitive_lap_run_ledger_2026-07-16.md`. The live-reproducible ABI is
camera/IMU/previous-action observations `0..22`, normalized official progress
at `23`, and reserved zeros `24..31`. N033 is the first accepted single
recurrent six-gate policy:
`checkpoints/drone_race_full_policy_six_gate_bootstrap/n033_scheduled_dagger/n031_expert_plus_student25_e8_lr5e5.bin`.
Scheduled-sampling DAgger (`25%` student/`75%` expert execution) produced
exact-512 success `100%` at radius `3.0` and `96.68%` at radius `2.75`, with
zero crash/timeout. Later decoder calibration and bounded PPO produced the
retained checkpoint parent N058:
`logs/drone_race_full_policy_six_gate_bootstrap/ppo_n057_r0p95_joint/drone_race_full_policy_six_gate_bootstrap/1784270090263/0000000001671168.bin`
(SHA-256
`973ed7bd6f77665731691b5c1811fcc6da65c4a9a9fb350dea29a37c550dc2df`).
It completes `412/512` at global radius `0.95` and `455/512` at radius `1.0`,
with zero crash/timeout. The current N071 environment-only curriculum state
keeps gate indices 0 and 5 at `0.75`, index 3 at `0.875`, and indices 1, 2,
and 4 at `0.95`; unchanged N058 completes `411/512`. N072's targeted-radius
PPO child peaked at `399/512` on index-3 radius `0.75` and is rejected. The
N073 FP32-direct collector repair now reproduces native common-seed outcomes
exactly; every older converted-layout N044 dataset remains excluded. N074's
broad spline teacher failed all Gate-4 entries. N075's policy-residual files
were invalidated after a byte audit exposed omitted target-contract settings.
With the corrected contract, unchanged N058 completes `97/128`; N076's best
Gate-4-only relative-state feedback completes `93/128`, and N077's best
gate-indexed constant bias completes `96/128`. Both are rejected without
fitting or exact-512 promotion. Relative-state/PD feedback and the local
Gate-4 bias branch are closed; do not repeat them.
N078 then found a causal Gate-3 (native index 2) output adapter and passed the
tight mixed target at `410/512`. Successive accepted aperture/roll rungs N081
and N083 retain the same N058 checkpoint with a deployable index-2 roll
`-0.0125` and thrust `+0.00125` bias. N084 is the current environment:
radii `0.75/0.95/0.85/0.75/0.95/0.75`, exact-512 success `412/512`, zero
crash/timeout. The sparse per-gate table is implemented in the C collector and
Python/Windows callable; N083's table form is byte-identical to its legacy
single adapter on common-seed data. N085 failed to promote index-2 radius
`0.825`. N086's attempted Gate-2 bias scored only `407/512` exact. N089 then
accepted a center-safe second entry: native index 1 adds roll `+0.0025` and
thrust `-0.00125` only at observable alignment `>=0.70`, while index 2 keeps
roll `-0.0125` and thrust `+0.00125`. N089/N084 scores `413/512` at the tight
fixed course and `416/512` at index-2 radius `0.875`, zero crash/timeout.
N090-N093 reject tighter Gate-2/Gate-5 apertures and local release variants.

The user then redirected work to genuine end-to-end six-gate training. N094
implemented a 50/50 nominal/randomized late-gate anchor mixture but was
rejected. N095 separated late-course sampling from the vehicle RNG, proving
that geometry randomization cannot silently change the solved initial state.
N096 trains raw N058 without any runtime adapter on all six ordered gates. The
retained raw randomized-geometry parent is
`logs/drone_race_full_policy_six_gate_bootstrap/ppo_n096_raw_anchor_mixture/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784286271692/0000000000491520.bin`
(SHA-256
`d580728192e42647607800083d7c6809f038173672f5033f6ee36fbdfbe68dd9`):
randomized common-seed success improves `9/128 -> 12/128`, average gates
`4.0546875 -> 4.1953125`, and exact-512 fixed N071 retention is `413/512`,
zero crash/timeout. N096 supersedes N058 only as the raw randomized-geometry
curriculum parent; N089/N084 remains the tighter fixed-course adapter parent.
N097's randomized-only `1.5 m` late teaching aperture regressed every exact-
aperture child; its best result was `11/128` and `4.1015625` average gates.
N098 then identified that gate-exit velocity rewards used a randomized next
gate before the policy could observe it. The backward-compatible
`gate_exit_reward_skip_randomized_next_gate` guard is implemented and native-
tested, but PPO with that correction also regressed: best `11/128`, average
`4.140625`, zero crash/timeout. Reject all N097/N098 children and retain N096;
do not repeat aperture widening or reward suppression alone.
N099-N101 then test a genuinely different elite self-imitation optimizer.
The collector can default-off record executed actions and retain only complete
successful episodes; a real smoke retained exactly five of eight successes.
N099 is rejected for three exploratory crashes and target-seed overlap. N100
uses disjoint seeds and half noise, retaining 190 successes; its one crash
matches deterministic N096's paired one-crash baseline, so N101 fit is allowed.
All three frozen-recurrent ridge children regress badly on held-out randomized
seeds: best `6/128`, average `3.75`, zero crash/timeout. N102 prevents prefix
leakage by selecting each learned decoder only at observable phases 4-5, but
still tops out at `11/128` and `4.164062`. Reject every N099-N102 child, retain
N096, and do not repeat or repackage elite action-perturbation imitation.
N103 tested observable-phase-aware recurrent replay while keeping one raw actor,
full-course starts, N096's exact course/reward contract, and legal progress
observation 23. All 32 children had zero crash/timeout. Best completion was
`11/128` with `4.15625` average gates; best progress was `4.1875` with
`10/128`, both below N096. Reject N103, retain N096, and do not repeat phase-
priority weighting alone. The default-off terminal-prefix-safe implementation
is tested infrastructure only; it authorizes no FlightSim run.
N104 is the retained raw six-gate native policy. It adds deployable normalized
progress plus a six-way active-gate observation in slots `23..29`, with
`30..31` reserved, then trains only encoder columns `27..29` for active Gates
4-6. A bitwise audit confirms no change to the solved Gates-1-through-3 encoder
columns or any shared MinGRU/decoder/action/value/logstd parameter. The initial
screens were invalid because a stale standalone collector emitted zeros in
`24..29`; do not use those ties. The rebuilt collector reproduces N096 exactly
at `12/128`, `4.1953125`, zero crash/timeout, then promotes N104 step
`1,048,576` at `17/128`, `4.21875`, zero crash/timeout. Step `983,040` also
passes at `16/128`, `4.2109375`. The control and both children each retain
fixed N071 at exact `413/512`, `5.21484375` average gates, zero crash/timeout.
Retained checkpoint:
`logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784295732298/0000000001048576.bin`
with SHA-256
`e248ca36a46d422a18c91afbf98be9b15a1c7191b42e6a189afc6456861b4c9e`.
N105's prebuilt `4/8/16/32` gain children remain quarantined and are canceled
without a valid screen: that ladder was registered to diagnose the invalid
flat result, and N104 now promotes without amplification. Do not select a gain
after this corrected outcome.
N104 offline callable parity passes on Linux and the actual Windows Python
runtime over an eight-agent, `2,368`-step native trace containing all six
phases: action max error `1.73e-6`, recurrent-state max error `7.16e-7`, and
all eight terminal resets match. This clears FP32 layout, phase ABI, state, and
callable inference parity only. It does not clear live camera-domain transfer;
the Windows controller worktree also has older runner/callable hashes and must
be packaged from the hashed current source before official use.
N106 then rejects immediate deployment: retained N104 scores only `62/128`
(`0.484375`) at fixed radius `0.75 m` on all six gates, average gates `3.75`,
zero crash/timeout. Terminal active phases `[0,24,42,0,0,62]` localize every
failure to Gates 2/3; every episode clearing them finishes. Do not run N106
exact-512 or FlightSim. The next bounded training mechanism may update only
one-hot encoder columns `25..26` for active Gates 2/3 while keeping Gate 1,
learned late columns `27..29`, and every shared parameter exact.
N107 executes that bounded continuation. All 33 files are byte-exact outside
columns `25..26`; best step `327,680` causally improves the common-128 target
to `74/128`, `4.1328125` average gates, zero crash/timeout. It fails exact-512
retention at only `284/512` (`0.5546875`), average gates `4.017578125`, zero
crash/timeout. Reject every N107 child and retain N104. N108 then tests the
anchored full-network aperture curriculum from N104 at `1e-7 -> 1e-8`; its
best common-128 child reaches only `64/128`, `3.828125`, zero crash/timeout, so
all N108 children are rejected. N109 uses rejected N107 step `327,680` only as
a causal curriculum parent and tests the one registered full-network
`1e-6 -> 1e-7` rung. Step `229,376` narrowly passes common-128 at `76/128`,
`4.1484375`, zero crash/timeout, but exact-512 collapses to `288/512`
(`0.5625`), `4.017578125`, zero crash/timeout versus required `410/512`.
Reject every N109 child, retain N104, and close direct PPO continuation of this
phase-column direction. Do not add another learning-rate rung after seeing
these results. Current-source Windows packaging/hash parity, fresh exact-v3385
reset recovery, and passive live Gate-1 parity now pass for retained N104. The
refreshed shadow captures and replays all `257/257` recurrent ticks at max
action error `4.35e-7`, matches all seven source hashes and the checkpoint, and
sends zero reset/arm/setpoint/disarm commands. Bounded Candidate
`n104_six_gate_gate1_bounded_001_post_relaunch` then advances official index
`0 -> 1` at `5.032423496 s`, triggers the index-1 stop at `5.016 s`, and passes
with no collision/invalid state/dropout/drain/rate violation. Effective command
rate is `55.423 Hz`; a passive follow-up proves disarmed `base_mode=65`, index
`1`, finish `-1`. The next and only authorized live flight is one unchanged
N104 reset attempt bounded to target/min/stop official index `2`. It must be
adjudicated before any Gate-3 extension; an unbounded lap remains unauthorized.
The fixed/randomized Gates 5-6 are synthetic completion-curriculum anchors
only, never official coordinates. No N-series checkpoint is authorized for
unbounded official flight until aperture/domain and native-to-live promotion
gates pass.

The Gate-4 miss is localized: the parent crosses the final gate plane roughly
`4-5 m` right of center, then reverses and collides. The bounded legal handoff
callable is `scripts/policy_callable_final_gate_handoff.py`. It preserves the
checkpoint for Gates 1-3 and uses only phase index, camera-relative gate
pose and IMU attitude at index `3`. Candidate 087 proved the corrected raw-pose
yaw sign in a bounded run: official index `3`, no finish/collision/health
violation, with Gate 4 still `25.26 m` ahead and bearing reduced to
`0.1914 rad` from Candidate 086's `0.680 rad`. It also stopped closure, so it is
an accepted alignment parent, not an approach solution. The next run must use
an explicit pre-crossing stop guard and remain incapable of passing Gate 4
while finish-lock semantics are unknown. Exact evidence and rejected variants
are in `docs/competitive_lap_run_ledger_2026-07-16.md`.

Candidate 098 initially passed official index `3` cleanly in `3/3` with an
associated pose, `15 m/+0.45 rad` centering bank, and projected `-0.8 m` switch
to `-0.50 rad` counter-bank. Larger evidence reopened reliability: Candidate
101 added a high-speed left miss; Candidates 102-106 stalled; Candidate 108's
`9.7 m/s` brake passed only `3/5`; Candidate 110's hysteretic governor passed
only `2/5`. Candidate 098 remains the best simple Gate-3 parent, not a reliable
promotion. Candidate 111's gentle far-phase pitch brake also failed at Gate 3
and is closed: it coupled into the recurrent action history and materially
changed lateral/vertical intercept geometry. Candidate 112 proved that hiding
the externally corrected action did not remove that physical coupling and was
also rejected. Do not tune another pitch brake or action-shadow variant. The
active preregistered child is Candidate 113's `20-15 m`, `>=7.0 m/s`
high-speed-only early `+0.45 rad` lateral bank. Candidate 113 exercised that
branch at `8.3+ m/s` and passed official index `3` cleanly. Candidate 114 then
passed the exact reliability batch `5/5`; two runs exercised the high-speed
branch and three preserved normal behavior. Combined evidence is `6/6`, with
`3/3` exercised and `3/3` unexercised. Promote this controller as the frozen
Gate-3 prefix; read the dated ledger before changing it.

Gate-3 entry reliability is restored at the preregistered threshold, so bounded
Gate-4 diagnostics may resume. Candidate 087 remains the best bounded Gate-4
alignment evidence and Candidate 100 proved hover-altitude wall clearance, but
neither is a final approach. Candidate 115 live-tested the `8 m` raw continuity
bound with the promoted prefix: it cleared the wall and rejected later-gate
steering reversals, but the active near gate remained cropped at the lower
image edge. The next narrow child may add only delayed vertical recovery after
the wall-clearance hover; keep continuity and the hard guard exact. Candidate
116 proved a three-second-delayed `0.22` descent is collision-free and reduces
apparent vertical error, but the retained `-0.12 rad` brake reversed away from
Gate 4. The next child may replace only that delayed brake with `+0.06 rad`
approach pitch. Candidate 117 tried that and collided with ID `1002` while the
active near gate was cropped at the right edge; open-loop recovery is closed.
Candidate 118 returns to Candidate 115's safe control and changes only
final-phase perception to prefer a dominant cropped edge frame over tiny
downstream apertures. It cleanly triggered the live stop at `18.78 m`, proving
the partial-frame guard. Its yaw remained stale because the first transition
pose seeded continuity and the true Gate-4 pose jumped `9.0 m` only `0.125 s`
later. The next child may add only a `0.5 s` handoff association grace before
restoring the exact `8 m` bound. Candidate 119 validated that grace: near Gate-4
right error improved from about `9.5` to `3.5 m` and bearing from `0.43` to
`0.17 rad`, while late aliases remained rejected. Candidate 120 may recombine
only the previously safe delayed `0.22` descent; forward recovery stays closed.
Candidate 120 never activated that final logic: it exposed the first Gate-3
failure after 11 clean prefix runs because the `7.0 m/s` early-bank threshold
did not fire until `16.74 m`. Candidate 121 lowered only that threshold to the
replay boundary `6.5 m/s` and passed the exact Gate-3 batch `5/5` with no
collision, invalid state, or finish; promote and freeze it. Candidate 122 still
failed Gate 3 after the branch correctly began at `18.46 m`; it remained
`2.59 m` left at the last fresh pose and never counter-banked, so final logic is
still untested. Candidate 123 may add only a severe-intercept lateral rule:
maximum `+0.50 rad` roll when a `<=22 m`, `>=5.0 m/s` trajectory projects more
than `3.5 m` left at the 2 m plane. Keep pitch, thrust, yaw, recurrent state,
final logic, and the index-3/`20 m` stop exact. Candidate 123 exercised the rule
from `20.43-17.75 m` and passed cleanly at official index `3`, but Candidate 124
failed the exact reliability batch `3/5`. Its severe collision released maximum
bank too early and hit frame ID `1001` only `0.195 m` outside the counter
threshold; another run stalled at index 1 before Gate-3 logic. Candidate 125
changes only the abrupt severe release to a continuous roll taper from `+0.50`
at `-3.5 m` projected miss to `+0.45 rad` at the retained `-0.8 m` counter
threshold. Candidate 125 exercised the taper and passed Gate 3 cleanly at
`9.406 s`; one inherited counter-threshold bounce occurred without sustained
oscillation. Candidate 126 met the unchanged promotion minimum exactly `4/5`.
Its failure was nearly centered (`0.16 m` left/`0.03 m` vertical/`0.048 rad`
roll) but still hit frame ID `1001`, so this is only a guarded-diagnostic prefix,
not a final reliability claim. Candidate 127 may combine it with exact Candidate
119 association plus the still-untested delayed `0.22` descent, zero recovery
pitch, and the mandatory index-3/`20 m` stop. Candidate 127 failed before final
activation: at about `19.2 m`, closure `4.40-4.90 m/s` sat just below the
adaptive `5.0` threshold, roll dropped to `+0.282-0.297 rad`, and the last pose
remained `1.26 m` left. Candidate 128 changes only the adaptive threshold to
`4.0 m/s`; it exercised at `19.59 m`/`4.47 m/s` and passed official Gate 3 at
`9.266 s`. Candidate 129 met the exact ten-reset gate exactly `8/10`: one
failure remained `1.36 m` left without counter, while the other was nearly
centered but recorded a tiny frame impact. Promote only for one guarded
diagnostic. Candidate 130 combines it with exact Candidate 119 association plus
the delayed `0.22` descent, zero recovery pitch, and the mandatory index-3/
`20 m` stop. Candidate 130 finally proved the descent: near Gate-4 down error
reduced about `16.7` to `9.2 m` and the guard fired cleanly at `18.78 m`.
Association yaw stayed stale because the 8 m consecutive-pose test can walk
through unlogged aliases at 60 Hz. Candidate 131 adds only a fixed 12 m
post-grace anchor bound alongside the 8 m step bound; it validated live, improving
guard geometry to `3.19 m` right/`0.158 rad` bearing, though down error remained
`9.94 m`. Candidate 132 retains the three-second wall-clearance delay and changes
only post-delay thrust `0.22` to `0.18`; it reduced down error to `5.8 m` but
collided with intermediate structure ID `1002`, so reject it. Candidate 133 tests
midpoint `0.20` and also hit ID `1002`, closing stronger descent at safe `0.22`.
Candidate 134 adds only final lateral roll (`kp=0.04`, max `0.15 rad`) to reduce
right error/move around the wall; it cleared the wall and reached `1.99 m` down
but overshot to `14.76 m` left. Candidate 135 retains stale-roll leveling and
tests only the lower scale `kp=0.01`, max `0.05 rad`; it reached official index
`3` but collided with the same intermediate structure ID `1002` at `15.750 s`,
so that authority is below the clearance threshold. Candidate 136 tests only
the arithmetic midpoint `kp=0.02`, max `0.10 rad`, retaining the exact anchor,
safe delayed descent, stale-roll leveling, and mandatory stop. Candidate 136
was clean but stopped early at `17.78 m` on the returned active-right pose after
the `0.5 s` grace had fixed a farther aperture; the rejected pose froze yaw and
leveled roll. Candidate 137 changes only association grace to `1.0 s`, which
covers the measured `~0.8 s` active-gate return but ends before Candidate 100's
documented `~1.85 s` later-gate aliases.
Candidate 137 validated that single change: clean guard stop at `16.00 m`
forward with `2.225 m` right/`0.138 rad` bearing/`7.650 m` down, improving all
three geometry terms over Candidate 131. Candidate 138 is an exact five-reset
reliability batch with a `4/5` clean promotion threshold; it achieved only
`3/5` clean. One failure was Gate-3 ID `1001`; the other passed Gate 3 then hit
ID `1002` while the final roll saturated near `-0.10 rad`. Candidate 139 keeps
`kp=0.02` and changes only the cap to `0.12 rad`, so errors at/below `5 m` are
unchanged while larger early errors gain clearance authority. Do not count a
wrapper progress pass when an individual attempt is collision/invalid/finish.
Candidate 139 exercised `0.12 rad` but still hit ID `1002` after more than three
seconds at descent thrust; cap tuning is closed. Candidate 140 restores the
`0.10 rad` cap and adds only disabled-by-default observable range gating:
`PUFFER_FINAL_GATE_DESCENT_MAX_FORWARD_M=25`. After the retained three-second
delay, descend only if the last associated active gate is also within `25 m`;
otherwise hold hover. Callable hash is
`26e7320fe65a8b256e1920c9d1aeae8dae937ef139d71142bee73dff3f9be9c8`,
root/Windows exact, with `70 passed` and successful Windows compilation.
Candidate 140 was clean and held hover to a `19.636 m` guard, but stopped before
the three-second delay elapsed, so the new range branch was not yet exercised.
Candidate 141 is an exact five-reset reliability batch requiring `4/5` clean;
it must include live evidence that a long path stays at `0.27` after three
seconds while associated range remains above `25 m`, with no post-Gate-3 ID
`1002` contact.
Candidate 141 reached `4/5` clean but its fifth run hit ID `1002`: once an
accepted `~22 m` pose enabled descent, rejected/far frames reused the in-range
last vector and held stale `0.22`. Candidate 142 fixes only enabled range-gate
semantics: dropout/rejection/current-far state explicitly restores `0.27`, and
descent requires a current accepted `(0,25] m` pose. Callable hash is
`dc97e2cfdb85c6c1f3976a6273e6cb5d1fd834390ba9d80b5396efc5f0190b3a`,
root/Windows exact, with `70 passed` and successful Windows compilation.
Candidate 142 live-validated the fix: after the three-second delay, rejected/far
current poses down to `21.1 m` all restored/held hover `0.27`; zero stale
descent samples, no collision/finish, and a clean `19.592 m` guard. Candidate
143 is an exact five-reset reliability batch requiring `4/5` individually
clean and allowing no post-Gate-3 ID `1002` or stale descent.
Candidate 143 promoted exactly `4/5` clean. Attempts 1-4 had zero final-wall
contacts, stale descent, invalid state, or finish; one ran the full `18 s` at
hover and three stopped at the guard. Attempt 5 was only the known pre-handoff
Gate-3 ID `1001` tail. Retain Candidate 143 as the safest diagnostic, not a
completed-lap claim: the hard stop remains, and Gate-3 reliability is still
`4/5` here (`8/10` in Candidate 129).
Candidate 144 is the first deliberate crossing-capable attempt: exact Candidate
143 controller, one reset, target/index `4`, `45 s`, with the pre-crossing guard
disabled and the new immediate official-finish stop enabled. No scalar changes.
If it finishes, archive the official time and prove MAVLink `31000` clears the
finish state before any second flight. Earlier diagnostics retain the guard;
only explicitly pre-registered finish attempts may omit it.
Candidate 144 failed before final activation at official index `2`, ID `1001`,
impact `5.9434`; finish and Gate-4 logic were untested. Its exact Candidate 145
retry also failed at Gate 3 only: official index `2`, ID `1001`, impact
`1.7020` at `9.125 s`, no finish, healthy `57.425 Hz`. The two failures do not
change any Gate-4 conclusion. Candidate 146 was the final exact sampling retry.
It passed Gate 3, ran the full `45.0 s` collision-free at official index `3`,
and did not finish; transport was healthy at `56.711 Hz`. Final descent occurred
for only five 8 Hz trace samples (`0.625 s`) before current-association rejection
restored hover, after which the gate left view. Candidate 147 is a guarded
final-phase diagnostic that adds only a one-shot `2.0 s` descent pulse: it may
start only after the existing `3.0 s` delay and a currently associated
`(0,25] m` gate with down error `>1.5 m`, may bridge dropout for the remainder
of that pulse, never restarts, and restores hover at expiry. Callable hash is
`0330567714776a0512f10823f4970cf1ae9d300ad97dbc951a7659fa83035e22`;
focused verification is `73 passed`, Windows compilation succeeds, and root/
Windows hashes match. Candidate 147 restores target/index `3`, `18 s`, and the
index-3/`20 m` hard stop; it is not finish authorization.
Candidate 147 never activated final control: local vision counted a third pass,
but authoritative state remained index `2` and Gate-3 collision ID `1001`,
impact `11.0876`, ended the run at `9.250 s`. Candidate 148 is one exact
activation retry with no code or scalar change; no pulse conclusion may use
Candidate 147.
Candidate 148 was clean at official index `3` and stopped at the configured
`19.2 m` guard, but the pulse did not trigger because current association had
not reconnected. Candidate 149 changes only the diagnostic guard from `20 m` to
`15 m`; controller code/scalars and the one-shot two-second cap remain exact.
Candidate 149 then ran the full `18 s` clean at official index `3`, but still
never re-associated or pulsed; guard-only sampling is closed. Candidate 150 adds
one disabled-by-default, one-shot re-anchor: a rejected raw cluster must remain
step-coherent for `0.75 s` and reach `(0,25] m`; then the anchor may move once
for that final phase. Large jumps reset persistence, dropouts clear it, and a
second re-anchor is forbidden. Hash
`2c898fa629708e0c783a579feb47c66ffbf0339b1de292d9cd52b85216db59a0`;
`75 passed`, Windows compilation succeeds, root/Windows hashes match. Candidate
150 keeps the two-second one-shot descent and `15 m` diagnostic stop; no finish.
Candidate 150 passed: full `18 s`, official index `3`, zero contact, one
re-anchor, one exact two-second descent pulse, then 23 sampled hover actions.
Candidate 151 retains one-shot descent but allows at most two `0.75 s` coherent
re-anchors inside a separate `30 m` re-anchor range; descent still requires
`(0,25] m`, so the second re-anchor cannot restart thrust. Hash
`a23869da02c2f381d2a54d797e76183d164d3703431928c5877a82bd40593c5c`;
`76 passed`, Windows compilation succeeds, hashes match. It remains an `18 s`,
index-3, `15 m`-stop diagnostic with no finish.
Candidate 151 stalled safely before handoff at official index `2` for `18 s`,
with no collision/invalid/health fault; two-epoch logic was untested. Candidate
152 is one exact activation retry with only the evidence tag changed.
Candidate 152 passed Gate 3 but the `15 m` raw guard fired at `9.219 s` on the
stale `1.50 m` Gate-3 pose before any final policy sample. Candidate 153 disables
only that diagnostic guard for `18 s`; target/index remains `3`, controller is
exact, and immediate official-finish stop remains enabled.
Candidate 153 passed custom acceptance: full `18 s`, official index `3`, zero
contact, two coherent re-anchors, one exact two-second pulse, 16 post-pulse hover
samples, and live yaw/roll restored at the second cluster. Candidate 154 allows
at most two two-second pulses separated by at least `1.0 s` hover; each still
needs a fresh associated `(0,25] m` low gate. Re-anchors stay capped at two.
Hash `5636035dafce1f8e98798690765bac2de628e52327a7fdd9570270540658d1ee`;
`77 passed`, Windows compilation succeeds, hashes match. Candidate 154 is a
`22 s`, target/index-3, unguarded diagnostic; immediate finish-stop remains.
Candidate 154 stayed clean for `22 s` and reached the best final geometry yet
(`7.45 m` forward, `2.84 m` down), but only one pulse fired because a third
coherent cluster exceeded the two-re-anchor cap. Candidate 155 sets re-anchor
max count to `3` and adds fresh-pose early pulse termination at the existing
`1.5 m` vertical tolerance; pulse count/duration/cooldown remain `2/2 s/1 s`.
Hash `78d3c268676a4f5d42058a770d492c5253f00b44d0449bb163514e513efe0fa7`;
`78 passed`, Windows compilation succeeds, hashes match. Run `22 s`, target/
index `3`, no raw guard, immediate finish-stop.
Candidate 155 failed only at Gate 3: official index `2`, ID `1001`, impact
`11.0822` at `9.047 s`; final logic was untested. Candidate 156 is one exact
activation retry with only its evidence tag changed.
Candidate 156 was clean at official index `3` for `22 s` but fail-closed at
hover: no coherent near cluster triggered a pulse. Candidate 157 is an exact,
maximum-five-reset reliability batch of the Candidate-155 controller. The outer
runner now supports `--stop-after-official-finish`; it records the finish and
halts the entire batch before another reset/flight. Focused verification is
`80 passed`, and Windows compilation succeeds. No tuning occurs within the batch.
Candidate 157 had official index-3 progress on `4/5` and `3/5` individually
clean attempts, but the sole two-pulse attempt hit ID `1002` at `18.719 s`,
impact `0.0462`, just after its second full two-second pulse; reject equal pulse
durations. Candidate 158 keeps the first pulse at `2.0 s` and caps later pulses
at `1.0 s`; all triggers/cooldowns/counts/re-anchors stay exact. Callable hash
`4d150ad4c797fa009187ebc3e37fcc5580c5965e16e3ace6144863248ab45686`;
focused verification is `81 passed`, Windows compilation succeeds, hashes match.
Candidate 158 still hit ID `1002` at `17.234 s`, impact `0.7683`, after only
`0.625 s` sampled time in pulse 2; it began too far away at `24.62 m`. Candidate
159 requires later pulses inside `10 m` while retaining `1.0 s` duration; pulse
1 still uses `25 m`/`2.0 s`. Hash
`791eb85bff431e9b9167a25ebfce6fb36a91ba4442b4a5f1023ac698cd1ec336`;
`82 passed`, Windows compilation succeeds, hashes match.
Candidate 159 used only pulse 1 and still hit ID `1002` at `18.500 s`, impact
`0.1955`, while the target remained `21.33 m` forward; any descent outside the
foreground wall is rejected. Candidate 160 sets the primary descent range to
`10 m` too, so all pulses are close-only; re-anchor tracking stays `30 m` and
all pulse duration/count/cooldown settings remain exact. Candidate 160 passed
cleanly at official index `3` for the full `25 s`: no contact, invalid state,
finish, or descent pulse. It proved hover is safe outside `10 m`, but braking
pitch created a deadlock—the nearest accepted low-gate family stayed around
`20 m` forward/`10.25 m` down, then receded and left view. Candidate 161 adds
only a `+0.03 rad` hover-altitude approach on a current associated low-gate pose
in `(10,30] m`, after the retained `3 s` delay. Missing/rejected poses force
`-0.12 rad` braking and hover; descent remains current-pose-gated to `(0,10] m`.
Callable hash is
`74cd28981020aef99b9bd94626016a25b620e02d12f7bbdeb60b82559f3bc27d`;
`83 passed`, Windows compilation succeeds, root/Windows hashes match. Run one
`25 s`, target/index-3 attempt with no raw guard and immediate finish-stop.
Candidate 161 passed generic acceptance at official index `3` for `25 s`, with
zero contact/invalid/finish and `57.2 Hz` commands. The branch fired on 22
trusted samples at hover, never stale-held through dropout, improved closest
accepted final range from about `20 m` to `16 m`, but did not reach `10 m`; no
descent occurred. Candidate 162 brackets only approach pitch to `+0.06 rad`,
retaining every range, thrust, association, descent, and prefix setting.
Candidate 162 failed before final activation at official index `2`, ID `1001`,
impact `10.6418` at `9.094 s`; the `+0.06 rad` branch was untested. Candidate
163 is one exact activation retry with only the evidence tag changed. Candidate
163 ran clean at official index `3` for `25 s` but did not improve closure: 18
trusted `+0.06 rad` samples stayed at `24-29.29 m`, the active partial frame
exited bottom-right, and no descent/finish occurred. Reject `+0.06`, restore
`+0.03`. Candidate 164 changes only first-pulse range from `10` to `17 m`;
later descent remains `10 m`. This is below rejected `21-25 m` starts and just
above Candidate 161's trusted `16 m` opportunity. Candidate 164 failed before
handoff at official index `1`, ID `1001`, impact `10.7113` at `6.359 s`; the
`17 m` threshold was untested. Candidate 165 is one exact tag-only retry.
Candidate 165 ran clean at official index `3` for `25 s`, but its 20 approach
samples stayed at `21.6-29.09 m`; no pulse/finish occurred, so the threshold is
still untested. Candidate 166 is an exact maximum-five-reset opportunity batch,
with no tuning and whole-batch stop on any official finish.
Candidate 166 exhausted all five attempts with no finish: attempts 1/3/5 were
clean index-3 runs, attempt 2 was an immediate index-0 ID `1002` reset-state
contact, and attempt 4 was Gate-3 ID `1001`. No pulse fired. Attempt 3 exposed
the blocker: centered low `frame_partial` samples at `16.62 -> 16.30 m` lasted
only `0.25 s`, so fixed-anchor rejection plus the `0.75 s` general re-anchor
missed the valid descent opportunity. Candidate 167 adds one close-only fast
re-anchor: `0.125 s`, `(0,17] m`, `|right|<=2 m`, low gate, after delay, one use.
Hash `e59dbe7ccc643ca278bf4e588d9ab123cb03cf703e12c20aa8c2fc1e834486da`;
`84 passed`, Windows compilation succeeds, hashes match.
Candidate 167 failed before final activation at official index `2`, Gate-3 ID
`1001`, impact `3.6712` at `9.031 s`; close re-anchor logic was untested.
Candidate 168 is one exact activation retry with only the evidence tag changed.
Candidate 168 also failed at Gate 3, index `2`, ID `1001`, impact `5.5783` at
`9.188 s`; final logic remained untested. Windows was measured at `93-99%` CPU
on four logical processors while controller delivery had fallen to `39-44 Hz`.
Do not kill unrelated apps or relaunch the healthy sim. Candidate 169 changes
only transient controller-process priority to High; controller/config stay exact.
Candidate 169 improved delivery to `49.138 Hz` but missed its `>=50 Hz` rule and
still failed at Gate 3, ID `1001`, impact `1.0489` at `9.219 s`; reject priority
as a fix. Candidate 170 is an exact maximum-five-reset Candidate-167 batch at
normal priority, no tuning, with whole-batch stop on any official finish.
Candidate 170 exhausted five attempts with no finish/pulse: three clean index-3
runs, one Gate-3 ID `1001`, one collision-free index-1 stall. Clean raw minima
were `19.64/19.64/24.69 m`; the centered `19.64/0.92/9.70 m` partial frame was
the closest post-delay opportunity. Candidate 171 brackets first range to
`20 m` but cuts first/later pulses to `0.5/0.5 s`; later range remains `10 m`.
Run a fixed maximum-five batch, no tuning, whole-batch finish stop.
Candidate 171 achieved `4/5` clean and one safe half-second pulse. That attempt
reacquired near `19.6 m`; down error improved from about `9.5-9.9` to
`7.4-8.1 m`, with no contact/finish. Candidate 172 changes only later-pulse
range from `10` to `20 m`, allowing one feedback-separated second half-second
pulse; max count stays two. Fixed maximum-five batch, no tuning, finish-stop.
Candidate 172 achieved one clean two-pulse attempt: pulses were separated and
bounded, no contact, then the active frame closed to `7.45 m` with about
`4.10 m` down; max count prevented a third eligible correction and no finish
occurred. Candidate 173 changes only maximum pulse count `2 -> 3`. Fixed
maximum-five batch, no tuning, whole-batch finish-stop. Candidate 173 produced
one clean index-3 handoff in five attempts but no pulse or finish. Its coherent
post-delay low-gate cluster reached `16.62/2.13/7.89 m` and then
`18.19/2.61/8.95 m` forward/right/down; the `|right|<=2 m` close-re-anchor
predicate alone excluded it. Candidate 174 changes only that separately
one-use close bound from `2` to `3.5 m`; three half-second stages, range,
cooldown, thrust, association, prefix, fixed maximum-five batch, and finish-stop
remain exact. Candidate 174 achieved `3/5` clean index-3 runs and one safe
half-second pulse; no finish occurred. The exercised run stayed collision-free
through `25 s`, but a later close family could not reauthorize because the fast
re-anchor's one-use budget was consumed. Candidate 175 adds only configurable
`PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_COUNT=2` (default remains `1`); the bound
stays `3.5 m` and pulse maximum stays three. Callable SHA-256 is
`1d22e54281dfa50190693c56167dc37da0fe5a1f5d7571429e1497f177b2a351`;
`87 passed`, Windows compilation succeeds, and runtime hashes match. Run a
fixed maximum-five batch with no tuning and whole-batch finish-stop.
Candidate 175 yielded two clean index-3 runs but no close pose below `25.95 m`,
so neither a pulse nor the second close-re-anchor budget was exercised; three
other attempts failed at Gate 3. Candidate 176 retains count `2` and changes
only the close lateral bound `3.5 -> 4.0 m`, directly covering Candidate 174's
measured second family at `3.72-3.94 m`; the `0.10 rad` roll cap and all other
settings remain exact. Run a fixed maximum-five batch, no tuning, finish-stop.
Candidate 176 achieved `3/5` clean index-3 runs and no finish. Its best run
safely executed all three half-second stages, reached about
`6.31/4.74/3.18 m`, and retained a current associated target after cooldown at
`7.45/4.72/3.42 m`; maximum count three alone blocked a fourth eligible stage.
Candidate 177 changes only descent pulse maximum `3 -> 4`; close count `2`,
bound `4 m`, all timing/range/thrust/association/prefix settings, fixed maximum-
five batch, no tuning, and whole-batch finish-stop remain exact.
Candidate 177 yielded two clean index-3 runs but no post-delay pose inside
`20 m`; three attempts failed in the prefix, so no pulse and no fourth-stage
evidence occurred. Candidate 178 is one exact tag-only fixed maximum-five
Candidate-177 opportunity batch with no tuning and whole-batch finish-stop.
Candidate 178 achieved two clean index-3 runs and no finish; its exercised run
used two safe stages. About `0.25 s` after stage 2, the target was still current
at `18.00/4.47/8.78 m`, but the `1 s` cooldown blocked stage 3; it then moved
beyond `20 m` before eligibility. Candidate 179 changes only cooldown
`1.0 -> 0.25 s`; half-second ceiling, maximum four, fresh authorization,
range/thrust/association/prefix, fixed maximum-five batch, and finish-stop stay
exact. Any contact rejects the shorter cooldown.
Candidate 179 achieved two clean index-3 runs and no finish/contact, but its
multi-stage run still separated starts by about `2.5 s`; the `0.25 s` cooldown
never enabled a stage and gave no vertical advantage. Restore retained cooldown
`1 s`. Candidate 180 uses the four-stage Candidate-178 parent and changes only
bounded descent thrust `0.22 -> 0.20`; half-second duration, max four, cooldown,
fresh authorization, association/prefix, fixed maximum-five batch, and finish-
stop remain exact. Reject on any post-handoff contact.
Candidate 180 produced one clean index-3 run with two safe `0.20` stages, but
best down error was only `7.73 m`, worse than Candidate 176's retained `3.18 m`;
restore `0.22`. Candidate 181 uses exact Candidate-178 pulse settings and
changes only final lateral roll cap `0.10 -> 0.12 rad` to counter measured
rightward drift before target loss. Gain, thrust/cooldown/count/range,
association/prefix, fixed maximum-five batch, and finish-stop remain exact.
Reject on any post-handoff contact.
Candidate 182 achieved `4/5` clean index-3 runs and no finish, safely exercising
one-second first stages repeatedly and a later half-second stage twice; no post-
handoff contact occurred. Best down error remained about `6.49 m`, so Candidate
183 changes only first-stage ceiling `1.0 -> 1.5 s`; later stages remain `0.5 s`,
and max four/cooldown/thrust/range/association/prefix, fixed maximum-five batch,
and finish-stop stay exact. Reject on any post-handoff contact.
Candidate 183 achieved `4/5` clean index-3 runs and no finish, safely exercising
1.5-second first stages with zero post-handoff contact; best close geometry
improved to about `10.94/7.11/4.80 m`. Candidate 184 changes only first-stage
ceiling `1.5 -> 2.0 s`, still fresh/current and `(0,20] m`; later stages remain
`0.5 s`. Run up to five individually inspected attempts, stopping the whole
branch immediately on any post-handoff ID `1002`/contact or official finish.
Candidate 184 completed five individually inspected attempts: four clean index-
3 runs, two full guarded two-second first stages, zero post-handoff contact, no
finish, and best `10.67/6.03/4.58 m`; close the duration ladder. Candidate 185
restores exact Candidate-178 half-second stages and changes only
`PUFFER_FINAL_GATE_HOLD_PULSE_LATERAL=1`: during an already active bounded pulse
dropout, hold the saved authorized lateral roll under the existing cap, then
level at expiry. Hash `75f36a9988109e7b1c38aa8690c79e9fdf2001281cb250fa118c3601191ae1f1`;
`88 passed`, Windows compile succeeds, hashes match. Fixed maximum-five batch,
no tuning, finish-stop; reject on post-handoff contact.
Candidate 185 yielded only one clean index-3 run; its `1.45 m` raw minimum was
stale pre-delay handoff geometry and no pulse fired. Four attempts failed in the
prefix, so saved-roll hold was initially untested. Candidate 186 supplied one
clean index-3 opportunity and exercised the mechanism exactly: a fresh
`17.28/1.89/9.18 m` `frame_partial` authorized a half-second `0.22` pulse and
`-0.0378 rad` roll; four immediately following `40-46 m` aperture aliases held
that roll, and pulse expiry leveled it with zero contact. The target still
receded/disappeared and no finish followed. Leave
`PUFFER_FINAL_GATE_HOLD_PULSE_LATERAL` disabled by default and close pulse-
authority tuning. The next change is perception-only: keep exact retained
control, but prevent an available final-phase cropped edge component from
losing single-candidate selection to a centered downstream aperture. Preserve
whole-batch finish-stop and reset proof.
Candidate 187 is preregistered as that single change. Set
`PUFFER_POLICY_FINAL_PREFER_ANY_EDGE_FRAME=1` with exact Candidate-178 control
(half-second first/later stages, maximum four, one-second cooldown, `0.22`,
close `2/4 m`, lateral `0.02/0.10`) and pulse-lateral hold disabled. The option
defaults off, applies only at/after raw index `3`, uses only existing red-pixel
edge checks, and logs selection/override counts; Gates 1-3 remain exact.
Detector hash `ebcdaae0df17bcb0c3e93ac9666c5574dffae6974b8d51a4ae98d7ba275d650a`
matches root/Windows; callable stays `75f36a...`; focused verification is
`86 passed` plus `7 passed` on Windows detector tests and compilation. Run a
fixed maximum-five `competitive_r1_gate4_any_edge_priority_batch_187` batch,
no tuning, finish-stop; reject false edge capture or any post-handoff contact.
Candidate 187 reached index `3` once and logged `10/3` edge selections/
overrides, but immediate priority captured the just-passed gate from
`1.72 -> 33.23 m`, drove yaw `1.87 -> 2.28 rad`, then lost all detections; one
estimator collision event also made the run invalid. Reject zero delay.
Candidate 188 changes only `PUFFER_POLICY_FINAL_ANY_EDGE_DELAY_S=3.0`: prior
dominant-edge behavior remains exact for the proven three-second handoff/wall-
clearance interval, then any-edge priority may activate. Current verification
is `87 passed` root and `35 passed` Windows; runtime hashes are detector
`ebcdaae0...`, callable `75f36a...`, Windows smoke `7e782b40...`. Run fixed
maximum five under `competitive_r1_gate4_any_edge_delay3_batch_188`, no tuning,
finish-stop; reject any pre-delay override, false edge capture, or contact.
Candidate 188 produced `3/5` clean index-3 runs, zero post-handoff contact, and
selector/override counts `13/5`, `9/4`, and `14/5`; retain the three-second
delay. Two runs used two safe half-second pulses, but best delayed geometry was
only `14.90/2.79/7.03 m`, worse than Candidate 176. Candidate 189 changes only
later-stage pulse duration `0.5 -> 1.0 s`; the first stays `0.5 s`, with max
four/cooldown/thrust/range/detector/association/prefix exact. Run fixed maximum
five under `competitive_r1_gate4_any_edge_delay3_later1_batch_189`, no tuning,
finish-stop; reject any post-handoff contact.
Candidate 203 exercised level pitch during two safe stages but best geometry
was only `18.38/3.19/9.11 m`; reject it and restore disabled. Candidate 204
produced four clean official-index-3 runs and one index-3 collision (ID `1001`,
impact `5.5551`) after three descent stages, with no finish or crossing sample;
best clean geometry was only `9.29/2.10/5.18 m`. Reject the unguarded expanded
window. Candidate 205 retains Candidate 204 exactly but adds the default-off
`PUFFER_FINAL_GATE_DESCENT_MAX_ABS_RIGHT_M=4.5` pulse-admission guard. Tag
`competitive_r1_gate4_pulse_right45_guard_batch_205`, maximum five, no tuning,
finish-stop; reject any post-handoff contact. Candidate 205 produced two clean
index-3 runs and three prefix failures; both clean runs used two in-corridor
stages with zero contact, no crossing/finish, and best `13.94/3.07/6.45 m`.
Retain the guard. Candidate 206 changes only cooldown `1.0 -> 0.375 s`, admitting
the measured fresh `16.62/4.08/8.00 m` state `0.407 s` after stage 2 while the
guard and maximum-three cap remain. Tag
`competitive_r1_gate4_guard_cooldown0375_batch_206`, maximum five, no tuning,
finish-stop; reject contact.
Candidate 189 was preflight-rejected on all five wrapper invocations because
the callable forbids later pulses longer than the first; no smoke/candidate
flight occurred. A separate reset snapshot proved active `0`, finish `-1`, zero
collisions, healthy status `4`, idle motors. Candidate 190 recombines accepted
Candidate 182 and 188 behavior as uniform `1.0/1.0 s` first/later stages; max
four/cooldown/thrust/range/delayed detector/association/prefix remain exact.
Run fixed maximum five under `competitive_r1_gate4_any_edge_delay3_uniform1_batch_190`,
no tuning, finish-stop; reject post-handoff contact or failed hover restoration.
Candidate 190 produced one clean index-3 run and one safe one-second pulse, but
best delayed geometry was only `18.38/-3.59/9.68 m`; restore Candidate-188
`0.5/0.5 s` stages. Candidate 191 changes only close re-anchor maximum count
`2 -> 3`; every admission retains `0.125 s`, `(0,20] m`, `|right|<=4 m`, low-
gate, current, and 8 m step guards. Run fixed maximum five under
`competitive_r1_gate4_any_edge_delay3_closecount3_batch_191`, no tuning,
finish-stop; reject false-family admission or post-handoff contact.
Candidate 191 produced two clean index-3 runs but no third close admission;
count three is untested and not retained. Its actionable trace was a coherent
`16.30/4.18/7.82 m` cropped family after one safe pulse, just outside the `4 m`
lateral admission. Candidate 192 restores count two and changes only close
right bound `4.0 -> 4.5 m`; all `0.125 s`, current, `(0,20] m`, low-gate, step,
pulse, delayed-detector, association, and prefix settings remain exact. Run
fixed maximum five under `competitive_r1_gate4_any_edge_delay3_close45_batch_192`,
no tuning, finish-stop; reject overshoot or post-handoff contact.
Candidate 192 produced two clean index-3 runs plus one index-3 ID-1002 contact.
The contact run executed all four half-second stages, reached
`9.60/2.42/4.02 m`, then fresh `13.15/3.53/0.35 m` vertically aligned geometry
before tiny impact `0.03191`; hover had resumed. Reject bound `4.5` standalone,
but use it for one remedial lateral bracket. Candidate 193 changes only lateral
gain `0.02 -> 0.03`, retaining cap `0.10` and every detector/pulse/association/
prefix setting. Run fixed maximum five under
`competitive_r1_gate4_any_edge_delay3_close45_latkp03_batch_193`, no tuning,
finish-stop; reject overshoot or any post-handoff contact.
Candidate 193 produced two clean index-3 runs and no contact, but neither
recreated Candidate 192's four-stage/vertically aligned state; best delayed
geometry was only `15.71/2.58/7.49 m`, so the gain bracket is untested at its
target geometry. Candidate 194 is one exact tag-only maximum-five retry under
`competitive_r1_gate4_any_edge_delay3_close45_latkp03_retry_batch_194`, no
tuning, finish-stop; require comparable four-stage evidence or finish and
reject any contact.
Candidate 194 then passed official index `3` cleanly in `5/5`, with zero
collision/invalid state and no finish, but the delayed cropped-edge selector
never activated (`0/0` selections/overrides in all five). Four runs emitted
one safe half-second descent stage, while closest delayed raw aperture families
remained about `19.59-21.33 m` forward and `18.35-19.73 m` laterally displaced;
the Candidate-192 near-finish geometry was not reproduced. Close the exact
retry and `kp=0.03` lateral branch. Do not run Candidates 193/194 again; retain
the new `5/5` clean Gate-3-prefix evidence and choose a new observable final-
gate acquisition/association mechanism.
Candidate 195 restores Candidate-188/178 `kp=0.02`, right bound `4.0 m`, and all
other safe settings, then adds only disabled-by-default observable dropout yaw
freeze. With `PUFFER_FINAL_GATE_FREEZE_YAW_ON_DROPOUT=1`, a missing/rejected
current final pose replaces the stale absolute-yaw setpoint with current
quaternion yaw; associated-pose servo, roll, brake/hover, pulses, ranges, and
prefix remain exact. Run fixed maximum five under
`competitive_r1_gate4_dropout_yaw_freeze_batch_195`, no tuning, finish-stop;
accept yaw arrest plus reacquisition/improved geometry and reject false-family
lock or any contact.
Launch audit: Candidates 194 and 195 accidentally omitted
`PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX=3`. This left the motion filter in the
final observation path and disabled final edge preference, explaining all-zero
selector counts. Their clean runs remain prefix/safety evidence, but neither is
a valid intended-parent final-mechanism evaluation. Candidate 196 corrects only
that launch error and re-runs Candidate 193 exactly with raw index `3`, yaw
freeze disabled, `kp=0.03`, right bound `4.5 m`, tag
`competitive_r1_gate4_any_edge_delay3_close45_latkp03_rawfix_batch_196`, maximum
five, no tuning, finish-stop. Audit raw pose/observation agreement before using
the result.
Candidate 196 confirmed the correction: its two clean index-3 attempts logged
`26/21` and `19/12` delayed edge selections/overrides with raw pose/observation
agreement and no final contact. Neither pulsed; best geometry was
`20.093/2.951/9.952 m`, just outside the `20 m` range. Three other runs failed
before handoff. Close Candidates 193/196 and the exact `kp=0.03` retry.
Candidate 197 restores raw index `3`, Candidate-188/178 `kp=0.02`/right bound
`4.0 m`, and enables only dropout yaw freeze under tag
`competitive_r1_gate4_raw_dropout_yaw_freeze_batch_197`, maximum five, no
tuning, finish-stop; require exercised raw-dropout yaw arrest plus safer target
retention/geometry and reject any post-handoff contact.
Candidate 197 produced three clean index-3 final phases; all `171/171` sampled
raw-dropout frames held current quaternion yaw, with no post-handoff contact.
One run used two safe half-second stages and reached `16.94/2.62/8.18 m`;
another repeated the quantized `20.093/2.983/9.952 m` cropped frame, only
`0.093 m` outside the strict pulse range. Retain yaw freeze. Candidate 198
changes only first/later range `20.00 -> 20.25 m` under the exact safe parent,
tag `competitive_r1_gate4_raw_yawfreeze_range2025_batch_198`, maximum five, no
tuning, finish-stop. Require safe activation near `20.093 m`; reject contact.
Candidate 198 produced two clean final phases and safely exercised `20.25 m`:
its second half-second stage began at `20.093/2.260/9.890 m`. A fresh post-stage
frame reached `19.20/4.11/9.45 m`, just outside the `4.0 m` close-right gate.
Retain `20.25 m`. Candidate 199 changes close right `4.00 -> 4.25 m` but caps
stages at three, admitting measured `4.11-4.18 m` evidence while staying below
rejected `4.5 m` and preventing the four-stage near-contact branch. Tag
`competitive_r1_gate4_raw_yawfreeze_range2025_right425_max3_batch_199`, maximum
five, no tuning, finish-stop; reject any post-handoff contact.
Candidate 199 was prefix-limited: four Gate-3 failures and one clean index-3
run whose closest cropped pose `20.57/4.37/10.16 m` stayed outside both bounds,
so no stage executed. Candidate 200 is one exact tag-only maximum-five retry
under
`competitive_r1_gate4_raw_yawfreeze_range2025_right425_max3_retry_batch_200`,
no tuning, finish-stop. If no eligible third stage occurs, close `4.25 m`.
Candidate 200 produced two clean final phases. Attempt 5 safely used all three
stages and reached collision-free raw poses `2.98/-0.49/0.84 m` then
`3.62/0.15/1.12 m`, but kept braking because the `>8 m` identity jump was
rejected and the existing close re-anchor requires `down>1.5 m`. Candidate 201
adds only a disabled-by-default, separate one-use crossing re-anchor after the
delay: `0.125 s` coherent raw cluster, forward `<=5 m`, `|right|<=1 m`,
`|down|<=1.5 m`. Normal aligned `+0.06` pitch and all other control/safety values
remain exact. Test first, then run tag
`competitive_r1_gate4_crossing_reanchor_batch_201`, maximum five, no tuning,
finish-stop; reject any off-center admission or contact.
Candidate 201 produced three clean index-3 final phases but no pose closer than
`16.0 m`; the `<=5 m` crossing branch did not activate. Candidate 202 is one
exact tag-only maximum-five retry under
`competitive_r1_gate4_crossing_reanchor_retry_batch_202`, no tuning, finish-
stop. If the near state does not recur, stop sampling and retain the mechanism
only as default-disabled/test-proven.
Candidate 202 produced four clean final phases but no pose closer than `16.30
m`; stop sampling the rare crossing state and retain its path only default-
disabled/test-proven. Candidate 203 adds only
`PUFFER_FINAL_GATE_LEVEL_PITCH_DURING_DESCENT_PULSE=1`: set pitch to `0 rad`
during an already-authorized bounded pulse instead of retaining `-0.12 rad`
brake. No pulse authorization, thrust, yaw/roll, association, range, stage cap,
prefix, or finish rule changes. Test first, then run tag
`competitive_r1_gate4_level_pitch_pulse_batch_203`, maximum five, no tuning,
finish-stop; reject any post-handoff contact.
Candidate 181 achieved three clean index-3 runs and no finish/contact, but its
exercised minima stayed about `17.63 m`; cap `0.12` did not retain the target or
improve Candidate 176, so restore `0.10`. Candidate 182 uses exact Candidate-178
settings and changes only first-stage ceiling `0.5 -> 1.0 s`; later stages stay
`0.5 s`, max four, cooldown `1 s`, thrust `0.22`, fresh `(0,20] m` authorization,
association/prefix, fixed maximum-five batch, and finish-stop remain exact.
Reject on any post-handoff contact.

Reproduce that exact controller before modifying it. Then extend its solved
prefix toward Gate 3. Use the deterministic `course-fsm`/visual-servo stack for
diagnostics and bounded fallbacks:

1. robust gate/guidance perception from the official camera;
2. camera geometry plus IMU/gyro state estimation;
3. race-index-aware acquisition, align, approach, commit, pass, brake, and
   reacquire phases;
4. gyro-closed body-rate/thrust control;
5. direct official-simulator segment timing and parameter optimization;
6. a learned detector or compact learned residual only if a frozen failure set
   proves the classical component inadequate.

Do not restart broad PPO, BC/DAgger, action sweeps, or another renamed policy
optimizer merely because that infrastructure exists. Preserve the recovered
Gate-2 checkpoint as an immutable official baseline. Any learned or classical
child must retain its two-gate prefix before it can be promoted.

## Execution and Evidence Discipline

- The copyable goal is `docs/competitive_lap_execution_prompt.md`.
- `PRD.md` is the product-level source of truth and must be updated whenever a
  major decision, blocker, milestone, or retained best lap changes.
- Append detailed experiments immediately to the execution prompt's run ledger
  or a dated linked ledger. Do not leave the PRD pointing at stale state.
- Every attempt gets a unique tag and paired JSON/CSV evidence under
  `logs/sitl/`. Never overwrite or silently repeat an unchanged configuration.
- Record simulator version/event, controller/config hash, exact command,
  official gate index progression, gate split times, finish time, collision and
  invalid status, heartbeat/command/camera health, accept/reject decision, and
  exactly one next action.
- Maintain a rejected-configuration index so agents do not circle back to the
  same gains, route, checkpoint, or failure signature.
- Before a run, confirm the event identity, fresh race state, camera/telemetry,
  arming, command publisher, and logging. After a run, verify evidence before
  changing anything.
- Preserve unrelated dirty-worktree changes and the healthy simulator process.
- Official rules require disclosure of FLOSS and generative-AI tools used in
  the entry. Keep dependencies and AI-assisted changes auditable; never expose
  credentials or publish competition recordings/content.

## Completion Contract

1. Run the locally available `AI-GP Virtual Qualifier R1` event/course.
2. Establish the exact finish-time lock/submission behavior before crossing the
   final gate.
3. Demonstrate reliable ordered, collision-free rehearsals without human
   intervention.
4. Produce a valid official full-course time and retain its complete evidence.
5. Continue optimizing until the time is competitive against an observed
   leaderboard/cutoff or until a pre-registered bounded search is exhausted.

Do not describe a native run, vision-only pass, or incomplete R1 lap as a
competitive valid result.

## N146 Current Initialization Context — 2026-07-17

- The active objective remains a competitive autonomous six-gate VQ1/R1 lap
  with an official finish and subsequent lap-time optimization. Passing an
  early gate is evidence, not completion. Only VQ1/R1 is available locally.
- Simulator: `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`.
  Reuse the healthy process. The normal in-place simulator reset is MAVLink
  `COMMAND_LONG` command `31000`. A transmitted reset is never assumed to have
  worked: require a new boot/race epoch plus policy-ready telemetry. v3385 can
  ignore command `31000` on an inactive or post-collision `THROTTLE DOWN`
  screen; use the documented no-relaunch UI recovery in that state.
- Candidate 007 is rejected and must not be retried unchanged. Its trace has
  `457` recurrent inference ticks but only `180` samples at `10 Hz`, so exact
  base-policy recurrent replay is impossible. The exact observable N145
  projection nonetheless proves its severe-direct proposal was eligible on
  `0/180` samples and changed `0/180` actions. Candidate 007 was therefore a
  failure of the unchanged N112 recurrent prefix, not evidence for or against
  the N145 severe-direct correction. Report
  `logs/sitl/n145_candidate007_trace_replay_20260717.json` SHA-256 is
  `33b7c5621b5610739a094271299ca2cf20160f30d5f17ea38ad2d6b5b2f4ec3b`.
- N146 is the simplest evidence-backed pivot. Gates 1--3 use the historical
  23-input `v6c_latest_obsmatch_r135_dropout_e10.bin` prefix, SHA-256
  `ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760`.
  Ten official Candidate-129 resets passed Gates 1 and 2 in `10/10` and Gate 3
  in `8/10`. The current runner carries the exact observable gate-pose
  confidence in reserved observation slot 30 only for Gates 1--3, reconstructs
  the legacy 23-input contract exactly, and clears slots 30/31 before the
  existing 32-input Gate-4/5/6 tail. Across `767` historical confidence
  samples the exact float32 bridge error is `0`; the rejected apparent-size
  approximation reaches `0.3993567`. Evidence report
  `logs/sitl/n146_hybrid_prefix_exact_bridge_20260717.json` SHA-256 is
  `bab4185386812a48c85f27f4a521abe35de20fef3ca0dec56187fe12d841b039`.
- Hybrid callable SHA-256 is
  `3d41de0abb2b4438055cbd69e5816a99ae978ff04266342d6d71d987ef13a00d`;
  the SHA-pinned Windows wrapper is
  `scripts/run_windows_six_gate_hybrid.ps1`, SHA-256
  `3398e62437d18933630c912feac1f2da51c4fa7d7dfd1317da3ab607cd996d66`.
  Actual Windows Python passes `75/75` focused tests, both PowerShell launchers
  parse, and direct Windows loading/inference across prefix and all tail models
  returns finite four-action outputs.
- No N146 simulator flight is authorized yet. The next interaction is one
  separately logged, telemetry-detected reset-only MAVLink command `31000` to
  restore visible stationary Gate 1, followed by zero-command tag
  `n146_six_gate_hybrid_shadow_010`. Shadow must send zero reset, arm,
  setpoint, and disarm itself and pass complete live trace replay, exact hashes,
  visible Gate 1, and `[50,100) Hz` no-op cadence before Candidate 008 can be
  preregistered.

## N146 Reset Setup and Passive Shadow 010 — Passed

- Before shadow, passive status proved the responsive preserved simulator at
  PID `24476`, active R1/index `0`, finish `-1`, pre-reset boot `83605 ms`, race
  start `3283 ms`. One separately logged MAVLink `COMMAND_LONG` command `31000`
  setup reset changed the boot epoch to `3248 ms` and race start to `3274 ms`;
  Gate 1 was visible. Setup snapshot SHA-256 is
  `b233997473bc3eeb310869c0467fc652c720b28cfe5cf95bf61b08568f651f4d`.
- Zero-command `n146_six_gate_hybrid_shadow_010` passed. It sent zero reset,
  arm, MAVLink setpoint, and disarm. All `59/59` Gate-1 inference samples were
  visible and replayed with maximum action error
  `1.8423500058872833e-7`; all 11 manifest entries match. Its no-op publisher
  counted `550` commands over `10.031 s`, effective `54.730 Hz`, with zero
  violations and zero probe setpoints. Official state remained index `0`,
  finish `-1`, base mode `193`, system status `4`; collision/dropout/drain hits
  were all zero. Shadow/parity SHA-256 values are
  `f0ee13930e8b09908201daaf48bcaad5984c8d9fc3766977137d97093b59e7ab`
  and `5418c12a00c487eccee8c426c980e098a0229846bdb612b7b06ac9e3b304b09d`.
- Preregister exactly one bounded flight under tag
  `n146_six_gate_hybrid_gate4_bounded_008`. Reuse PID `24476`, issue one normal
  policy-ready MAVLink command `31000` reset and require detection, request
  `80 Hz`, run at most `30 s`, repeats/min-valid `1/1`, target/min/authoritative
  stop at official index `4`. Accept only index-4 stop with finish `-1`, actual
  wire cadence `[50,100) Hz`, zero fast violations, all frozen hashes, no
  collision/invalid/dropout/drain issue, one exit disarm, and passive disarm
  proof. Any failure rejects Candidate 008 with no unchanged retry. `FullLap`
  remains unauthorized.

## N146 Candidate 008 Rejected; N147 Complete Legacy ABI

- Candidate 008 ran exactly once and is rejected without an unchanged retry.
  Its command-`31000` reset was detected (`197730 -> 3435 ms`, race start
  `3367 ms`, `60/60` calibration samples). It made zero official passes and
  remained Gate 1/index `0`, finish `-1`. Closest accepted Gate-1 pose at
  `6.657 s` was `[3.168317, 1.851485, 0.202970] m`; collision ID `1002`, impact
  `0.25549981`, threat `1`, terminated the run at `19.016 s`. Transport passed
  at `1031/18.234 = 56.488 Hz`, zero fast violations, zero dropout/drain hit.
  Exit disarm was sent once; passive status is `base_mode=65`, system status
  `3`, index `0`, finish `-1`, SHA-256
  `ba58413c6eabfcc02b1c280666e38ba5488856a75812fbf3714838e2418d975f`.
- Root cause is an incomplete legacy observation ABI, not a valid v6c
  reproduction. N146 restored field 17 confidence only. Candidate 129 supplied
  field 18 as `elapsed/10.5`; Candidate 008 supplied `elapsed/30.0`. Legacy
  field 22 is `official_gate_index/3`, while Candidate 008 supplied current
  last-yaw command. N147 carries exact legacy confidence in slot 30, exact
  `elapsed/10.5` in slot 31, and reconstructs field 22 as `gate/3`; tail inputs
  still clear slots 30/31. All 23 float32 prefix values now round-trip exactly.
- N147 evidence
  `logs/sitl/n147_hybrid_prefix_complete_bridge_20260717.json` passes with
  SHA-256
  `c561b89e2209a744efc2a2dd73ad32fd5103ec532e2b21834477dd586eae4b56`.
  Callable/runner/verifier/wrapper SHA-256 values are `55ffc257...`,
  `87d62ec8...`, `446e2200...`, and
  `446c18048f6e18e3c1523b3282e460ba6989867485816cc5c0743b35af195022`.
  Actual Windows Python passes `78/78` focused tests, both PowerShell launchers
  parse, and direct prefix/tail loading/inference passes.
- No N147 flight is authorized. Next send one logged command-`31000` setup
  reset and require epoch/camera recovery. If the post-collision screen ignores
  it, do not repeat it; recover R1 via Pause -> Back To Main -> sole VQ1/R1
  without relaunching PID `24476`. Then run zero-command
  `n147_six_gate_hybrid_shadow_011`. Only a complete passing shadow can
  authorize a separately preregistered bounded Candidate 009.

## N147 Reset Setup and Passive Shadow 011 — Passed

- One setup MAVLink command `31000` succeeded even after Candidate 008:
  pre-reset boot `67590 ms` changed to `3330 ms`, race start to `3324 ms`, and
  Gate 1 was visible. No UI recovery or process relaunch was needed. Snapshot
  SHA-256 is
  `e14720147e5de1a4593091bd4f5fad3287bfd79b1073d741bf20f2b9c9cfe99d`.
- Passive `n147_six_gate_hybrid_shadow_011` passed with exact reset/arm/MAVLink
  setpoint/disarm zeros. All `54/54` complete samples were visible Gate 1 and
  replayed within `1.8943729400422438e-7`; all 11 manifest artifacts match.
  No-op cadence was `571/10.031 = 56.824 Hz`, zero violations and probe
  setpoints. Official state stayed index `0`, finish `-1`; collision/dropout/
  drain hits were zero. Shadow/parity SHA-256 values are
  `bfe416b4841bf850167f0e0eae0222741b801884ef32c6ce0f6687e936b3878e`
  and `ac28cd83014c4753b8720dcae9d3c2ef7f20a98229fb13c60746b1df73b9148d`.
- Preregister exactly one corrected bounded attempt under tag
  `n147_six_gate_hybrid_gate4_bounded_009`: reuse PID `24476`, one detected
  policy-ready command-`31000` reset, `80 Hz`, maximum `30 s`, repeats/min-valid
  `1/1`, target/min/stop official index `4`. Accept only clean index-4 stop,
  finish `-1`, wire cadence `[50,100) Hz`, zero fast violation, matching hashes,
  no collision/invalid/dropout/drain issue, one exit disarm, and passive disarm
  proof. Failure rejects Candidate 009 without unchanged retry; FullLap remains
  unauthorized.

## N147 Candidate 009 Rejected; N148 Recurrent-Yaw Correction

- Supersede the preceding N147 field-22 interpretation. The archived report
  names outer observation field 22 `official_active_gate_index_norm`, but the
  historical callable copied the observation and then executed
  `recurrent_observation[22] = _LAST_YAW_ACTION` before every checkpoint
  inference. It updated that state from `action[3]`. The recurrent checkpoint
  therefore expects the previous normalized yaw action, not `gate/3`.
- Candidate 009 ran once under tag
  `n147_six_gate_hybrid_gate4_bounded_009` and is rejected with no unchanged
  retry. Its MAVLink `COMMAND_LONG` command `31000` reset was detected
  (`134628 -> 3375 ms`, race start `3278 ms`, detection `0.515 s`, calibration
  `60/60`). It made zero official passes, stayed at index `0`, finish `-1`, and
  collided with ID `1002`, impact `10.26780033`, threat `2`, at the `11.265 s`
  stop. Transport remained healthy at `619/11.187 = 55.243 Hz`, zero fast
  violations/dropout/drain hit. Passive follow-up proves base mode `65`, system
  status `3`, index `0`, finish `-1`, SHA-256
  `3447b06e56bcd6f6cc09c1dc6c6ef92f2dc93ac544f414506853fd69f07b6d85`.
- N148 removes the N147 `legacy[22] = gate/3` overwrite. It restores only the
  actual ABI differences: exact legacy confidence from slot 30 and exact
  `elapsed/10.5` from slot 31. Current field 22 remains the runner-maintained
  previous-yaw action; the six-way one-hot remains outside-model phase
  selection. The source-contract/evidence report passes with zero blockers,
  `10/10` Gate-1/2 and `8/10` Gate-3 history, `767` archived samples, exact
  float32 bridge error `0`, and historical callable yaw-contract proof. Report
  SHA-256 is
  `9c9be7edceac7951f982d58bb93eb5959da7b3bdc1c41b5650e3261dfdb99d12`.
- N148 callable/runner/shadow-verifier/bridge-verifier/wrapper SHA-256 values
  are `a8e3ae0f...`, `87d62ec8...`, `446e2200...`, `52818ae2...`, and
  `39a356e336edf265a2efe8d971b2fe8683aab7ba396114d9dcb2635a06c317a5`.
  Actual Windows Python passes `75/75` focused tests and the PowerShell wrapper
  parses. No N148 flight is authorized yet. Use one separately logged command
  `31000` setup reset with detected epoch/camera recovery, then only passive
  zero-command `n148_six_gate_hybrid_shadow_012`. FullLap remains unauthorized.

## N148 Reset Setup and Passive Shadow 012 — Passed

- One separately logged MAVLink command `31000` setup reset succeeded after
  Candidate 009. Pre-reset boot `244531 ms` changed to `3307 ms`, race start
  became `3298 ms`, Gate 1 was visible, and the preserved simulator process was
  not relaunched. Snapshot SHA-256 is
  `f517a08edd61e2c8d7a989569ccf1670bbc66af1d6eb2635908990c89a41a04c`.
- Passive `n148_six_gate_hybrid_shadow_012` passed through actual Windows
  Python. It sent zero reset, arm, MAVLink setpoint, and disarm. All `44/44`
  trace samples were visible and replayed with maximum action error
  `2.5586318969095245e-7`; all 11 deployment artifacts match. No-op cadence
  was `561/10.047 = 55.738 Hz`, zero fast violations, zero probe setpoints.
  Official state remained index `0`, finish `-1`, base `193`, system `4`, with
  zero collision/dropout/drain hit. Shadow/parity SHA-256 values are
  `9ed223ffbb23a21e2c0c86582588501f06b13c275e37717e7f3ea183eda08c53`
  and `8536d25564d660d6cd6f7f319c2c94f6c190b84f6fd76ceae9e04c7ba696e409`.
- Preregister exactly one bounded flight under tag
  `n148_six_gate_hybrid_gate4_bounded_010`: reuse the current FlightSim process,
  one normal policy-ready command-`31000` reset with required detection,
  requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop index `4`. Accept only clean index-4 stop with finish
  `-1`, cadence `[50,100) Hz`, matching hashes, zero fast violation/collision/
  invalid/dropout/drain issue, one exit disarm, and passive disarm proof. Any
  failure rejects Candidate 010 without unchanged retry. FullLap remains
  unauthorized.

## N148 Candidate 010 Rejected; N149 Scheduler-Starvation Fix

- Candidate 010 ran once and is rejected without unchanged retry. Its command
  `31000` reset was detected (`171210 -> 3415 ms`, race start `3301 ms`,
  detection `0.500 s`, calibration `60/60`). It made zero official passes,
  stayed index `0`, finish `-1`, and collided with ID `1001`, impact
  `11.93652916`, threat `2`, at `5.562 s`. Transport passed at
  `300/5.266 = 56.779 Hz`, zero fast violation/dropout/drain hit. Passive
  follow-up is base `65`, system `3`, index `0`, finish `-1`, SHA-256
  `7925f0a9cb2726aa7897f4c7684fd71c68724b873fb474e2b8e4b1b7efab98b9`.
- N148's fresh first recurrent action matches historical Candidate-129 attempt
  010 within maximum `0.0003507`, so the corrected ABI, checkpoint, initial
  hidden state, and decoder agree. Divergence follows scheduler starvation:
  successful Candidate-129 reports process gate-motion updates at
  `10.857--12.926 Hz` (median `11.524`), while Candidate 010 achieved only
  `4.675 Hz` and `5.753` policy inferences/s.
- Root cause is the Windows publisher's full-period busy spin, which retained
  legal wire cadence while starving camera/inference work. N149 changes only
  waiting: release the GIL for most of each period and retain a final `1 ms`
  hot deadline. Evidence
  `logs/sitl/n149_legacy_prefix_cadence_correction_20260717.json` passes with
  zero blockers, SHA-256
  `5623d409ce320835d4ebb64d9fc273ef6394003c01b9daf5f71ed98556b833d0`.
- N149 runner/shadow-verifier/wrapper SHA-256 values are `0c8352df...`,
  `7538094e...`, and
  `74047bcb799ec6cd03fa9c04beed815beb655e5d5c8642cb410c7d725ae0f321`.
  Shadow verification now fails closed below `10.0` inference Hz as well as on
  prior replay/cadence/health checks. No flight is authorized. Next is one
  reset-only command `31000` setup plus passive tag
  `n149_six_gate_hybrid_shadow_013`; FullLap remains unauthorized.

## N149 Reset Setup and Passive Shadow 013 — Passed

- One command-`31000` setup reset changed pre-reset boot `47901 ms` to
  `3159 ms`, race start `3299 ms`, with visible non-black Gate 1 and no process
  relaunch. Snapshot SHA-256 is
  `d01444946a600519155a96ece4c442d5ef48312a5436c469444104ca8dc3f403`.
- Passive `n149_six_gate_hybrid_shadow_013` sent zero reset/arm/setpoint/disarm
  and passed. Scheduler correction raised inference to `306/10.0 = 30.6 Hz`;
  all `306/306` visible samples replayed within
  `1.9684982299761344e-7`, with all 11 hashes matching. No-op wire probe was
  `600/9.985 = 59.990 Hz`, zero violations/setpoints. Camera processed `122`
  frames and `120` detections. Official index `0`, finish `-1`, base `193`,
  system `4`; collision/dropout/drain all zero. Shadow/parity SHA-256 values:
  `b9e93071556688fdbf92057d1706ba7b8c365c5f4e58edc99a49daafeae2d855`,
  `820b2b2cdae5a271f03aa6b213e8689cd8ca53c0b58ededcee9f72aa13f5cc9d`.
- Preregister exactly one bounded Candidate 011 under tag
  `n149_six_gate_hybrid_gate4_bounded_011`: current pinned wrapper, actual
  Windows Python, reused FlightSim, one detected command-`31000` reset,
  requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop index `4`. Accept only clean index `4`, finish `-1`,
  cadence `[50,100) Hz`, matching hashes, zero fast violation/collision/
  invalid/dropout/drain issue, one exit disarm, and passive disarm proof.
  Failure rejects it without unchanged retry. FullLap remains unauthorized.

## N149 Candidate 011 Rejected at Gate 3; N150 Close-Severe Boost

- Candidate 011 ran once and is rejected without unchanged retry. Reset
  command `31000` was detected (`186089 -> 3358 ms`, race start `3300 ms`,
  `0.500 s`, calibration `60/60`). It passed official Gates 1 and 2, reached
  index `2`, finish `-1`, then collided approaching Gate 3 (ID `1001`, impact
  `9.45933437`, threat `2`) at `10.687 s`. Transport was healthy at
  `638/10.422 = 61.121 Hz`, zero violations/dropout/drain hit; policy inference
  was `452/10.687 = 42.294 Hz`. Passive snapshot proves base `65`, system `3`,
  index `2`, finish `-1`, SHA-256
  `e64bd4d4354f233c8657f0ba9afac90af5b03d1ef0f9c5db46f6ad19d0e454c3`.
- Gates 1-2 and scheduler are frozen. Candidate 011's last Gate-3 observation
  at `2.693 m` predicted a severe `-1.072 m` lateral miss; the promoted taper
  commanded roll `0.9101` and the run remained on the same lateral failure
  branch as historical failed attempt 001. N150 changes roll only: when the
  already-promoted `adaptive` mode remains active inside `4 m`, command full
  normalized roll `1.0`. Pitch, thrust, yaw, all other distances/modes, and all
  other gates remain unchanged.
- Offline evidence is selective: Candidate 011 has one eligible sample;
  historical lateral failure 001 has two; only one sample across all eight
  successful Gate-3 reports is affected. Non-roll maximum error is `0`.
  `logs/sitl/n150_gate3_close_severe_boost_20260717.json` passes with SHA-256
  `3328ee70d9f20d763c5e5a1dd9bad4f59f5ffe89c5585f63ecaa3b20cb74d21d`.
  Callable/runner/verifier/wrapper hashes are `d512c26b...`, `0c8352df...`,
  `7538094e...`, and
  `4561e4c89ae757200f1bdc190532ef0532c64921917e4f9ef434940cde326ff6`;
  actual Windows Python passes `81/81` focused tests and wrapper parse passes.
- No N150 flight is authorized. Next is one setup command-`31000` reset with
  detected recovery plus passive `n150_six_gate_hybrid_shadow_014`. FullLap
  remains unauthorized.

## N150 Reset Setup and Passive Shadow 014 — Passed

- One separately logged MAVLink `COMMAND_LONG` command `31000` reset restored
  a policy-ready VQ1/R1 epoch: simulator boot time changed from the prior
  `115371 ms` epoch to `3053 ms`, race start was `3283 ms`, Gate 1 was visible
  in a non-black stream, base mode was `193`, system status was `4`, and the
  existing FlightSim process was reused. Setup snapshot SHA-256 is
  `b2b9a5d02d1a2d4ad39739ad515c7bdd6a367ca10a93149ee15a5f7337f3432b`.
- Passive `n150_six_gate_hybrid_shadow_014` passed through actual Windows
  Python. It sent zero reset, arm, MAVLink attitude setpoint, and disarm.
  Every `315/315` visible trace sample replayed within
  `2.675651550321234e-7`; policy inference was `31.5 Hz`, all 11 deployment
  hashes matched, and the no-op wire probe counted `591/9.938 = 59.368 Hz`
  with zero fast violations and zero probe setpoints. Official state remained
  index `0`, finish `-1`, with zero collision/dropout/drain hit. Shadow/parity
  SHA-256 values are
  `718d9080fa13b9437d386e759412c21f72c0c100a0d718fcf7205461a5b6862a`
  and `2b386ba7129be7f4ba29a5f1b61dc53c95827ced8ba9b4f2757c4e5ada473e92`.
- Preregister exactly one bounded Candidate 012 under tag
  `n150_six_gate_hybrid_gate4_bounded_012`: use the current pinned wrapper and
  actual Windows Python, reuse the running FlightSim process, issue one normal
  policy-ready command-`31000` reset and require a detected boot/race epoch,
  request `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, and set target,
  minimum, and authoritative stop to official index `4`. Accept only a clean
  index-4 stop with finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  fast violation/collision/invalid/dropout/drain issue, one exit disarm, and
  passive disarm proof. Failure rejects Candidate 012 without unchanged retry.
  FullLap remains unauthorized.


## Candidate 012 Reset Rejection; N151 Reset-Ready Reuse Fix

- Candidate 012 did not fly and is rejected without unchanged retry. The
  launcher observed a live reset-ready race at index `0`, finish `-1`, boot
  `398002 ms`, base mode `193`, system status `4`, but classified it as stale
  solely because race age was `364.9 s > 300 s`. UI recovery then accepted the
  still-observable old race start (`3283 ms`) as fresh. The policy-ready
  MAVLink `COMMAND_LONG` command `31000` was sent but no boot rollback was
  detected, so the fail-closed guard aborted before policy arm or attitude
  setpoint publication. Summary/per-attempt/reuse/stream SHA-256 values are
  `747b25daf555328f004bc711053d97b9bf8bd9fe65614137f57b3cdaf9a671e2`,
  `c2395081d0d75d3ae48a2ebe1b70e7168f59b009041192018cb9016297e15934`,
  `5899de166bf91338afc7b294289f10b86fe5415e8d865bc9a6395386581ac3b9`,
  and `b57cf930239ffe407e27b7512f85ebadfd683334e5b58dd3e9aa8f22a67053dc`.
- Passive post-failure state has visible non-black Gate 1, no collision, index
  `0`, finish `-1`, and a continuing old boot epoch (`490448 ms`), proving no
  reset occurred. The reset preflight's disarm left base `65`, system `3`;
  snapshot SHA-256 is
  `acf0e8f3eaf1ba7601b33ac93e679533e2c4d5c97743fdafd403fde461438fd5`.
  This is lifecycle evidence only and says nothing about the N150 policy fix.
- N151 changes only healthy-process reuse. A race with live
  `race_started=true`, nonnegative age, base `193`, and status `4` is
  MAVLink-31000-reset-ready regardless of age and must be left untouched for
  the normal reset command. UI recovery is reserved for a non-reset-ready
  state. Startup SHA-256 is
  `198b356d613d238077577b2595b25466c59bc4832a03369c95820b032e8775af`;
  the N151 wrapper pins it and has SHA-256
  `2d6f7a85e1845e7a8228a61b534678fea8fd88d4ae3f0ba9b5e2583c9020149b`.
  Focused Windows tests pass `67/67`, and both PowerShell files parse.
- No N151 flight is authorized. Recover the UI only because the current state
  is demonstrably base `65`/status `3`; then send one separately logged reset-
  only command `31000` and require boot/race epoch detection plus visible Gate
  1. If that succeeds, run only passive zero-command
  `n151_six_gate_hybrid_shadow_015`. FullLap remains unauthorized.

## N152 Window-Relative Recovery, Reset Setup, and Shadow 015 — Passed

- The N151 age-only correction was necessary but incomplete. The reused
  simulator window's physical rect was `(949,465)-(2891,1601)`, while
  `Invoke-Click` treated viewport-relative coordinates as absolute screen
  coordinates. The intended Back-to-Main click therefore opened Graphics.
  Recovery also returned success on old `race_started=true` telemetry despite
  base `65`/status `3`. Screenshots prove the Graphics misclick, corrected
  pause menu, actual main menu, VQ1 event prompt, and restored six-gate R1.
- N152 converts every UI click to `window_rect.left/top + relative x/y` under
  per-monitor DPI awareness. UI recovery now succeeds only at live
  `race_started=true`, base `193`, status `4`; old race packets cannot pass it.
  The earlier rule remains: an already reset-ready base `193`/status `4` race
  is reused regardless of age so no UI click occurs. Startup SHA-256 is
  `1f75d88d8ff6839e921af01872b699fe8beedfad7eca0332fbc7df013d79474d`;
  the pinned wrapper SHA-256 is
  `b6c6650d0c127d05e1975d581153dc4f49c5085955556dd30a20d1beefed9158`.
  Focused Windows tests pass `68/68`, and both PowerShell files parse.
- Measured recovery changed the stale race start from `3283 ms` to
  `1337380 ms` without relaunching PID `24476`. One subsequent reset-only
  MAVLink command `31000` then changed boot `1354001 -> 2989 ms`, scheduled
  race start `3288 ms`, and restored visible Gate 1, base `193`, status `4`,
  index `0`, finish `-1`. Setup snapshot SHA-256 is
  `757acbf536304d768cbe60c9757785546c45b5022fc25d682f8243140f949a27`.
- Passive `n152_six_gate_hybrid_shadow_015` sent zero reset, arm, setpoint, and
  disarm and passed. All `279/279` visible samples replayed within
  `2.609935760289339e-7`; inference was `27.855 Hz`, all 11 deployment hashes
  matched, no-op wire cadence was `605/10.0 = 60.4 Hz`, and all collision/
  dropout/drain/rate counts were zero. Shadow/parity SHA-256 values are
  `135d4911b2741fc3810ea5d50ea626e1d076f02b5472bc4f4ca9572e690c69d7`
  and `971752ebce398f09fa0b7096c13b61e2ebe68059e1159fd8dafcb0a7e8c3136f`.
- Preregister exactly one bounded Candidate 013 under tag
  `n152_six_gate_hybrid_gate4_bounded_013`: reuse PID `24476`, current pinned
  wrapper and Windows Python, one detected policy-ready command-`31000` reset,
  requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop official index `4`. Accept only clean index `4`, finish
  `-1`, cadence `[50,100) Hz`, all hashes, zero rate/collision/invalid/dropout/
  drain issue, one exit disarm, and passive disarm proof. Failure rejects it
  without unchanged retry. FullLap remains unauthorized.

## Candidate 013 Gate-3 Miss; N153 Observation-Epoch Fix

- Candidate 013 ran once and is rejected without unchanged retry. The corrected
  lifecycle reused PID `24476` and skipped UI clicks, then command `31000` was
  detected (`298904 -> 3402 ms`, race start `3296 ms`, detection `0.485 s`,
  calibration `60/60`). It passed official Gates 1 and 2, stayed index `2`,
  finish `-1`, and timed out at `30 s` without collision. Runtime was healthy:
  `1832/29.719 = 61.61 Hz`, `1652/30 = 55.067` inference Hz, zero fast
  violation/dropout/drain hit, `7691` messages, `3885` IMU samples, `374`
  frames, `113` detections, and one exit disarm. Aggregate/per-attempt/smoke/
  CSV SHA-256 values are
  `d068ed7cb74cc53ecca0535084bf4409617d444a6f4c1f02cf2e500ff7f01bc3`,
  `a765790388adaf8f0eca1581ea1103fff6593645f6b70db93a4c246dff72ada2`,
  `7c142e1bfe1c1519b37fecbffe708e9636ff9e3f73004e0206f65bf67c3aad63`,
  and `c0b0ad8b08285d1e5335591b7c7ca88e12b557b9872f81320ff9a81ef7d7a5ae`.
  Passive base `65`, status `3`, index `2`, finish `-1`, no collision; snapshot
  SHA-256 is
  `a046df830216fae4adacf96e0b9e68d393fb965980c9297319958ce523734ea4`.
- Root cause is a camera/command ordering race at an official gate transition.
  At Gate-3 activation, the visible new gate body vector was
  `[24.615, -3.192, 5.538] m`, but a cached just-crossed Gate-2 pose initialized
  the new identity and the policy observed `[3.730, -2.324, -0.734] m`, a
  `21.824 m` mismatch. During the remaining visible Gate-3 interval the motion
  filter accepted zero new samples and rejected 21, held roll `0.9`, then lost
  the gate. Thus Candidate 013 did not exercise N150's intended valid close
  Gate-3 branch.
- N153 clears gate/control detection caches and their timestamps and resets
  observable gate motion immediately when official gate identity changes,
  then waits for a new frame. It changes no policy, action, decoder, or
  scheduler value. Evidence
  `logs/sitl/n153_gate_transition_cache_invalidation_20260717.json` passes with
  zero blockers, SHA-256
  `b99b451cb22ca7a3f9ccff5ce82f0a2b424e434a6930870b68ce3d9a252c0d15`.
  Runner/wrapper/verifier SHA-256 values are
  `26933542fa3d6dec17fa8859adef9c88e145b7e4072e2fee04a9e2a1b7be09d3`,
  `20d53f406484dfc7124e8b7f0c050a2e00a444c3679259bcf5cd5bc5b2213e70`,
  and `7bbd8b16632c07148572cfea993b353d935ed552a080cabf9a844611b446b6e6`.
  Focused Windows tests pass `82/82`; the wrapper parses. One short timing test
  initially sampled `49.33 Hz` but passed three isolated reruns; live cadence
  remained `60.4--61.61 Hz`.
- No N153 flight is authorized. Next send one separately logged reset-only
  command `31000` with detected boot/race epoch and visible Gate 1, then run
  only passive zero-command `n153_six_gate_hybrid_shadow_016`. FullLap remains
  unauthorized.

## N153 Reset Setup and Passive Shadow 016 — Passed

- One reset-only MAVLink command `31000` rolled the continuing Candidate-013
  epoch back to boot `3128 ms`, scheduled race start `3268 ms`, and restored
  visible non-black Gate 1, base `193`, status `4`, index `0`, finish `-1`,
  without relaunch. Setup SHA-256 is
  `be222e85a1a43200a8c899ba2cba8a0d71780fb26cbc8cf6a3a688f745191169`.
- Passive `n153_six_gate_hybrid_shadow_016` sent zero reset/arm/setpoint/disarm
  and passed. All `309/309` visible samples replayed within
  `2.2564048768325407e-7`; inference was `30.567 Hz`, all 11 hashes matched,
  and no-op cadence was `617/10.109 = 60.936 Hz` with zero violations or probe
  setpoints. Official index stayed `0`, finish `-1`; collision/dropout/drain
  counts were zero. Shadow/parity SHA-256 values are
  `6e7fdb132ae2180653e79baab1671120f3019eecc937053992777de124086e44`
  and `17902c975f59c3f5640d0a49884161c3dfb801a637d5639f9e287fd15425d64c`.
- Preregister exactly one bounded Candidate 014 under tag
  `n153_six_gate_hybrid_gate4_bounded_014`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop index `4`. Accept only clean index `4`, finish `-1`,
  cadence `[50,100) Hz`, matching hashes, zero rate/collision/invalid/dropout/
  drain issue, one exit disarm and passive disarm proof. Failure rejects it
  without unchanged retry. FullLap remains unauthorized.

## N153 Candidate 014 Rejected at Gate 2; N154 Gate-3+ Scope

- Candidate 014 ran once and is rejected without unchanged retry. It reused
  responsive PID `24476`; the normal MAVLink `COMMAND_LONG` command `31000`
  reset was detected (`198589 -> 3331 ms`, race start `3303 ms`, detection
  `0.484 s`, calibration `60/60`). It passed Gate 1, then collided while
  official index was `1`: collision ID `1001`, threat `2`, impact velocity
  `9.970706939697266 m/s`, finish `-1`. The flight lasted `7.313 s`; the
  publisher sent `377` commands over `7.0 s` at `53.714 Hz`, with zero rate,
  dropout, or drain violations. Aggregate/per-attempt/smoke/CSV SHA-256:
  `836d116272e91bd5636c200ea0cc4f01658f56cbc2b70d1a8d378bd7e4fee4de`,
  `32c35a08a8f868af63c2f19e01ff5a0588eaa841dc320b8e06e4c3f926a0dcf4`,
  `bf6d724b2a4dc4b08ba4e5d41ea121cab0c612ba617efe3e6bf6550bb04fc0ed`,
  and `9fa39f626770407dfe08312539a77011d9ff817dfd9d897a3013bc34d7194197`.
  Passive post-stop state was base `65`, status `3`, index `1`, finish `-1`;
  snapshot SHA-256 is
  `cb5035cbfb1803f34e93b75af5f33a119e4a032cfc2e53327f70a715c40c0ffa`.
- Candidate 014 shows N153 was over-broad. N154 observes every official
  identity but clears cached gate/control poses, timestamps, and motion only
  when the new index is at least `2` (Gate 3 or later). This preserves the
  proven Gate-1-to-Gate-2 handoff while fixing Candidate 013's measured
  `21.824339 m` stale-pose alias at Gate 3. Policy, action, decoder, and
  scheduler values are unchanged.
- Evidence `logs/sitl/n154_gate3plus_transition_cache_invalidation_20260717.json`
  passes with zero blockers. Evidence/runner/verifier/wrapper SHA-256:
  `437c54fba6812468207f8a5713f71d6b72ec18f489df7c59c9b6756b24894b35`,
  `bb10b470e017c05c57fe2c5b9d5763708a9cd9b66133cdee8170d78496817f65`,
  `66a02b41c8011206fee4b2c337da21cbee3214ae013f16a3c622b754c96159f9`,
  `1965de670b84578c2184aa4e11f1de3bd7277cc93193b671d8ca8441b9a4cf6b`.
  Direct tests pass `54/54`; PowerShell parses. The wider set has 72 stable
  passes and only the known short scheduler probe fluctuating under load; live
  shadow cadence remains promotion authority.
- No N154 flight is authorized. Next use one separately logged reset-only
  command `31000`, require detected rollback and visible Gate 1, then run only
  passive `n154_six_gate_hybrid_shadow_017`. Reuse the healthy process; do not
  relaunch or use FullLap.

## N154 Reset and Passive Shadow 017 — Passed

- One reset-only MAVLink command `31000` rolled the post-Candidate-014 boot
  counter from `58715` to `3274 ms`; new race start `3288 ms`, base `193`,
  status `4`, index `0`, finish `-1`, visible non-black Gate 1. Setup SHA-256:
  `defad1af9403be583528022b293e70a41f70f16682c3fde610b5d8b2709952be`.
- Passive `n154_six_gate_hybrid_shadow_017` passed: zero reset/arm/setpoint/
  disarm, `298/298` visible replay samples, maximum action error
  `1.6721377373019042e-7`, inference `29.752 Hz`, all deployment hashes, no-op
  cadence `596/10.016 = 59.405 Hz`, and zero rate/health issue. Shadow/parity
  SHA-256 values are
  `4e082189f6015abcf1ef95546c1b9ee9c7c1a8818ca7509601e2a30c4b6a451a`
  and `ac0b9fb2d4d8374004f3c5cd9440b3a95b113920a70c4e9e00471e6b5fcc7b20`.
- Preregister exactly one bounded Candidate 015 under tag
  `n154_six_gate_hybrid_gate4_bounded_015`: current pinned wrapper, Windows
  Python, reuse PID `24476`, one detected command-`31000` reset, requested
  `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/minimum/stop index
  `4`. Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, hashes,
  zero collision/invalid/dropout/drain/rate issue, exit disarm and passive
  proof. Reject failure without unchanged retry. FullLap remains unauthorized.

## N154 Candidate 015 Rejected at Gate 2; N155 Projected-Aperture Governor

- Candidate 015 ran once and is rejected. PID `24476` was reused; command
  `31000` reset was detected (`188703 -> 3403 ms`, race start `3278 ms`,
  detection `0.516 s`, calibration `60/60`). It passed Gate 1, then collided
  at official index `1`, finish `-1`: ID `1001`, threat `2`, impact
  `9.644123077392578 m/s`. Publication was `438/7.25 = 60.276 Hz`, zero
  rate/dropout/drain issue; `2632` messages, `1328` IMU, `93` frames,
  `92` detections. Aggregate/per-attempt/smoke/CSV hashes:
  `6c75db9426b9bc03e9d965a775534d8ebab3fed2f6f039d92f64aa99f0145563`,
  `f6313de16fb79100cd8224581a6b1ac3fcc7798dc34bddbb4ea2fe253c3212e4`,
  `89338315b99eae47d5d5722dce145c0adafb803dc89a9ce11823fa8c678c2833`,
  `7dc8ffb5a50ad434d477e6e5ac861833c487ed194a410d9192978daada60fd18`.
  Passive disarm snapshot SHA-256:
  `24d3064cddf3d0b1603e3ed6e3760998c454aa88365603c5334cce42a3702b77`.
- Candidate 015's final visible Gate-2 pose was `[2.047,0.822,0.115] m`;
  observable rate projected `+0.706 m` right at the plane. Candidate 014
  covered the opposite edge at `-0.658 m`. N155 adds a Gate-2-only governor:
  inside `6 m`, if absolute projected lateral miss exceeds `0.5 m`, cap pitch
  action at `-0.2` and use bounded observable lateral PD roll (`kp=0.6`,
  `kd=0.35`, limit `0.7`). Gate 1, Gate 3+, thrust, and yaw are unchanged.
- Archived Candidates 013--015 evidence passes with zero blockers:
  `logs/sitl/n155_gate2_projected_aperture_governor_20260717.json`. Evidence/
  policy/verifier/wrapper SHA-256 values are
  `336ff1e1b2b5cb9449ef592ef481c4a442208fb8c31c84393439cdde3c431faa`,
  `440a3bee51c3ceb4d039dec6757e970e348b989cd2d03c91d75f5d71100aaa21`,
  `5fd9e03e1270c20e4b90fd994aba44af509fcac503a72c30d151f19f87000410`,
  `8b0a55a49c9d68deb543ea19cf3113158522f962300d1b31493a8e6fc3f5bdb5`.
  Focused tests pass `28/28`; PowerShell parses.
- No N155 flight yet. Use one reset-only MAVLink command `31000` with detected
  rollback and visible Gate 1, then passive zero-command
  `n155_six_gate_hybrid_shadow_018`. FullLap remains unauthorized.

## N155 Reset and Passive Shadow 018 — Passed

- Reset-only command `31000` rolled boot `38048 -> 3124 ms`, race start
  `3306 ms`, visible Gate 1, base `193`, status `4`, index `0`, finish `-1`.
  Setup SHA-256:
  `937039ba64beddd9a7a39c3912a15acfe8dbdd981a84d7446492c6589a1643d0`.
- Shadow 018 sent zero reset/arm/setpoint/disarm and passed: `238/238` visible
  samples, maximum error `1.8356218338400065e-7`, inference `23.762 Hz`, all
  hashes, no-op cadence `568/10.016 = 56.609 Hz`, zero health/rate issue.
  Shadow/parity SHA-256:
  `0ec4680f3863024bd7a82c348695b6b8c48f62f45a47fe8a415532ca438c75f6`,
  `73b5f94168771a87c6c2403c2c8e4057976830219d9aa4fdcc30bd539c3f567e`.
- Preregister exactly one bounded Candidate 016 under tag
  `n155_six_gate_hybrid_gate4_bounded_016`: reuse PID `24476`, current pinned
  wrapper and Windows Python, one detected command-`31000` reset, `80 Hz`,
  maximum `30 s`, repeats/min-valid `1/1`, target/minimum/stop index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, hashes, zero
  health/rate issue, exit disarm plus passive proof. Reject without unchanged
  retry; FullLap unauthorized.

## N155 Candidate 016 Passed Gates 1--3; Rejected at Gate 4

- Candidate 016 ran exactly once and is rejected without unchanged retry. It
  reused responsive simulator PID `24476`; the normal MAVLink `COMMAND_LONG`
  command `31000` reset was detected (`140703 -> 3288 ms`, race start
  `3300 ms`, detection `0.531 s`, calibration `60/60`). It passed official
  Gates 1, 2, and 3. N155's Gate-2 governor activated once and converted the
  prior edge trajectory to a clean Gate-2 pass. At official index `3` it then
  collided with ID `1002`, threat `2`, impact `5.396681785583496 m/s`; finish
  remained `-1`.
- Transport was healthy: `1195/19.812 = 60.267 Hz`, duration `20.113 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit; `5399`
  messages, `2725` IMU samples, `252` detector frames, and `238` detections.
  Aggregate/per-attempt/smoke/CSV SHA-256 values are
  `839cb7e3bd5b83a716add57db441db0fb6881f955efef18d387210763c1ae0e1`,
  `549b46db4279b29f32ebbd0e9dd34503381246044b902c6270828254872044dc`,
  `556b05f935077baf16d69f5c0d7f6fa77067a4ef8f978d1c9e2ee178e9050af9`,
  and `9e7ae6c195708c01a3418f8c5b9b04aed162daae58a9c080b059be02bf61da13`.
  Passive post-stop snapshot SHA-256 is
  `340542e8c4e2c0e9cd2f3553165743bae78094eb3179d1c5751f45f6fbbb524d`.
- The Gate-4 transition cache was correctly empty on the first policy tick,
  so this is not the old cross-gate alias. The current tail accepted several
  mutually inconsistent aperture families and descended/aggressively rolled
  before a stable active-gate approach. Visual inspection corrects the initial
  diagnosis: the final raw detection is not an unrelated sign; it is the
  cropped right upright/panel of the actual Gate-4 frame, while the closest
  saved frame shows its full scoring aperture. Thus the vehicle missed the
  opening laterally while the tail retained excessive authority. Historical
  Candidate-143 evidence isolated the same ID-1002 risk and proved a safer
  Gate-4 bridge: raw observation at index
  `3`, one-second acquisition grace, `8 m` step and `12 m` anchor bounds,
  bounded lateral roll, delayed/range-gated descent, and hover fail-safe on
  dropout/rejection/far observations. N156 must port that behavior into the
  current 32-input hybrid while preserving N155 Gates 1--3. No N156 flight or
  FullLap is authorized before offline evidence and a zero-command shadow.

## N156 Gate-4 Safe Handoff — Offline Passed

- N156 preserves the complete N155 Gates-1-through-3 path and advances the
  learned Gate-4 recurrent state, but bounds its visible output with the
  retained live-safe handoff invariants. Gate 4 alone receives raw pose:
  one-second acquisition grace, `8 m` consecutive step and `12 m` anchor
  bounds, at most two coherent `0.125 s` close re-anchors inside
  `20.75 m`/`4.5 m`, `0.02` lateral gain with `0.10 rad` roll cap, yaw
  centering, `0.12 rad` brake, three-second/current-pose descent gate, minimum
  thrust `0.22`, and level `0.27` hover plus yaw freeze on rejection/dropout.
  The runner now supports a raw-observation end index; the wrapper sets start
  and end to `3`, so Gates 5--6 retain their trained motion-filter ABI.
- Deterministic replay of all Candidate-016 Gate-4 evidence passes with zero
  blockers: `80` samples, `71` raw-visible and `9` dropout, both retained close
  re-anchors exercised, maximum governed normalized roll `0.2`, minimum thrust
  `-0.5555556` (decoded `0.22`), and every dropout level/hover. At the closest
  raw aperture sample (`4.909/3.620/1.350 m`), the failed action
  `[0.642,-0.943,-0.943,0.001]` becomes bounded
  `[-0.24,-0.145,-0.30,-0.202]`. Evidence report
  `logs/sitl/n156_gate4_safe_handoff_20260717.json` SHA-256 is
  `61eb7e8610920751b9eb6df0a4b0a351368efdc3fa99f9e064e3cb7a7b55f5e6`.
- Policy/runner/verifier/wrapper SHA-256 values are
  `4edec25979ecfab0a59b16ec05cec80673389ccf7cd622cbb10393ce2ca23302`,
  `0cb94723a2e6e266c690a0127c4c60450479a8aacc61d5594be51f33be4d0936`,
  `a72619c768f55b941398c0d93773601b9f79239c758ed7cb0e664dd3e7d1229f`,
  and `f62757b3221cb35af02729c3091d8718aaf445f32a108d84029dcc206ab0fd00`.
  Linux deployment tests pass `69/69`; actual Windows targeted tests pass
  `16/16`, Windows compilation succeeds, and PowerShell parses.
- Do not fly N156 yet. Reuse PID `24476`; send exactly one separately logged
  reset-only MAVLink `COMMAND_LONG` command `31000`, require detected boot/race
  rollback and visible Gate 1, then run passive zero-command
  `n156_six_gate_hybrid_shadow_019`. No FullLap.

## N156 Lifecycle Recovery and Reset Setup

- Passive pre-setup telemetry proved the preserved PID `24476` was inactive
  after Candidate 016: boot `1741494 ms`, race start `-1`, base `65`, status
  `3`. The normal `31000` command was therefore not sent yet. The recovery
  helper correctly chose UI recovery without relaunch, but its stored Back To
  Main Menu coordinate clicked Toggle HUD and left the pause menu open. The
  failure stayed passive; its race-start report hash is
  `8e97abbff31d93eae809721b551a38baa4db88343462fad1d08486775d73ebbd`.
- Live window capture located the v3385 Back To Main center at relative
  `(480,805)`, not `(480,755)`. The same PID was returned to the main screen,
  the sole VQ1/R1 event selected, and passive telemetry proved base `193`,
  status `4`, index `0`, finish `-1`, race start `2050755 ms`, and visible Gate
  1. Startup helper SHA-256 is now
  `5122b892ebae263d5c28ecbf8694578fab353646a1cfd6363e44fb36bb807d40`;
  wrapper SHA-256 is
  `f62757b3221cb35af02729c3091d8718aaf445f32a108d84029dcc206ab0fd00`.
  Focused tests pass `62/62`; both PowerShell scripts parse.
- Exactly one reset-only MAVLink command `31000` then rolled boot
  `2093597 -> 3085 ms`; new race start `3313 ms`, base `193`, status `4`, index
  `0`, finish `-1`, visible Gate 1. Setup snapshot SHA-256 is
  `bccac933e615ce9738961deb16b087ff3a5b163218363d6cb6f19d7e059eb87d`.
  Next run only passive zero-command Shadow 019. No flight or FullLap.

## N156 Passive Shadow 019 — Passed; Candidate 017 Bounded

- Passive `n156_six_gate_hybrid_shadow_019` sent exactly zero reset, arm,
  MAVLink setpoint, and disarm commands. All `311/311` samples were visible and
  replayed within maximum action error `2.2075195310611306e-7`; inference was
  `31.1 Hz`, every one of 11 deployment hashes matched, and no-op publisher
  cadence was `594/9.969 = 59.484 Hz` with zero violation/setpoint. Official
  index stayed `0`, finish `-1`; collision/dropout/drain counts were zero.
  Shadow/parity SHA-256 values are
  `e029e12a27cfee7d7d32b0af0e0757d3a9682b2f81b7f08502fa928277c9ca98`
  and `6a42838d8aa0e2ab816a1af027c8fd3218ab88369dd483b9fbe113a6041ee548`.
- Preregister exactly one bounded Candidate 017 under tag
  `n156_six_gate_hybrid_gate4_bounded_017`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected MAVLink command-`31000`
  reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/
  minimum/authoritative stop official index `4`. Accept only clean index `4`,
  finish `-1`, actual cadence `[50,100) Hz`, matching hashes, zero collision/
  invalid/dropout/drain/rate issue, one exit disarm and passive disarm proof.
  Reject failure without unchanged retry. FullLap remains unauthorized.


## N190 Candidate 050 — Gate-2 Vertical-Acquisition Stall

- Candidate 050 ran exactly once on reused responsive PID `24476`; no
  simulator process was launched. The tracked reset path sent normal disarm
  before learned-target MAVLink `COMMAND_LONG 31000`, confirmation and
  parameters 1--7 zero. Reset was detected in `0.5 s`: official boot rolled
  `156600 -> 3419 ms`, fresh race start was `3295 ms`, index was `0`, finish
  `-1`, and calibration completed `60/60` samples.
- The run cleanly passed Gate 1, then never passed Gate 2. It ended at the
  preregistered `45 s` bound with official index `1`, finish `-1`, and no
  collision. Transport was healthy: `2716` commands over `44.906 s`
  (`60.460 Hz`), zero rate violations, dropouts, drain hits, or malformed
  messages, `11110` telemetry messages, `557` camera frames, and `98`
  detections. The command-free post-stop capture sent no reset/control
  command and confirmed disarmed/standby base/status `65/3`, index `1`,
  finish `-1`, and no collision.
- N190 was not causal and supplied no treatment on this path. Its projected
  condition was true on `344` recorded Gate-2 samples, but it changed exactly
  zero commands because base normalized thrust was already above `+0.3`.
  The last real detections aged out, while the observable motion-filter hold
  remained marked visible near `5.177 m` forward and official index `1`.
  Gate-3 samples and N189 severe/PD activations were all zero. This is a
  Gate-2 severe vertical-acquisition miss followed by stale held observation,
  not evidence about the Gate-3 PD.
- Smoke/attempt/summary/post-stop/N190-diagnostic/N189-diagnostic/transition-
  comparison SHA-256 values are
  `c4d47dd62e4e502306e4ee35c9f780c282450c6415c2b6c3939efb75ea4d8e32`,
  `0731f3082c78fa46f24b24778620d89e89bb2ffd22188058f195659a74584311`,
  `a910eaef565458cd8d3192e540739615456d17b83b98488bf24637bfa1ff1458`,
  `4ad1b886c09daf099442981cbcf076472a2fd4d4b3b78e7e7184c47196f4b007`,
  `11a92e47009d39038d17f85dbfa7b7d887a639c4d4d17c6bb851a772df5d19f9`,
  `2917dfb52f3482d34a2b5e13ca54a604af67a3f4022546f5b9fa0345046a4060`,
  and `22f6ec411ac1f578ed643791b791cfda13b8e37ed6db834d5461e1be36628783`.
- Candidate 050 is rejected and must not be retried unchanged. Candidate 051
  and FullLap are unauthorized pending an offline-separated mechanism, hashes,
  tests, an exact reset proof, and a passive zero-command shadow.

## N191 Early Severe Gate-2 Vertical Floor — Offline Freeze

- N191 keeps N190, N189, and the learned prefix source-hash exact. It adds one
  stateless Gate-2-only severe rule before the existing close moderate floor:
  while visible and inside `16 m`, project down position to the gate plane
  with the existing `3 m/s` closing-speed floor; only if projected down is
  below `-4.0 m`, floor normalized thrust at `+1.0`. It changes no pitch,
  roll, or yaw, does not latch, and releases as soon as the projection
  recovers. N190's close `-1.1 m/+0.3` floor and N189's Gate-3 composition
  remain unchanged.
- Replay across Candidates 037--050 gives exact activation separation.
  Candidates 037--049 have zero severe activations; their worst projection is
  no lower than `-3.277 m`. Candidate 050 alone reaches about `-5.988 m` and
  receives exactly three thrust-only changes at `6.266`, `6.438`, and
  `6.579 s`, raising recorded base thrust `0.495/0.313/0.290` to `1.0`.
  There are zero non-Gate2 changes, non-thrust changes, invalid outputs, or
  invariant blockers. This is bounded trace evidence, not counterfactual pass
  proof.
- Actual Windows passes `18/18` focused N187--N191 controller, verifier,
  composition, and runner tests; the PowerShell runner parses. Policy/verifier/
  offline-report/deployment-runner SHA-256 values are
  `f301b682edc10efffc3c131910ea31a479cd323beb888c1d3ddccfd1f13bd73a`,
  `5c974f623d3c25dd32951575d780506c64a36d6cb449a4fe7be49cc203936e99`,
  `297f24aa64a58e66c6c416c15b7f4ea4f0d48c780e019467272c12c5cc6b919e`,
  and `f14ddfd8854dbf96309f55015b77e36ae2e83085f751114888a220711abeef5b`.
- No simulator control command occurred during N191 development. Candidate
  051 and FullLap remain unauthorized. Next only: reuse PID `24476` without
  relaunch; because Candidate 050 stopped inactive, recover sole VQ1/R1 through
  the UI with `-NoRelaunch -SkipLoginClick`; then use the current tracked
  normal-disarm plus learned-target `COMMAND_LONG 31000` path. Require boot
  rollback over `1000 ms`, fresh nonnegative race start, index `0`, finish
  `-1`, and visible Gate 1. Then run passive zero-command Shadow 052 with
  `-Gate2SevereVerticalN190`; it must send zero reset, arm, setpoint, and
  disarm commands and pass exact hash/parity, cadence, telemetry, camera, and
  fault checks before any separately preregistered bounded flight.

## N191 Live Reset Proof and Passive Shadow 052 — Passed

- The simulator remained responsive on PID `24476`; no process was launched.
  The same VQ1/R1 session was already active and reset-ready, so
  `-NoRelaunch -SkipLoginClick` correctly made no UI clicks. The command-free
  ready proof showed base/status `193/4`, index `0`, finish `-1`, and old
  boot/race epoch `92169/3289 ms`.
- The current tracked reset path then learned target IDs, sent normal disarm,
  waited `100 ms`, and sent exactly one MAVLink `COMMAND_LONG 31000` with
  confirmation/parameters 1--7 zero. Boot rolled back to `3167 ms`, a fresh
  race start appeared at `3282 ms`, and the result was base/status `193/4`,
  index `0`, finish `-1`, visible Gate 1, healthy telemetry, and no collision.
- Passive `n191_gate2_severe_vertical_n190_shadow_052` passed and sent zero
  reset, arm, MAVLink setpoint, and disarm commands. No-op cadence was
  `537/9.953 = 53.853 Hz` with zero violations. All `223/223` visible samples
  replayed at `22.3 Hz` with maximum action error
  `1.9613943099891507e-7`; every deployment hash matched. It observed `2423`
  telemetry messages and `106/106` camera detections with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- Ready/reset/shadow/parity SHA-256 values are
  `c92790e728a2dbca6f9a5366d7caf6f899cfe60f655ebf137a3c17f340832120`,
  `1c355ac5763656b5f8f8a55b21b435fa316172cee1e69667f1f72be7ed56b9c3`,
  `0b71d79789b9bdba56e193e959cfe2ab95658c67eeae8ff0599ace34af5b9734`,
  and `87d44efd9a3377e98fbf912444cf514d6b5120dc92a2aadb18118950d23cba55`.
- Authorize exactly one Candidate 051 `BoundedGate3` under tag
  `n191_gate2_severe_vertical_n190_bounded_051`: frozen N191 hashes, actual
  Windows Python, reused PID `24476`, one detected normal-disarm-then-command-
  `31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`,
  and target/minimum/authoritative stop official index `3`. Accept only a
  clean index-3 stop with finish `-1`, cadence `[50,100) Hz`, exact hashes,
  zero rate/collision/invalid/dropout/drain issue, one exit disarm, and a
  command-free passive post-stop proof. Reject failure without unchanged
  retry. FullLap remains unauthorized.

## Current Status Override — N167 Awaiting Shadow 030

This tail status supersedes every earlier preregistration or next-action note
in this file. Candidate 027 was executed once, passed only official Gates 1--2,
and was rejected at Gate 3/index `2` after collision `1001` at impact
`6.9826836585998535`; finish stayed `-1`. Its normal MAVLink reset was
`COMMAND_LONG` command `31000`, proved by boot rollback `126297 -> 3389 ms`.
The full result and N167 evidence are recorded above under "N166 Candidate 027
Rejected at Gate 3" and "N167 Gate-3 Pitch-Brake Retirement — Offline Passed."
The current pinned wrapper is N167 (`2c3805f9...`), which restores recurrent
Gate-3 pitch while retaining terminal roll level and the vertical thrust floor.
The only authorized next sequence is same-PID reset-ready recovery, exactly one
reset-only command `31000` with measured boot/race rollback, then passive zero-
command `n167_six_gate_hybrid_shadow_030`. Candidate 028 and FullLap remain
unauthorized until that shadow passes.

## N166 Candidate 027 Rejected at Gate 3

- Candidate 027 ran exactly once under tag
  `n166_six_gate_hybrid_gate4_bounded_027` and is rejected without an
  unchanged retry. The responsive simulator PID `24476` was reused. The
  runner sent one normal MAVLink `COMMAND_LONG` command `31000`; telemetry
  proved reset success by rolling simulator boot time `126297 -> 3389 ms`,
  creating race epoch `3316 ms`, and detecting the rollback in `0.516 s`.
  Calibration completed `60/60` before policy publication.
- It passed only official Gates 1 and 2. At active Gate 3/index `2`, collision
  ID `1001`, threat `2`, impact `6.9826836585998535 m/s` aborted the run;
  official finish stayed `-1`. The vision diagnostic counted a third apparent
  pass, but it is not authoritative. The last Gate-3 detection was nearly
  centered and level: about `2.248 m` forward, `+0.011 m` right, `-0.105 m`
  down, image center `(321.5,165.0)`, normalized roll `0.0`, and forced pitch
  `-0.2`.
- Transport was healthy: `638/10.281 = 61.959 Hz`, duration `10.390 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit; `3283`
  messages, `1658` IMU samples, `133` frames, `130` detections, and one exit
  disarm. Aggregate/attempt/smoke/CSV SHA-256 values are
  `ce8ec534a079cc4d3d79b7b898427bedcfa6e1c7e05760a1ef4f8045a9ad07be`,
  `2b7d5e25d035c11d39f8a6af66f1543a210ffc7cc317e188003270448bc0f6ed`,
  `d48fc9fa6ab2a306aca602010a521933ef86f720b8d2061713541869d9993f66`,
  and `57ebceff8b4321ff8c4ecd712e51840c6b41ab41f322f826a1d97abc4fc2ebc2`.
- The command-free passive post-stop snapshot proves base/status `65/3`,
  official index `2`, finish `-1`, black post-collision frames, and
  `reset_sent=false`. Its SHA-256 is
  `9d64a4de0f5accda2cb2e372c4cb37ce5a7ba4623708b154484bae4f9522339a`.
  A sent reset is never proof by itself; require measured boot-time rollback.

## N167 Gate-3 Pitch-Brake Retirement — Offline Passed

- N167 removes only the N165 early and N164 terminal Gate-3 normalized pitch
  cap `-0.2`; recurrent pitch again owns that channel. It retains the promoted
  roll intercept/close boost, N166 projected-negative-down hover-thrust floor,
  terminal safe-aperture roll leveling, yaw, Gate 2, and the Gate-4 tail.
- Source-hashed comparison covers five archived official Gate-3 passes
  (`016/018/020/021/022`), counter-roll failure `024`, and forced-brake
  failures `025/026/027`. All `18/18` close samples in the passing runs used
  positive learned pitch. All `9/9` close samples in `025/026/027` were forced
  to `-0.2`, and all three failed. Candidate 024 still exercises two full-
  negative-roll corrections by the retained terminal leveler.
- N167 produces zero pitch/yaw changes across all nine traces. The vertical
  floor also produces zero roll or positive-thrust change; the terminal
  leveler produces zero pitch/thrust/yaw change. Gate-2 and Gate-4 preservation
  pass with zero blockers.
- Policy/pitch-retirement verifier/evidence/vertical-floor/terminal-level/
  Gate-2/Gate-4/wrapper SHA-256 values are
  `91ceaf1a14f59610496f2116b69cb86c2f5c90ecc372bcc4fe1c4ea03d43cd2b`,
  `12786cca8b346893fd1edd042776c9fd0ed126fd89d842d9c808cdbd84a04a6a`,
  `6387e509ef3db61615702c7364ce75b4a66d584c3668f6fbd626647bf35b96ba`,
  `a58cb6bb852d238d6d1afb4905aeb083c82eeff6b03f06ea21cada4d575fce97`,
  `56e52245ae102a5dc360414516488a3d8814ca2740a33dfa74544b36ad6d8c88`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `d707f0cb30034c9f642e4e0d44f2e5316a4969a79b5b2c3a60982a2f0acc2df7`,
  and `2c3805f9033c309d8b9971d37d6d6cb949f7ac868a45ec108ac06b0bd733b762`.
  Linux and actual-Windows focused suites pass `63/63`; compilation and both
  PowerShell launcher parses pass.
- Do not fly N167 yet. Recover PID `24476` to reset-ready VQ1/R1 without
  relaunch if possible. Then authorize exactly one separately logged reset-
  only MAVLink `COMMAND_LONG` command `31000`, requiring boot/race rollback,
  official index `0`, finish `-1`, and visible Gate 1. After that, run only
  passive zero-command `n167_six_gate_hybrid_shadow_030`. No arm, flight,
  second reset, Candidate 028, or FullLap until Shadow 030 passes.

## N162 Recovery, Reset-31000 Setup, and Passive Shadow 025 — Passed

- Recovered the sole VQ1/R1 event on the same responsive simulator PID
  `24476`; no relaunch or flight command occurred. Recovery ended at base/status
  `193/4`, official index `0`, finish `-1`. Recovery SHA-256:
  `b299444ea7ef46aad8f28e9333e2f30d3380d51192cc91c644c915c37a8cb395`.
- Exactly one reset-only MAVLink `COMMAND_LONG` command `31000` then rolled
  simulator boot `583001 -> 4437 ms`; race start was `3338 ms`, base/status
  `193/4`, index `0`, finish `-1`, with zero collision and a non-black visible
  Gate 1 (`72` detections, first confidence `0.80257`). No flight setpoint was
  sent. Setup snapshot SHA-256:
  `7f8c8d2dc387704653125dc447fad9582aedda05f713fcdc04f1f14886002403`.
- Passive `n162_six_gate_hybrid_shadow_025` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. Replay passed `313/313` visible samples with
  maximum action error `1.9664711952555036e-7`; inference was
  `31.25312031952072 Hz`, all 11 deployment hashes matched, and no-op cadence
  was `611/10.015 = 60.909 Hz`. Official index stayed `0`, finish `-1`;
  collision/dropout/drain/rate counts were zero. Telemetry/camera totals were
  `2438` messages, `1255` IMU samples, `142` completed frames, and `126`
  detections. Shadow/parity SHA-256 values:
  `e709c89653c897df160c438fbd09515c6cb04d7e974e08a53978321b5b088444`,
  `838fd78f37edb339db728d72ebb7e4bbd9504bc421bfb7ba3eacd7634bb7cfcc`.
- Preregister exactly one bounded Candidate 023 under tag
  `n162_six_gate_hybrid_gate4_bounded_023`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected MAVLink command-`31000`
  reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/
  minimum/authoritative stop official index `4`. Accept only clean index `4`,
  finish `-1`, actual cadence `[50,100) Hz`, matching hashes, zero collision/
  invalid/dropout/drain/rate issue, one exit disarm and passive disarm proof.
  Reject failure without unchanged retry. FullLap remains unauthorized.

## Candidate 023 — Rejected at Gate 2; No Retry

- Candidate 023 ran exactly once on reused PID `24476`. Its sole MAVLink
  command-`31000` reset was detected (`283598 -> 3412 ms`, race start
  `3300 ms`, detection `0.500 s`, calibration `60/60`). It passed Gate 1, then
  hit Gate-2 geometry at collision-time official index `1`, finish `-1`:
  collision ID `1001`, threat `2`, impact `4.348381042480469 m/s`. N162's
  Gate-4-only mechanism was never reached or evaluated.
- Publication was `416/7.000 = 59.286 Hz`; zero command-rate, telemetry
  dropout, or drain-limit issue; `2535` messages, `1280` IMU samples, `89`
  frames, `85` detections, and exactly one exit disarm. Aggregate/attempt/
  smoke/CSV SHA-256 values:
  `8c050294f2a16191e51369032b0a1caef1003c8a91ca9476c4d4c6fe052d38b9`,
  `6487400add7765f58daf6aafb69799c510feaeb7dfdbbe5e1e05c3644ef87f68`,
  `c9fc184a8641272985a1949e890359b7078a3ca94bec104ccbb514a3f0ec864d`,
  `3fe1884952dd628f282dbd49e8ee9789da2e51cc8d46836eb6ca454c8d2366cf`.
- Passive post-stop proof sent no reset or setpoint and found inactive
  base/status `65/3`, finish `-1`. Momentum after the collision eventually
  advanced the displayed index to `2`; this is not a valid Gate-2 pass because
  the official collision made the run invalid. Passive SHA-256:
  `91a9e7270ed27cd912535f3c3ae22b5823900bdfab090024de6aef8e3158be35`.
- The existing Gate-2 governor did run, but its `6 m` forward admission first
  latched at `4.055 m`, projected miss `+0.502 m`, only `0.344 s` before the
  collision trigger. The same accepted trajectory already projected
  `+0.845 m` at `11.847 m`. Diagnostic SHA-256:
  `b827e2d471e2e661d6c4c342bfb98bdf4527b09c8ed38662f62a724be86bef48`.
  Reject unchanged retry. Next prove an earlier Gate-2 admission window offline;
  no flight or FullLap is authorized yet.

## N163 Gate-2 Early-Brake Window — Offline Passed

- N163 makes one bounded Gate-2 change: an unsafe projected miss may latch the
  existing `-0.2` normalized brake pitch inside `12 m` instead of waiting for
  `6 m`. Between `6..12 m`, recorded learned roll, thrust, and yaw are preserved
  exactly. Inside `6 m`, the existing `kp=0.6`, `kd=0.35`, `0.7`-limited roll
  PD remains unchanged. Gates 1 and 3--6, including all N162 Gate-4 laws, are
  unchanged.
- Candidates 016--023 replay with zero blockers. Early-brake sample counts are
  `4/6/4/5/3/5/5/6`; measured lead over the close governor is
  `0.468/0.687/0.750/0.578/0.344/0.656/0.641/0.687 s`. Every early sample has
  zero roll/thrust/yaw error. Candidate 023 now latches at `11.847 m`, projected
  miss `+0.845 m`, while retaining its learned `-0.221` roll; close PD still
  starts at `4.055 m`.
- The independent N162 Gate-4 verifier also passes unchanged: carry/expiry,
  anchors/reanchors, zero non-yaw violations, and zero pre-anchor roll
  violations remain exact.
- Policy/Gate-2-verifier/Gate-2-evidence/Gate-4-preservation/wrapper SHA-256:
  `80cb465c4d5a2eea8bc9e557c121f1d3382761fc0666457d65d0e54c10f677f8`,
  `29ba34411310c888191351abdb4a969440debabf78aac4157664bf73f2f3bd96`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `2d469b512e83fd86ae63598ee615f8721a485e92f06d44e0e996b9a28a73b50f`,
  `38f3f400b7fc74c98b8b81430f34018bdb5fcd80a9f7514aebb355d5527e85ba`.
  Linux deployment tests pass `132/132`; actual Windows tests pass `34/34`;
  Windows byte-compilation and both PowerShell parsers pass.
- No N163 flight yet. Recover the sole VQ1/R1 race on responsive PID `24476`
  without relaunch, send exactly one reset-only MAVLink `COMMAND_LONG` command
  `31000` and require detected boot rollback/index `0`/finish `-1`/visible Gate
  1, then run passive `n163_six_gate_hybrid_shadow_026` only. FullLap remains
  unauthorized.

## N163 Recovery, Reset-31000 Setup, and Passive Shadow 026 — Passed

- Reused responsive simulator PID `24476`; UI recovery restored sole VQ1/R1
  to base/status `193/4`, official index `0`, finish `-1`, without relaunch,
  reset, or flight. Recovery SHA-256:
  `625f20efec9c3d673da11edb8919d65373abf7ad8f7413fa3a96fb2cb1e06aba`.
- Exactly one reset-only MAVLink `COMMAND_LONG` command `31000` rolled boot
  `1119238 -> 2942 ms`; new race start `3269 ms`, base/status `193/4`, index
  `0`, finish `-1`, zero collision, non-black visible Gate 1 (`152`
  detections, first confidence `0.76460`). No flight setpoint was sent. Setup
  SHA-256:
  `51b4576dfb17a3be70a02630dbdcde57d4fb5e5ac5399c1011c297eeaba2180c`.
- Passive `n163_six_gate_hybrid_shadow_026` sent zero reset/arm/setpoint/
  disarm. Replay passed `324/324`, maximum error `2.651107788298468e-7`,
  inference `32.4 Hz`, all 11 deployment hashes, no-op cadence
  `623/10.0 = 62.2 Hz`; index `0`, finish `-1`, zero collision/dropout/drain/
  rate issue; `2423` messages, `1248` IMU, `143` completed frames, `125`
  detections. Shadow/parity SHA-256:
  `98f9925ba06fbd9bc732b0a6d36100c04565f8bb0a56792ddc7df26df98c259a`,
  `73525574036ba95b4e709bfa171958eeaf22b29ec3d2dfac1a8657c12e0f4113`.
- Preregister exactly one bounded Candidate 024 under tag
  `n163_six_gate_hybrid_gate4_bounded_024`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop official index `4`. Accept only clean index `4`, finish
  `-1`, cadence `[50,100) Hz`, matching hashes, zero collision/invalid/dropout/
  drain/rate issue, one exit disarm and passive disarm proof. Reject without
  unchanged retry. FullLap remains unauthorized.

## Candidate 024 — N163 Passed Gate 2; Rejected at Gate 3

- Candidate 024 ran exactly once on reused PID `24476`. Its command-`31000`
  reset was detected (`175602 -> 3416 ms`, race start `3309 ms`, detection
  `0.500 s`, calibration `60/60`). N163 cleanly advanced official Gates 1 and
  2, proving the early-brake change live, then the existing Gate-3 controller
  collided at official index `2`, finish `-1`: ID `1001`, threat `2`, impact
  `10.180516242980957 m/s`. Gate 4 was not reached.
- Publication `628/10.406 = 60.254 Hz`; zero rate/dropout/drain issue; `3289`
  messages, `1661` IMU, `133` frames, `132` detections, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive SHA-256:
  `950d02b452a70c9645c945d1e343ec9af73cd7615f75b5cbf980d8cc7f35477d`,
  `a5ccbf982fe988b9c9314dfebf1ab199fdf2852289d95f2b3acd515b51238bab`,
  `5c790b9cefb2caa58bd02a52964cd5a07e94923178b219dc2279590c9cf8bddf`,
  `ccfbae57ca1a7fe1f3f01a7f780c36e5cb50cc482d1d988853358d13f4f217e8`,
  `be6c4cc343fe08f6c34412e54d89e1ac9561983b18ca8ede979834d5499ea8ab`.
  Passive state was base/status `65/3`, index `2`, finish `-1`, no reset.
- Gate-3 trace reached a safe projected aperture at `5.38 m`: lateral
  `-0.37 m`, vertical `-0.82 m`. The heuristic nevertheless commanded
  normalized pitch/roll `+0.24/-1.0`. Its close counter floor then held roll
  at zero for two samples, but released on the final `+0.19 m` direct-right
  sample and returned full `-1.0` counter-roll at projected lateral `+0.206 m`.
  Diagnostic SHA-256:
  `f1a65cb90d61e1f8ae74b9d036c8a2e518b72a39efe6f7fb1758b3574d3211e`.
- Reject unchanged retry. Next prove a Gate-3 terminal safe-aperture coast:
  latch only inside `6 m` when projected lateral/vertical errors are within
  `0.5/1.0 m`, brake pitch and level roll, preserve thrust/yaw. No flight or
  FullLap before offline evidence and a new passive shadow.

## N164 Gate-3 Terminal Safe-Aperture Coast — Offline Passed

- N164 changes only official Gate 3. Inside `6 m`, when the existing
  observable projection is within `0.5 m` lateral and `1.0 m` vertical, it
  latches normalized brake pitch `-0.2` and level roll `0.0` until official
  transition; learned thrust and yaw are preserved exactly. Before admission,
  the complete N163 controller remains unchanged.
- Candidates 016/018/020/022 have zero activation and are bit-exact. The two
  Gate-3 failures activate as intended: Candidate 019 gets two coast samples
  from `2.850 m`; Candidate 024 gets four from `5.384 m`, first projection
  `[-0.374,-0.822] m`, replacing pitch/roll `+0.240/-1.0` with `-0.2/0.0`.
  Clean Candidate 021 activates only at `1.771 m` for three samples, where its
  recorded roll was already `0.0`; only terminal pitch is braked. All coast
  samples have zero thrust/yaw error and zero pitch/roll-law violation.
- Independent nine-trace N163 Gate-2 preservation and five-trace N162 Gate-4
  preservation reports both pass with zero blockers.
- Policy/verifier/evidence/Gate-2-preservation/Gate-4-preservation/wrapper
  SHA-256:
  `eaa37cc016ee3f78fbff511db949da270e8ab69a0232b43db0933a9bf0168da4`,
  `9ff6f1ac7201adf3a5d199012803f0b68e364d4b1a9d1801adaf890e1edeb91f`,
  `dbd26ae238af9f6cb76d51906abca480783ab63d3cfecffab9f659db1fef9364`,
  `0aa2c8280dfacd6c38a4f2bf1d2046b9e53bce27708b23d66051a8cba209164a`,
  `aca52f7ca35dd6e20a9677e2ad03f50d43bbbcdc680345e65a4f328c795e428b`,
  `1c6697bf5cc760d0ca640620aa717f991cb709d6679bbe81a79e00d734957c22`.
  Linux deployment tests pass `136/136`; actual Windows tests pass `38/38`;
  Windows byte-compilation and both PowerShell parsers pass.
- No N164 flight yet. Recover sole VQ1/R1 on responsive PID `24476` without
  relaunch, send exactly one reset-only MAVLink command `31000` with detected
  boot rollback/index `0`/finish `-1`/visible Gate 1, then only passive
  `n164_six_gate_hybrid_shadow_027`. FullLap remains unauthorized.

## N164 Recovery, Reset-31000 Setup, and Passive Shadow 027 — Passed

- Recovery reused responsive FlightSim PID `24476` without relaunch and
  restored sole VQ1/R1 at base/status `193/4`, official index `0`, finish
  `-1`. Recovery artifact SHA-256 is
  `e1861c289d4358f83a4b075e49c159329a642c2acf40f89145990f74b41a1e9b`.
- Exactly one separately logged reset-only MAVLink `COMMAND_LONG` command
  `31000` rolled boot `1056545 -> 3090 ms`; the new race start was `3307 ms`.
  Setup remained base/status `193/4`, index `0`, finish `-1`, collision-free,
  with `200` visible Gate-1 detections (first confidence `0.7901247`) and no
  flight setpoint. Setup SHA-256 is
  `172cc787d5a71ab53c7f34a1448cafc80a004d36830598f962eee138c1a489d3`.
- Passive Shadow 027 then sent zero reset, arm, MAVLink setpoint, and disarm
  commands. All `297/297` visible samples replayed with maximum action error
  `2.2150001527387886e-7`; inference was `29.656 Hz`, all 11 deployment hashes
  matched, and no-op cadence was `608/9.968 = 60.895 Hz` with zero rate
  violation or setpoint. Official state stayed index `0`, finish `-1`; health
  was clean with `2438` messages, `1256` IMU samples, `143` completed frames,
  and `127` detections. Shadow/parity SHA-256 values are
  `2a12e5f15239904caf9f651a2b4da1fe2e4c0d6db4df1fa60511114b34ffda3d`
  and `6bb172f849ecd4d8b47d058f1b1cefe1d07815baf687d04be8680754936b6807`.
- Preregister exactly one bounded Candidate 025 under tag
  `n164_six_gate_hybrid_gate4_bounded_025`: current pinned wrapper and actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop official index `4`. Accept only clean index `4`, finish
  `-1`, actual cadence `[50,100) Hz`, matching hashes, zero collision/invalid/
  dropout/drain/rate issue, one exit disarm, and passive disarm proof. Reject
  failure without unchanged retry. FullLap remains unauthorized.

## Candidate 025 — Rejected at Gate 3; No Retry

- Candidate 025 ran exactly once on reused PID `24476`. Its MAVLink command
  `31000` reset was detected (`291494 -> 3403 ms`, race start `3269 ms`,
  detection `0.515 s`, calibration `60/60`). Official Gates 1 and 2 passed;
  Gate-3 collision ID `1001`, threat `2`, impact `8.014646530151367 m/s`
  stopped the run at authoritative index `2`, finish `-1`. A third ordered
  vision pass is diagnostic only and does not override official progress.
- Transport was healthy: `601/10.203 = 58.806 Hz`, duration `10.328 s`, zero
  rate/dropout/drain issue; `3246` messages, `1639` IMU, `131` frames, `130`
  detections, one exit disarm. Aggregate/attempt/smoke/CSV SHA-256 values are
  `fc3c345b5e4b60555076b438fd7eb53c0101fb0cf03c7dd2e3e3e53872d1442a`,
  `c4ef5963712dfe7d890c7d2b7c05dfbae93541b5fb026b8d09f1df3467f0b53e`,
  `a60c4318a2112003911046001e1437e5f5a24030694fa87e977a87489d8d9ea1`,
  `89069e751e2860d543954cbda6dd915ba82fc8b0baedebea5e672788e5a7fac9`.
  Passive post-stop proves base/status `65/3`, index `2`, finish `-1`, no
  reset; SHA-256 is
  `493f7011c1022dd6d50c52bffc8cc6d070083959f31f22718469ef2818b7631d`.
- N164 latched at `4.156 m`, projected lateral/vertical `-0.490/-0.212 m`,
  only `0.359 s` before collision. It correctly held pitch/roll `-0.2/0.0`
  for four samples, but the vehicle entered terminal control with excess
  speed/bank. Diagnostic SHA-256 is
  `ef20e8a683f0b53f5591a348c0c6c458d3fe38dfe3db86a10e552384bd72d292`.
  Reject without unchanged retry; Gate 4 was not reached.

## N165 Gate-3 Projected-Miss Early Brake — Offline Passed

- N165 changes only official Gate 3. At a visible closing pose inside `12 m`,
  projected lateral error outside `0.5 m` or vertical error outside `1.0 m`
  latches normalized pitch brake `-0.2` until official transition. It leaves
  learned roll, thrust, and yaw exact; the N164 safe-aperture terminal latch
  remains responsible for leveling roll inside `6 m`.
- Candidates 016/018/019/020/021/022/024/025 all activate at
  `10.438..11.992 m` and receive `6..13` early-brake-only samples with zero
  roll/thrust/yaw error or pitch-law violation. Candidate 024 gains `0.719 s`
  and Candidate 025 gains `0.735 s` before their existing terminal latches.
- Gate-2, N164 terminal-coast, and Gate-4 preservation replays independently
  pass. Policy/verifier/evidence/Gate-2/terminal/Gate-4/wrapper SHA-256:
  `4cca1c0e82c947962f5b99bb4f70d9f3f352d113e104ca27299ba92b4f7979e0`,
  `0b87f7cdbbaaee360caf9761a2915c5a250de799db809b5b247a6919fbd25b16`,
  `88f64d1360e2061aea9a2ede67f43a27f0b05d30f8efcffe4e1671f29fc50277`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `3e5d4d8e64e4d36453cbfe7dac753a77cea5deff8d07073747a605c474aca564`,
  `8768d67b497b5096a14b56d69cc0f82bb1b4cd93d13b9cb13063ecf5712d3ee1`,
  `abba4d696d613bee00f086224d71629daeb68ad67bfb2a240786f1eee667cd2f`.
  Linux and actual Windows targeted suites pass `62/62`; Windows compilation
  and both PowerShell parsers pass.
- No N165 flight yet. Recover sole VQ1/R1 on responsive PID `24476` without
  relaunch, send exactly one reset-only command `31000` with detected rollback/
  index `0`/finish `-1`/visible Gate 1, then run only passive zero-command
  `n165_six_gate_hybrid_shadow_028`. FullLap remains unauthorized.

## N165 Recovery, Reset-31000 Setup, and Passive Shadow 028 — Passed

- Recovery reused PID `24476` without relaunch and restored VQ1/R1 at base/
  status `193/4`, index `0`, finish `-1`. Recovery SHA-256 is
  `d414d98e2885a0cd6becde4046d1768684beaa7ed6124cb7a194d8d1a0af24a9`.
- Exactly one reset-only command `31000` rolled boot `920822 -> 3218 ms`; race
  start `3301 ms`, index `0`, finish `-1`, zero collision, visible Gate 1
  (`210` detections, first confidence `0.85875`), no setpoint. Setup SHA-256:
  `4207e9612c22cd853586d3f953b3bdc9a97bdab82767749d2c1e057b173ba74d`.
- Shadow 028 sent zero reset/arm/setpoint/disarm; all `256/256` visible samples
  replayed with maximum error `1.899723053033764e-7`, inference `25.6 Hz`, all
  11 hashes, no-op `601/9.985 = 60.09 Hz`, index `0`, finish `-1`, zero health/
  rate issue; `2419` messages, `1245` IMU, `142` completed frames, `123`
  detections. Shadow/parity SHA-256:
  `ca88b471ebee77a361de19b022459bab67e5a49c63174c3564e80558a25cd62f`,
  `76e3a4cb1703a25149da3ce848014ecce09a6cf9edeb94fa05e2e9cb9cd53dbe`.
- Preregister exactly one bounded Candidate 026 under tag
  `n165_six_gate_hybrid_gate4_bounded_026`: current pinned wrapper/actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  `80 Hz`, max `45 s`, `1/1`, target/minimum/stop official index `4`. Accept
  only clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes,
  zero collision/invalid/dropout/drain/rate issue, one exit disarm and passive
  proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 026 — Rejected at Gate 3; No Retry

- Candidate 026 ran once on PID `24476`; reset `154304 -> 3422 ms`, race start
  `3306 ms`, detection `0.485 s`, calibration `60/60`. Official Gates 1/2
  passed; collision ID `1001`, threat `2`, impact `8.48799991607666 m/s`
  stopped at index `2`, finish `-1`. Gate 4 was not reached.
- Transport remained healthy: `641/10.594 = 60.412 Hz`, duration `10.688 s`,
  zero rate/dropout/drain issue; `3344` messages, `1688` IMU, `135` frames,
  `130` detections, one exit disarm. Aggregate/attempt/smoke/CSV/passive hashes:
  `7794f37b85547c7a57787c81ea6e4583e87f85d6e0907d0b42ad35e00d87a41f`,
  `f0a2d415e149744800814df77c5c792a2957ec600f0c3f7181193fb0ffec6112`,
  `a83113a633180c39916ec12d19d1892c8936d144c1625633cecf370e3242d506`,
  `286ddc27b4b839bcdbbd3ef570a14ca4593734a9b41f83b29b43bf18a58aefc0`,
  `11a6d2109ff4b3d45032d4788930005d64dad6a7d455d845d7ce6d08eaff9591`.
  Passive base/status `65/3`, index `2`, finish `-1`, no reset.
- N165 early braking activated at `9.751 m`, but projected vertical miss was
  already `-1.835 m` while learned thrust was below hover (`-0.210`
  normalized). Terminal admission at `4.320 m` projected `-0.127/-0.908 m`;
  projected vertical then crossed `-1.065 m`. Early/terminal diagnostic hashes:
  `17f54f00d1a917fcb2c8db73b230e11e7e12040eeb8165356cb94de73cbdda4d`,
  `3b94636ea36320a7f742edf378300c08d6c42112356ba14400cb1603139f7bde`.

## N166 Gate-3 Early Negative-Vertical Thrust Floor — Offline Passed

- N166 changes only the N165 Gate-3 early window. When projected down error is
  below `-1.0 m`, it latches a normalized thrust floor of `0.0` (physical hover
  `0.27`) until official transition. Roll, yaw, every positive/stronger thrust
  command, early pitch brake, N164 terminal law, and other gates are unchanged.
- Nine archived traces activate at `9.751..11.992 m` for `9..13` floor samples.
  Only below-hover samples change (`0..6` per trace); roll/yaw error, positive-
  thrust changes, and floor violations are zero. Candidate 026 changes six
  samples, first from normalized thrust `-0.210` to `0.0`.
- Gate-2, N164 terminal, and Gate-4 preservation pass. Policy/verifier/
  evidence/Gate-2/terminal/Gate-4/wrapper SHA-256:
  `f17479db4c2bfcadb06295c2344fc9565f0c5044b465cbb9630f25ab2c689115`,
  `978c1c4c9af6ac468f24b0e18279c95c655abd3ffdf7f38b8cf3596ac5647605`,
  `37b625628428e5fa1cb0ba051421bb052c6767a891172be8c7e0ca671492faa7`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `d40089d2532bb0cd458c1a925332b63214bd0d0aeee13b40619bd7d2605080ba`,
  `7dbe6d6dff49091aee2070079580e6d99f4edc0af00342c8f9bbec4c197ba630`,
  `a5a429a6718b3940f61026c3c8e4ae61788cb0425ea563898b67fe447c867c44`.
  Linux/actual Windows targeted suites pass `63/63`; compilation and both
  PowerShell parsers pass.
- No N166 flight yet. Recover VQ1/R1 on PID `24476` without relaunch, send one
  reset-only `31000` with detected rollback/index `0`/finish `-1`/Gate 1, then
  only passive zero-command `n166_six_gate_hybrid_shadow_029`. FullLap remains
  unauthorized.

## N166 Recovery, Reset-31000 Setup, and Passive Shadow 029 — Passed

- Recovery reused PID `24476` without relaunch, restoring VQ1/R1 base/status
  `193/4`, index `0`, finish `-1`. Recovery SHA-256:
  `2028b6bcf6e2f376b5430df995369d3f13d79ea41068188870805e36c1371bdc`.
- Exactly one reset-only `31000` rolled boot `824284 -> 3009 ms`, race start
  `3282 ms`, index `0`, finish `-1`, zero collision, visible Gate 1 (`193`
  detections, first confidence `0.6206715`), no setpoint. Setup SHA-256:
  `9946313de517ea51c82768c67c33c67224956f518a9ebac5dc8c30dc1b2db560`.
- Shadow 029 sent zero reset/arm/setpoint/disarm; `238/238`, maximum error
  `1.574008941518379e-7`, inference `23.762 Hz`, all 11 hashes, no-op
  `523/10.0 = 52.2 Hz`, index `0`, finish `-1`, zero health/rate issue; `2414`
  messages, `1242` IMU, `124` completed frames, `107` detections. Shadow/parity:
  `64fe8cdfdb4a71c22c4b4884bbd751deb4f473f7a6fefcc5571c2c207fb98c9a`,
  `2470e798abb9247e047b0dc7aaa5a06447b9b67dd26dd2d03347b447f0c0e25a`.
- Preregister exactly one bounded Candidate 027 under tag
  `n166_six_gate_hybrid_gate4_bounded_027`: pinned wrapper/actual Windows
  Python, reused PID `24476`, detected reset `31000`, `80 Hz`, max `45 s`,
  `1/1`, target/minimum/stop official index `4`; strict clean index/finish/
  cadence/hash/health/exit-disarm/passive proof. Reject without retry. FullLap
  remains unauthorized.

## N158 Candidate 019 Rejected at Gate 3

- Candidate 019 ran exactly once under
  `n158_six_gate_hybrid_gate4_bounded_019` and is rejected without retry. The
  corrected readiness waiter left the active VQ1/R1 UI untouched. One MAVLink
  `COMMAND_LONG` command `31000` then produced the required epoch rollback
  (`358025 -> 3278 ms`, race start `3305 ms`, detection `0.515 s`) before the
  reset-owned arm and policy publication. The same simulator PID `24476` was
  reused; it was not relaunched.
- The run passed official Gates 1 and 2, then collided while Gate 3/index `2`
  was active: ID `1001`, threat `2`, impact `3.598033905029297 m/s`, finish
  `-1`. The vision diagnostic's `ordered_gate_passes=3` is not official race
  progress. N158 Gate-4 control never activated, so its live proof remains
  outstanding.
- Transport was healthy: `622/10.390 = 59.769 Hz`, duration `10.754 s`, zero
  rate violations, telemetry dropouts, or drain-limit hits; `3322` telemetry
  messages, `1678` IMU samples, `132` camera frames, `131` detections, one
  reset-owned arm sequence, and one exit disarm.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `d327e3dfbff14bb7c5a29b5a04744d7465a14e0aa69694b1e8001100282d5b45`,
  `9d024e4aef2590f9d172712bdaef0c9926adfe5b99abf9f283088d725f8a55ea`,
  `c5ea83374a6c119eed8c108398ab5edfa942b508781e200ee1e088e25a5fcb43`,
  `24e5f9dc7c66e2f481a5ddb90e9e39b1ce44682fbed92b39daf51da780da614e`,
  and `23d68708ce0681ff2d34841cf58693ed136b4eeb1f9ae1f9381cf9f73a5a12e2`.
- Gate-3 trace comparison shows the aperture position itself was centered:
  final filtered `forward/right/down` was approximately
  `[2.199,-0.292,-0.191] m` with projected right about `-0.252 m`. The failure
  signature is repeated positive-intercept/full-counter roll reversal. It
  drove measured roll to about `-0.10 rad` before contact, whereas the clean
  Candidate-016 and Candidate-018 Gate-3 crossings retained about `+0.12` and
  `+0.48 rad`. Diagnose and replay a Gate-3-only close-plane roll-persistence
  change before any further reset; preserve Gates 1--2 and N158 Gates 4--6.

## N159 Gate-3 Close-Plane Roll Persistence — Offline Passed

- N159 changes only official Gate 3 roll inside `4.0 m`. A still-severe
  promoted `adaptive` miss retains the existing normalized `+1.0` boost. When
  the projected miss has entered promoted `counter` mode but the directly
  observed gate remains left, the normalized roll floor is `0.0` instead of
  the abrupt `-1.0` counter-bank. Pitch, thrust, yaw, all other ranges, Gates
  1--2, and N158 Gates 4--6 are unchanged.
- Three-trace replay passes with zero blockers. Clean Candidate 016 exercises
  exactly three close counter suppressions; clean Candidate 018 changes zero
  samples; rejected Candidate 019 changes exactly its final three
  `-1.0 -> 0.0` roll actions at `3.927`, `2.850`, and `2.199 m`. Non-roll
  maximum error is zero on all three traces; Candidate 019's final projected
  right remains the measured `-0.252 m`.
- Policy/verifier/evidence/wrapper SHA-256 values are
  `98f08ae181d62621a4fd3c3adb451c9f017aa747c5fdb42883136e42a9a040b9`,
  `e9d49d59fb3d0e8c30b5913195f86fb09c30871449baa01294ad43cd88a1794a`,
  `68d7930b11cd9ed785b0cefcc6a45e109e7f6f716784bd8fbfe5274c31754e89`,
  and `629053098cc940a2a033104f286aaadc98c0e735fc3abce432099de829534ecf`.
  Linux deployment tests pass `85/85`; actual Windows tests pass `24/24`;
  Windows byte-compilation and both PowerShell parses pass.
- No N159 flight yet. Reuse responsive PID `24476`; recover the same process
  to sole VQ1/R1 only if passive status requires it. Then send exactly one
  separately logged reset-only MAVLink `COMMAND_LONG` command `31000` and
  require detected boot/race rollback, base/status `193/4`, index `0`, finish
  `-1`, and visible Gate 1. After that run only passive zero-command
  `n159_six_gate_hybrid_shadow_022`. No additional lifecycle command, flight,
  unchanged retry, relaunch, or FullLap before shadow acceptance.

## N159 Reset-31000 Setup and Passive Shadow 022 — Passed

- Same PID `24476` was recovered through the UI because passive telemetry was
  inactive; it was not relaunched. The sole VQ1/R1 selection restored
  base/status `193/4`, index `0`, finish `-1`. Exactly one reset-only MAVLink
  `COMMAND_LONG` command `31000` then rolled boot `1004917 -> 3333 ms`; new
  race start was `3312 ms`. The camera was non-black and Gate 1 was detected
  (`215` detections; last saved confidence `0.84164`). Recovery/setup SHA-256:
  `b68c238406338bba0f6296b597fcbb2827b4150edec0d834054132dfc6f5fcae`,
  `948e8b826b9b78eac4a4c751808f9dece2358d40e1eaa6133c4bb1199ae0f25b`.
- Passive `n159_six_gate_hybrid_shadow_022` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. Replay passed `304/304`, maximum action error
  `2.473499298161208e-7`, inference `30.306 Hz`, all `11/11` deployment hashes,
  and no-op cadence `598/10.016 = 59.605 Hz` with zero rate issue/setpoint.
  Official index remained `0`, finish `-1`; zero collision/dropout/drain hit;
  `2419` messages, `1246` IMU, `126` frames/detections; raw scope `3..3`.
  Shadow/parity SHA-256:
  `7fef15cd5f0552e86b9fc2a575d5ce4fe2f8822543af0a382524f38f9fc7e2d1`,
  `8904d09ab12c0d972e458e5c342048f996b0b6da304bc3e73c1db4457003908f`.
- Preregister exactly one bounded Candidate 020 under tag
  `n159_six_gate_hybrid_gate4_bounded_020`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop official index `4`. Accept only clean index `4`, finish
  `-1`, actual cadence `[50,100) Hz`, matching hashes, zero collision/invalid/
  dropout/drain/rate issue, one exit disarm and passive disarm proof. Reject
  failure without unchanged retry. FullLap remains unauthorized.

## N159 Candidate 020 Rejected Collision-Free at Gate 4

- Candidate 020 ran exactly once under
  `n159_six_gate_hybrid_gate4_bounded_020` and is rejected without retry. It
  reused PID `24476`; the readiness check left reset-ready VQ1/R1 untouched,
  and the runner's one MAVLink command `31000` reset was detected
  (`145647 -> 3406 ms`, race start `3314 ms`, detection `0.515 s`, calibration
  `60/60`).
- N159 passed official Gates 1--3 live and reached authoritative index `3`,
  finish `-1`. This proves the close-plane Gate-3 roll persistence. The run
  then stayed collision-free at Gate 4 for the full `45.0 s` bound and failed
  only the required index-4 progress; N158 therefore proved safe but not
  effective.
- Publication was `2641/44.906 = 58.789 Hz`, with zero command-rate violation,
  telemetry dropout, drain hit, or collision; `11005` messages, `5556` IMU,
  `527` frames, `188` detections, `339` no-quad frames, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `4000e4ee7dfe2d96c2fc09afba4240227b2cdaf0af3d9841c94c0ceb73c1d38e`,
  `10b9a744bf3e8e30f1bd2ef11e8243f6849820badc51803a9b1b411b0aee24dc`,
  `123326118595c95984337c9f16451b3fd6e24e9150c94583c02509c52b654380`,
  `9a7f5b90027fb21b27a048b377d0ad11fe51be993ce237bbcd0d310da58268f4`,
  and `0c9ee53e19cda74535f31890be493eecab8e1f689ef66c763ecef29f4bbd41f1`.
- Gate 4 began at `10.844 s`. Every visible raw pose failed the N158 initial
  lateral bound: right began `+9.469 m`, rose through `+19.001 m` observation
  saturation, and disappeared after `16.391 s`. The safe rejection action
  remained brake/level/hover but also held yaw near `-0.056 rad`, so the target
  moved off-axis and acquisition deadlocked. N160 should change only the
  rejected-visible Gate-4 yaw component: bounded yaw-only centering after the
  existing grace, while retaining brake pitch, zero roll, hover thrust, all
  association/anchor/descent gates, N159 Gates 1--3, and Gates 5--6. Prove on
  Candidates 016/018/020 before any reset or flight; no FullLap.

## N160 Gate-4 Rejected-Pose Yaw Acquisition — Offline Passed

- N160 changes only the yaw component of Gate 4's existing fail-safe after its
  `1.0 s` acquisition grace. If a pose is visible but rejected by the existing
  range/lateral/association rules, command a current-pose yaw step toward it,
  capped at `0.35 rad`. Retain normalized pitch `-0.24`, roll `0.0`, hover
  thrust `0.0`; detector dropout still holds current yaw. All anchor, reanchor,
  descent, N159 Gates 1--3, and Gates 5--6 behavior is unchanged.
- Candidate-016/018/020 replay passes with zero blockers. Yaw-only acquisition
  counts are `29/4/37`; maximum steps are `0.35/0.340845/0.35 rad`, all target
  directions are correct, and non-yaw violations are zero. All `9/7/240`
  dropout samples hold current yaw. Candidate 018's first centered anchor stays
  exactly `12.656 s`; Candidate 016 retains both reanchors; the complete N158
  anchor/descent regression passes.
- Policy/verifier/evidence/regression/wrapper SHA-256 values are
  `587e4b57f7fc4b28fabeaea2e189a81200835ebb645bb3f06465e885f9572953`,
  `2c1eee4a15088354a1dd32f70f8b60641c6d1fe0c38cdffb0f710d3410ee89ba`,
  `7ad9998a8dabc8bfe96f231ce6842abf2ee00f4d1199728a3f6ae34f76df6c4d`,
  `1d099acec8c05f3cc3fbe7da7a849af734d15f805edc7b912ffda684d09c8eec`,
  and `7c5e80a30ed7f78e5f4b22a0f806a972ebdf2d5161210213b3919dcfa2987901`.
  Linux deployment tests pass `86/86`; actual Windows tests pass `25/25`;
  Windows byte-compilation and both PowerShell parses pass.
- No N160 flight yet. Reuse PID `24476`; recover sole VQ1/R1 without relaunch,
  then issue exactly one separately logged reset-only MAVLink `COMMAND_LONG`
  command `31000` with detected boot/race rollback, index `0`, finish `-1`,
  and visible Gate 1. Next run only passive zero-command
  `n160_six_gate_hybrid_shadow_023`. No additional lifecycle command, flight,
  unchanged retry, or FullLap before shadow acceptance.

## N160 Reset Setup and Passive Shadow 023 — Passed; Candidate 021 Bounded

- Reused responsive simulator PID `24476` without relaunch. Same-PID recovery
  restored sole VQ1/R1, then exactly one reset-only MAVLink `COMMAND_LONG`
  command `31000` rolled simulator boot `642916 -> 3096 ms`; race start was
  `3299 ms`, base/status `193/4`, official index `0`, finish `-1`, with visible
  non-black Gate 1. Recovery/setup SHA-256 values are
  `a18fa1ccc6eeb2f916896d2223d5b60c55bbd98cb90424936cab0c5cc74ec339`
  and `25afeafd311b52c3248385c14b8fcf9eb514312dba1306dfaee4f500ad54d45f`.
- Passive `n160_six_gate_hybrid_shadow_023` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. Parity passed `285/285` visible samples with
  maximum action error `2.2688121795177985e-7`, inference `28.409 Hz`, and all
  11 deployment hashes matching. No-op cadence was `572/9.922 = 57.549 Hz`;
  collision, telemetry dropout, drain-hit, and rate-violation counts were zero.
  Official state remained index `0`, finish `-1`; telemetry/camera totals were
  `2426` messages, `1250` IMU, `140` completed frames, and `119` detections.
  Shadow/parity SHA-256 values are
  `d413d77bead96c6a4b42929a201d41c6cd64afc50c8490669c3791e48f2fca40`
  and `8f5c03dba529849e58a87a064a934c34dea966dfcb2d0327798e407e02174d7e`.
- Preregister exactly one bounded Candidate 021 under tag
  `n160_six_gate_hybrid_gate4_bounded_021`: use the current pinned wrapper and
  actual Windows Python, reuse PID `24476`, send one detected MAVLink command-
  `31000` reset, request `80 Hz`, cap at `45 s`, repeats/min-valid `1/1`, and
  target/minimum/authoritative stop official index `4`. Accept only clean index
  `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero collision/
  invalid/dropout/drain/rate issue, one exit disarm, and passive disarm proof.
  Reject failure without unchanged retry. FullLap remains unauthorized.

## N160 Candidate 021 Rejected at Gate 4 — N161 Coordinated Acquisition

- Candidate 021 ran exactly once and is rejected without retry. It reused PID
  `24476`; command-`31000` reset was detected (`281466 -> 3402 ms`, race start
  `3274 ms`, detection `0.516 s`, calibration `60/60`). It passed official
  Gates 1--3, then stopped at index `3`, finish `-1`, on Gate-4 contact ID
  `1002`, threat `1`, impact only `0.036701612174510956 m/s`.
- Runtime `22.0 s`; publication `1343/21.906 = 61.262 Hz`; zero rate violation,
  telemetry dropout, or drain hit; `5861` messages, `2956` IMU, `335` completed
  frames, `257` detections, one exit disarm. Aggregate/attempt/smoke/CSV hashes:
  `81d4a2eaf669c9d22e348577d735f8853e16a6959281df4b5ae33a112f653f12`,
  `2b2929392e9fbd055a43f35b21206c121d1e6dba0aa2c37aeecbaffb005ab57e`,
  `5ea58acf47b4643dab648ec9b5f25a05f46f4904f2e33149fd13024a67d4183f`,
  `923cc7b3d53002769ea58370d6d98707c4808cebf231ce4e51d7a25378aee1a7`.
  Passive post-stop sent no reset and proved base/status `65/3`, index `3`,
  finish `-1`; snapshot SHA-256 is
  `f9a5a9a663e49a4c0d5b9f1d3dec97d515abe3ac9b0b6aece4f95ab8e81ec4b6`.
- N160 successfully kept Gate 4 in view and first anchored at `14.422 s`, but
  pre-anchor rejection retained zero roll while the vehicle cut laterally into
  inside course geometry. Gate-4 entry velocity was approximately
  `[6.872,-2.767,-1.916] m/s`; one second later lateral speed was `-4.701 m/s`.
  The last visible target remained right/down at
  `[16.941,2.118,8.074] m`, yaw error `0.124 rad`, before dropout/contact.

## N161 Gate-4 Coordinated Initial Acquisition — Offline Passed

- N161 changes only rejected-visible Gate 4 after the existing `1.0 s` delay
  and only before the first coherent anchor: retain N160 brake pitch, hover
  thrust, and bounded yaw centering, while adding the already established
  lateral law `roll=-0.02*right`, capped at `0.10 rad`. After first anchor and
  on dropout, level-hover/yaw-hold and every association/reanchor/descent rule
  remain unchanged; Gates 1--3 and 5--6 are exact.
- Candidate-016/018/020/021 replay passes with zero blockers and exercises
  `16/2/37/23` coordinated samples. Maximum normalized roll is `0.2` in every
  trace; pitch/thrust, roll-law, yaw-law, and dropout violations are all zero.
  First anchors remain `13.629/12.656/none/14.422 s`; reanchor counts remain
  `2/0/0/1`.
- Policy/verifier/evidence/wrapper SHA-256 values are
  `0b8dbd20347972ae64e8753cdb79018e3def97278e7df25ac982ec04106ea098`,
  `dbc5f774a943e63e6b04b97dd43c6f10ec1f52ac98a50afa329ede8f5ef383f5`,
  `b5a69a3a0b55a77c4802b8e8b1fe527a2a912c619e30f78ffe9357d80712a920`,
  `8f6252c670e6a0096235e058f5038111b595c53a724af96ddb0b90f2a470dfb7`.
  Linux deployment tests pass `130/130`; actual Windows tests pass `32/32`;
  Windows byte-compilation and both PowerShell parsers pass.
- No N161 flight yet. Recover sole VQ1/R1 on PID `24476` without relaunch,
  issue one separately logged reset-only MAVLink `COMMAND_LONG` command
  `31000`, require boot rollback/index `0`/finish `-1`/visible Gate 1, then run
  only passive `n161_six_gate_hybrid_shadow_024`. No FullLap.

## N161 Reset Setup and Passive Shadow 024 — Passed; Candidate 022 Bounded

- Same PID `24476` recovered to sole VQ1/R1 without relaunch. One reset-only
  MAVLink `COMMAND_LONG` command `31000` rolled boot `1052102 -> 4381 ms`, race
  start `3298 ms`, base/status `193/4`, official index `0`, finish `-1`, with
  collision-free visible non-black Gate 1. Recovery/setup SHA-256 values:
  `2d4dda8c379b929e9a1dacca8d0fe90d4bc8277a1d21e3080a3355b2e47a3f63`,
  `7aa038c3f2541f27ffdc66db7ca3e8d3d0672f0e3943110b85090bfdccde63ab`.
- Passive `n161_six_gate_hybrid_shadow_024` issued zero reset/arm/setpoint/
  disarm. Replay passed `299/299`, maximum error
  `2.3133983612089182e-7`, inference `29.9 Hz`, all 11 hashes, and no-op
  cadence `597/9.985 = 59.69 Hz`. Collision/dropout/drain/rate counts were zero;
  index stayed `0`, finish `-1`; `2416` messages, `1245` IMU, `141` completed
  frames, `125` detections. Shadow/parity hashes:
  `14ee0ebb15482960aa8340406af0944d9f9dcb7724543e07b18d11a5befa9df0`,
  `8e2f9b1e81f0c43fe010cb545e7bcbf9c46600687a6be2146bf89f8d30380143`.
- Preregister exactly one `n161_six_gate_hybrid_gate4_bounded_022`: current
  pinned wrapper, actual Windows Python, reused PID `24476`, one detected
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, target/minimum/authoritative stop official index `4`. Accept only clean
  index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero collision/
  invalid/dropout/drain/rate issue, one exit disarm and passive proof. Reject
  failure without unchanged retry. FullLap remains unauthorized.

## N161 Candidate 022 Rejected at Gate 4 — Roll Mechanism Removed

- Candidate 022 ran once and is rejected without retry. Reused PID `24476`;
  detected command-`31000` reset `149965 -> 3415 ms`, race start `3341 ms`,
  detection `0.500 s`, calibration `60/60`. It passed official Gates 1--3,
  then collided at index `3`, finish `-1`: ID `1002`, threat `2`, impact
  `8.278605461120605 m/s`.
- Runtime `19.484 s`; publication `1156/19.390 = 59.567 Hz`; zero rate/
  dropout/drain issue; `5278` messages, `2664` IMU, `301` completed frames,
  `237` detections, one exit disarm. Aggregate/attempt/smoke/CSV/passive hashes:
  `37d3a8d817bb01ccd0d260d42ac0b279e24728a8d4e8cc3ec619f279a5e09986`,
  `8e4ddbbfea05a8f6c0893b670c7cce695aa134193603a519d616c78f96ac669d`,
  `dcea9e3bbd68c6bb8ba2a0bbea69d91e1f34c7610ea5c879d9b98b89fffd6763`,
  `5595b2e6311835e2662d303ee51f20b2c36915bcf53b3b1c80bd370bac966019`,
  `f3fd4e6a834c786df6a982d2d71333c79db05644f0cac25b0eae62db29da1be8`.
  Passive post-stop: no reset, base/status `65/3`, index `3`, finish `-1`.
- N161 is falsified and removed. Before initial anchor it saturated roll at
  `-0.10 rad` against the positive family, then flipped to `+0.10 rad` on
  alternating aliases; estimated lateral position moved from `-7.504 m` at
  `11.203 s` to `+55.082 m` at collision. Do not reuse N161 roll.

## N162 Gate-4 Bounded Dropout Yaw Carry — Offline Passed

- N162 restores N160's zero-roll rejected acquisition exactly. It changes only
  a Gate-4 detector dropout: for at most `0.5 s`, retain the last visible yaw
  target, re-clamped to at most `0.35 rad` from current yaw; keep brake pitch,
  zero roll, and hover thrust. On expiry, resume current-yaw hold. Association,
  anchor/reanchor/descent, Gates 1--3, and Gates 5--6 are unchanged.
- Candidate-016/018/020/021/022 replay passes. Dropout carry/expired-hold counts
  are `4/5`, `4/3`, `0/240`, `3/10`, `0/0`; maximum carried yaw deltas are
  `0.35/0.021800/0/0.094046/0 rad`. All dropout non-yaw/yaw-law and pre-anchor
  roll violations are zero. First anchors remain
  `13.629/12.656/none/14.422/14.406 s`; reanchors remain `2/0/0/1/0`.
- Policy/verifier/evidence/wrapper SHA-256 values:
  `052e1869da50d359af526ed90d02e2ad30fdf9fb523bd737311b497764776b68`,
  `c95a12c0de71b09385fa1f59d38bad4fc8061f3148882029475ed8dda82be16e`,
  `209cffdcce091262f4ce682a66979a146cc20defa6277ae291d563dbbf724d03`,
  `16d7c79be471fca0cf2f574e13147e614e02f76973a5d4c7950998386707c78a`.
  Linux deployment tests pass `131/131`; actual Windows tests pass `33/33`;
  Windows byte-compilation and both PowerShell parsers pass.
- No N162 flight yet. Recover sole VQ1/R1 on PID `24476` without relaunch,
  send one reset-only MAVLink command `31000` with detected rollback/index `0`/
  finish `-1`/visible Gate 1, then passive
  `n162_six_gate_hybrid_shadow_025` only. No FullLap.

## N157 Candidate 018 Rejected at Gate 4 — N158 Anchor/Descent Timing

- Candidate 018 ran exactly once and is rejected without retry. It reused PID
  `24476`; MAVLink command `31000` reset was detected (`280952 -> 3393 ms`,
  race start `3306 ms`, detection `0.516 s`, calibration `60/60`). N157 passed
  official Gates 1--3, proving the Gate-2 latch live, then collided at Gate 4:
  official index `3`, finish `-1`, ID `1002`, threat `2`, impact only
  `1.7557381391525269 m/s`.
- Transport was healthy: `836/15.734 = 53.07 Hz`, runtime `16.047 s`, zero
  command-rate violation, dropout, or drain hit; `4492` messages, `2270` IMU,
  `168` frames, `157` detections, one exit disarm. Aggregate/per-attempt/smoke/
  CSV/passive SHA-256 values are
  `1b433638ab9b25d74bfe3fb94d8ab7d27696b52098a0f8767a7df9bc9fe4b602`,
  `bbbb0a6adbb480374f7dce6523607d07a8c99231f046c1e4472e6c8a05c0fa84`,
  `ed860bba7bb4cca7c27f8abd58f83af9601f3f636145d6c62ae2b35965e8b3d9`,
  `ea8449c4197787881e6fa4e1bca985268f55e9b845a47b04aa47cd5d39c689e6`,
  and `1b8848ff297685bfb66d1dae1e4632f5a251008753cfc5323986fdafc6e64f87`.
  Passive post-stop state is base `65`, status `3`, index `3`, finish `-1`.
- Gate-4 replay shows the one-second unconditional grace anchored false raw
  poses near `[30.0,10.875,12.188]` and `[26.667,11.458,12.042] m`. The first
  coherent centered family was `[25.412,0.556,13.818] m`, but the stale anchor
  rejected it; vertical descent began only at `13.797 s`. The final visible
  pre-impact vector `[26.667,-0.458,8.125] m` was also rejected solely by
  `12 m` stale-anchor drift. N158 replay advances descent by `1.141 s`.
  Replay report SHA-256 is
  `70e33d9064fa526d1fd1553a20b3d680df09a17a01fa08855fc36e12da39bacc`.
- N158 should change only Gate-4 association/timing: initial acquisition must
  satisfy the already retained `4.5 m` lateral bound, and vertical correction
  may start after the `1.0 s` association grace instead of waiting `3.0 s`.
  Keep the `3.0 s` close-reanchor delay, all gains/bounds, N157 Gates 1--3,
  and Gates 5--6 exact. No N158 flight or FullLap before two-candidate replay,
  tests, reset proof, and passive shadow.

## N158 Gate-4 Initial Anchor and Vertical Timing — Offline Passed

- N158 preserves N157 Gates 1--3, Gate-4 gains/bounds, the `3.0 s` close-
  reanchor delay, and Gates 5--6. Only Gate 4 changes: an initial anchor must
  have `abs(right)<=4.5 m`, and vertical control begins after the existing
  `1.0 s` association grace instead of after `3.0 s`.
- Two-candidate replay passes with zero blockers. Candidate 016 keeps `80`
  Gate-4 samples (`71` visible, `9` dropout), both reanchors, level-hover
  dropout behavior, normalized roll cap `0.2`, and decoded thrust floor `0.22`;
  `23` false initial samples fail safe before the first centered association at
  `13.629 s`. Candidate 018 rejects `8` false initial samples, first associates
  and commands `0.22` descent at `12.656 s` on
  `[25.412,0.556,13.818] m` (`1.141 s` earlier), and keeps the final visible
  `[26.667,-0.458,8.125] m` sample associated/descent instead of hover.
- Policy/verifier/evidence/wrapper SHA-256 values are
  `de3c3069476fdcf56789a8a822f1f4ee03edecfa3410b0caaa1bce9c27933fa6`,
  `0bf4380646da5d8a189426cc022626061336e8ab24d48270bf6860bf5c3cdc16`,
  `67f9190d2231fed172f08d97a96d27dadb9f20e03d06aa67123bc5b2d41d64ca`,
  and `73ec596346e3f0b3b816f7a6bda2647068dee11cabf9a5e1d5f55607c8604cfa`.
  Linux deployment tests pass `68/68`; actual Windows targeted tests pass
  `21/21`; Windows compilation and both PowerShell parses pass.
- No N158 flight yet. Reuse PID `24476`; send exactly one separately logged
  reset-only MAVLink command `31000` after same-PID VQ1/R1 recovery, require
  detected boot/race rollback and visible Gate 1, then run passive zero-command
  `n158_six_gate_hybrid_shadow_021` only. No FullLap.

## N158 Reset-31000 Setup — Passed

- Same responsive PID `24476` recovered to VQ1/R1 and was reused. Exactly one
  reset-only MAVLink command `31000` rolled boot `1006213 -> 2443 ms` during
  countdown; race start `3317 ms`. Later passive status proved base `193`,
  status `4`, official index `0`, finish `-1`; Gate 1 was non-black and visible
  (`74` detections, first confidence `0.84875`). No relaunch or flight setpoint.
- Setup/status/image SHA-256 values are
  `cf1dbc53f9b500a3b0dea6628c133aad737efe873973c7088a0c3c23a2a6051a`,
  `c25f1c485bf9d7d33702dd17854df7b0b1dc812864cdfb776d1469a2efffc6ee`,
  and `b431d19d38eee722c47e4953e4ae106195fa210b47e0a31227635b3ead61dffa`.
  Next run only passive `n158_six_gate_hybrid_shadow_021`; no additional reset,
  lifecycle command, flight, or FullLap.

## N158 Passive Shadow 021 — Passed; Candidate 019 Bounded

- Passive `n158_six_gate_hybrid_shadow_021` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. All `308/308` visible samples replayed within
  maximum error `1.7313575745303567e-7`; inference was `30.8 Hz`, all 11
  deployment hashes matched, and no-op cadence was
  `603/9.984 = 60.296 Hz` with zero violation/setpoint. Official index stayed
  `0`, finish `-1`; collision/dropout/drain counts were zero. Telemetry/camera
  totals were `2476` messages, `1275` IMU, `124` frames, `121` detections; raw
  scope remained Gate 4 only (`3..3`). Shadow/parity SHA-256 values are
  `ca90056ecc823434aebdd186e3832177f2202a0a66df2cf8c31fbfc96a69b387`
  and `756bb1dd40973c8218d39b59d1a5218c4d02dca27b4333aa3f78d3a0ec27c7c3`.
- Preregister exactly one bounded Candidate 019 under tag
  `n158_six_gate_hybrid_gate4_bounded_019`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected command-`31000` reset,
  `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/authoritative
  stop index `4`. Accept only clean index `4`, finish `-1`, cadence `[50,100)`
  Hz, hashes, zero collision/invalid/dropout/drain/rate issue, one exit disarm
  plus passive proof. Reject failure without retry. FullLap unauthorized.
- Lifecycle waiter now requires active race status plus non-null heartbeat
  base/status fields before returning, preventing a race-status-first packet
  from causing unnecessary UI recovery. SHA-256 is
  `9c23a70544d0216b66bb3c6bf285b179be27dc1864fe966805d75bd881ac9cec`;
  focused Linux `8/8` and actual Windows `3/3` tests pass.

## N156 Candidate 017 Rejected at Gate 2

- Candidate 017 ran exactly once and is rejected; N156 Gate-4 control was not
  reached or evaluated. It reused PID `24476`; command `31000` reset was
  detected (`342878 -> 3330 ms`, race start `3302 ms`, detection `0.546 s`,
  calibration `60/60`). It passed only official Gate 1 and collided at active
  index `1`, finish `-1`: ID `1001`, threat `2`, impact
  `10.29787540435791 m/s`. The smoke artifact's `ordered_gate_passes=2` is a
  vision diagnostic; authoritative official progress is index `1`.
- Transport was healthy: `454/7.469 = 60.651 Hz`, duration `7.766 s`, zero
  command-rate violation, telemetry dropout, or drain hit; `2655` messages,
  `1340` IMU samples, `97` detector frames, `91` detections. Aggregate/per-
  attempt/smoke/CSV hashes are
  `3521cae3eda65ed6c0f900babdb541c96c493764e11210d9d13db132da28aea6`,
  `c7910fbda07d57eee2fde9b847501507211d428e350e915f97504a041b3f9f8c`,
  `473c97741465c743de814f1c3d07e789c6aa3664a0de906913fc56b423278417`,
  `e6de6799e6010c0cd9fb2d1b9cf50c50b26b3227300c701904ea89aebed8a010`.
  Passive post-stop base `65`, status `3`, index `1`, finish `-1`; snapshot
  SHA-256 is
  `a8e496d22d600b603303919cbc78399d10a25a12978d7a5e632f4139c23caf65`.
- N155's projected Gate-2 governor first activated only at `7.437 s`: the prior
  `4.916 m` sample projected `+0.331 m`, below its `0.5 m` threshold; one frame
  later the estimate jumped to `+0.757 m`. It remained outside at `+0.708 m`
  with `0.400 rad` image bearing near the collision. Candidate 016's clean
  Gate-2 pass ended near `+0.418 m`. N157 should change only Gate-2 admission:
  lower the symmetric projected threshold to `0.3 m` and latch braking/PD
  recentering until official Gate 2 advances, preventing one-frame release and
  starting one sample earlier in both archived trajectories. Preserve N156
  Gate 4 and every other output. No N157 flight or FullLap before offline proof
  and a new passive shadow.

## N157 Gate-2 Latched Aperture Governor — Offline Passed

- N157 changes only official Gate 2: the projected plane-miss admission is
  `0.3 m` instead of `0.5 m`, and the existing brake/roll-PD governor remains
  latched until the official gate index advances. Existing brake pitch
  `-0.2`, roll `kp=0.6`, `kd=0.35`, limit `0.7`, thrust, yaw, Gates 1 and 3--6,
  and the complete N156 Gate-4 safe handoff are unchanged.
- Replay passes with zero blockers. It activates `0.125 s` earlier on rejected
  Candidate 017 (`4.916 m`, projected `+0.331 m`) and `0.265 s` earlier on the
  clean Candidate-016 Gate-2 path (`5.759 m`, projected `+0.492 m`), retains
  the latch through that clean path's next `+0.284 m` frame, preserves thrust
  and yaw exactly, and remains inside the existing roll bound.
- Policy/verifier/evidence/wrapper SHA-256 values are
  `743f1ae9fc656999ef1f7ef4450017850f724754924e4deec4913289d3c94e38`,
  `44903c877a6d2e658ef5be5316917f041083d6ad2362f7ad4da620016dd371fa`,
  `8a49fe26bcdb91d0056faeffbe261fbc960cc10742a50c391ac596ca5a19cdc6`,
  and `6cc7977c62e002514ec229250916139627d2cf4b371a9bd9b3481e2f5d7b0f8f`.
  Linux deployment tests pass `67/67`; actual Windows targeted tests pass
  `20/20`; Windows compilation and both PowerShell parses pass.
- N157 recovery confirmed the preserved PID reached the VQ1 event list, but
  the old event/race Y values targeted edges rather than control centers. The
  measured centers are event `277` and race `872`; startup helper SHA-256 is
  `d9e97ff6ac38c7a476eebc5ddfa5fd6f47d705c11f613922c17e9690f9031f7d`.
  Manual center clicks restored base `193`, status `4`, index `0`, finish
  `-1` on PID `24476`; no reset or flight occurred during recovery.
- Reuse responsive simulator PID `24476`; do not relaunch. Next authorize only
  one separately logged reset-only MAVLink `COMMAND_LONG` command `31000`.
  Require boot-time rollback, a new race epoch, reset-ready state, official
  index `0`, finish `-1`, and visible Gate 1. Then run passive zero-command
  `n157_six_gate_hybrid_shadow_020` only. No N157 flight or FullLap yet.

## N157 Reset-31000 Setup — Passed

- Same responsive simulator PID `24476` was reused. Exactly one reset-only
  MAVLink `COMMAND_LONG` command `31000` was sent after reset-ready recovery;
  the simulator was not relaunched and no flight setpoint was sent.
- Telemetry proves the reset: pre-reset boot `1112445 ms` rolled to the new
  epoch (`25848 ms` at the status sample), new race start `3292 ms`, base
  `193`, status `4`, official index `0`, finish `-1`. The camera was non-black
  and detected Gate 1 (`48` detections; first confidence `0.85375`).
- Setup/race-status/Gate-1-image SHA-256 values are
  `c73c6f465aa8e35f405a1dc4da9a867e88483b619efd7fc9701cd4e42c1b25c5`,
  `15def8bc55a264bfef2d345fccb9de73dba62222c6bffc2ba9f868a77d515e77`,
  and `ee72e40319600462dcb6f3b2a3e81722eaa7bd5287294a742051ebca25d95636`.
  Next run passive zero-command `n157_six_gate_hybrid_shadow_020` only. No
  additional reset, arm, setpoint, disarm, flight, or FullLap.

## N157 Passive Shadow 020 — Passed; Candidate 018 Bounded

- Passive `n157_six_gate_hybrid_shadow_020` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. All `132/132` visible samples replayed with
  maximum action error `2.4103546142351107e-7`; inference was `13.2 Hz`, all
  11 deployment hashes matched, and no-op publication was `555/10 = 55.4 Hz`
  with zero rate violation or setpoint. Official index stayed `0`, finish
  `-1`; collision/dropout/drain counts were zero. Telemetry/camera totals were
  `2435` messages, `1254` IMU, `100` frames/detections. Raw observation scope
  remained Gate 4 only (`3..3`). Shadow/parity SHA-256 values are
  `9b97f8b03db72d4bd34444b5f19bdc1bc92f0af6b6c7131017a7123e9f0abc29`
  and `1f397060c618c6a1d6b5a010c384b42d7e3f5fbf9a75d849e0ef0198d36a17ed`.
- Preregister exactly one bounded Candidate 018 under tag
  `n157_six_gate_hybrid_gate4_bounded_018`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one detected MAVLink command-`31000`
  reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/
  minimum/authoritative stop official index `4`. Accept only clean index `4`,
  finish `-1`, actual cadence `[50,100) Hz`, matching hashes, zero collision/
  invalid/dropout/drain/rate issue, one exit disarm and passive disarm proof.
  Reject failure without unchanged retry. FullLap remains unauthorized.

## Final Current Status — N167 Awaiting Shadow 030

Supersede all earlier next-action notes. Candidate 027 ran once, passed only
official Gates 1--2, and was rejected at Gate 3/index `2` after collision
`1001`, impact `6.9826836585998535`, finish `-1`. Its reset was MAVLink
`COMMAND_LONG` command `31000`, proved by boot rollback `126297 -> 3389 ms`.
N167 restores recurrent Gate-3 pitch while retaining terminal roll level and
the vertical thrust floor; wrapper SHA-256 begins `2c3805f9`. Next only:
same-PID reset-ready recovery, exactly one reset-only `31000` with measured
rollback, then passive zero-command `n167_six_gate_hybrid_shadow_030`.
Candidate 028 and FullLap are unauthorized until Shadow 030 passes.

## N167 Recovery, Reset Setup, and Shadow 030 Passed — Candidate 028 Bounded

- The responsive simulator PID `24476` was recovered through the UI without
  relaunch or MAVLink control. It returned to base/status `193/4`, official
  index `0`, finish `-1`; recovery SHA-256 is
  `643e1a8c377ea29071dcc20ed7766c51131bb25122d651402abf5db52d0ebf25`.
- Exactly one separately logged reset-only MAVLink `COMMAND_LONG` command
  `31000` rolled the recovery boot epoch `1527982 -> 4338 ms`, created race
  start `3309 ms`, left index `0`/finish `-1`, produced zero collision, and
  detected Gate 1 on all `251` sampled frames. Setup SHA-256 is
  `51f981ea99371b392097fe2e73ce3ce2372c6023c4a7210c6a21524032fe88fb`.
- Passive `n167_six_gate_hybrid_shadow_030` sent zero reset, arm, MAVLink
  setpoint, and disarm commands. Parity passed `307/307`, maximum action error
  `4.644851684387774e-7`, inference `30.7 Hz`, and all 11 deployment hashes.
  No-op cadence was `605/9.969 = 60.588 Hz`, with zero violation/setpoint.
  Official index stayed `0`, finish `-1`; collision/dropout/drain counts were
  zero, with `2435` messages, `1254` IMU samples, and `122` frames/detections.
  Shadow/parity SHA-256 values are
  `ae86706ee7258effe10a4f1fff300b06d16165144d03b02972525fea3dfb966f`
  and `74cc002c8d0738c2f8f80b26b66ef0619c11a0a8361f3d8eca37df23528aebdf`.
- Authorize exactly one bounded Candidate 028 under tag
  `n167_six_gate_hybrid_gate4_bounded_028`: current pinned wrapper, actual
  Windows Python, reused PID `24476`, one normal detected command-`31000`
  policy-ready reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, target/minimum/authoritative stop official index `4`. Accept only a
  clean index-4 stop with finish `-1`, cadence `[50,100) Hz`, matching hashes,
  zero collision/invalid/dropout/drain/rate issue, one exit disarm, and passive
  disarm proof. Reject failure without unchanged retry. FullLap remains
  unauthorized.

## Candidate 028 Rejected for Collision; Delayed Official Gate-3 Pass Proven

- Candidate 028 ran exactly once under
  `n167_six_gate_hybrid_gate4_bounded_028` and is rejected without unchanged
  retry. Its normal `COMMAND_LONG`/`31000` reset was detected with boot
  `188257 -> 3422 ms`, race start `3295 ms`, detection `0.500 s`, and
  calibration `60/60`. It passed official Gates 1 and 2 before collision
  `1001`, threat `2`, impact `4.625654220581055 m/s` aborted the controller.
- The attempt artifact ended on stale race status index `2`, but the required
  command-free passive snapshot subsequently received authoritative official
  index `3`, last-gate race time `10.754043579 s`, finish `-1`, base/status
  `65/3`, and `reset_sent=false`. Therefore N167 did pass Gate 3 before/while
  striking the frame; this is not another Gate-3 miss, nor a clean milestone.
  Gate 4 control never activated because collision abort preceded the delayed
  race-status transition.
- Delivery was `604/10.297 = 58.561 Hz`, duration `10.625 s`, zero command-rate
  violation, telemetry dropout, or drain hit; `3311` messages, `1673` IMU,
  `130` frames/detections, and one exit disarm. Aggregate/attempt/smoke/CSV/
  passive SHA-256 values are
  `6eaf8c40deaf0f52981d07d338a375934e333296b01583269e13d1e49996eb2e`,
  `78f1be2c82d1292f9d8552f0a9f32a80c5b9bc8975cdd363c7af742d0c6a2cf3`,
  `a4d4d7dfd37209fa4a410c54c59c4793d074c8fba832cdd1e63fa7dcf8a70268`,
  `5c6cd4fcbce3e88adc2109f92f528050f37f6e45d2b09580391c05d90d2da90e`,
  and `e5abeada7ff2a60ff44efc38d52e49857577b861e2de0180aa37f5dbd784cffc`.

## N168 Gate-3 Extended Counter Suppression — Offline Passed

- Candidate 028 commanded full negative roll while Gate 3 remained left at
  `6.957` and `6.447 m`, delaying positive recovery until about `5.68 m`; its
  observed roll was only about `0.04 rad` near crossing. N168 extends only the
  left-side counter-bank suppression from `4` to `7 m`, flooring negative roll
  at normalized `0.0`. The adaptive full-positive boost remains bounded to
  `4 m`; pitch, thrust, yaw, terminal level/floor, Gate 2, and Gate 4 remain
  unchanged.
- Source-hashed replay changes zero samples on all five archived clean Gate-3
  pass paths (`016/018/020/021/022`). It changes exactly Candidate 028's two
  measured counter commands from `-1.0` to `0.0`, plus one exercised sample
  each in rejected `019/024/026`; non-roll error and floor violations are zero.
  Candidate 028's command-free official index-`3` snapshot is embedded in the
  evidence.
- Policy/verifier/evidence/pitch/vertical/terminal/Gate-2/Gate-4/wrapper hashes:
  `7a3783cdf65800c6ea4ae98f070c95a3c96c7c3f27b827f6142ad779862506b5`,
  `a6f5cf6d0c1e28a2419b2f48faa356e54305f2a5e9138bec737b3b9c27838cee`,
  `40ae0bd2e9a7f6de4302364c85f815bff0614eb4468c8d5d2636ccca42ce6e35`,
  `bdb97fa67ed879a8efcfe823012add794f30594a4c157d0ed9db13500aa4b5d5`,
  `77b21226b0f7b5a53b0f57da1c8687e79945fb63709381f56af1df9c21369a74`,
  `4804e7d0d7118ba00e41e05914f19e617bc48241d2c132b73c5b74822987beca`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `4596e55191cb70998dc584005b5c3f970cc61b7ed61ce42796773d06b5e22838`,
  and `c75d9eb1a1255f89a0bac19232e28cb108559da6c8a09ef15fe85226d12c972a`.
  Linux/actual-Windows suites pass `63/63`; compilation and both parsers pass.
- Do not fly N168 yet. Recover PID `24476` without relaunch, issue one
  separately logged reset-only command `31000` with measured boot/race
  rollback and visible Gate 1, then run only passive zero-command
  `n168_six_gate_hybrid_shadow_031`. Candidate 029 and FullLap remain
  unauthorized until Shadow 031 passes.

## N168 Recovery, Reset Setup, and Shadow 031 Passed — Candidate 029 Bounded

- Same responsive PID `24476` recovered without relaunch/control to
  base/status `193/4`, index `0`, finish `-1`; recovery SHA-256
  `5e27125029535437aa15b2d628dd847ca82ec0e2a512a74e3637c9a55fe3e70f`.
  One reset-only command `31000` rolled boot `873911 -> 4307 ms`, race start
  `3300 ms`, zero collision, and `264` Gate-1 detections; setup SHA-256
  `f082f0b0d134a735db2114767dd45e403e9c1c12d8140d6b55146d015d17906c`.
- Shadow 031 sent zero reset/arm/setpoint/disarm. Parity passed `281/281`, max
  error `2.5453186036639153e-7`, inference `28.1 Hz`, all 11 hashes; no-op
  cadence `571/9.985 = 57.086 Hz`, zero violation/setpoint. Index stayed `0`,
  finish `-1`, zero collision/dropout/drain issue; `2427` messages, `1250`
  IMU, `119` frames/detections. Shadow/parity hashes:
  `6a1957ade87b45e1a1827c8f79d6e2591f7679ceabb62630dd8026bfaca64cc1`,
  `22b0551ae2866e27d8899210d3ebbdc34013aedef5a86b40a5c9721d0e0d6d9c`.
- Authorize exactly one `n168_six_gate_hybrid_gate4_bounded_029`: pinned
  wrapper/actual Windows Python/reused PID, one detected normal `31000` reset,
  `80 Hz`, maximum `45 s`, `1/1`, target/minimum/stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching
  hashes, zero collision/invalid/dropout/drain/rate issue, exit disarm plus
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.


## Candidate 041 Rejected at Gate 3; Fixed Full-Scale Brake Branch Retired

- Candidate 041 ran exactly once and is rejected without unchanged retry. Its
  detected command-`31000` reset rolled boot `115465 -> 3317 ms`, published a
  fresh race start at `3297 ms`, was detected in `0.484 s`, and completed
  calibration `60/60`. It passed official Gates 1 and 2 at `7.394020557 s`,
  then collision `1001`, threat `2`, impact `9.259342193603516 m/s` stopped
  control at official index `2`, finish `-1`.
- Delivery was healthy: `641/10.500 = 60.952 Hz`, duration `10.610 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `3286`
  MAVLink messages, `1658` IMU samples, `135` frames and `134` detections,
  plus one exit disarm. The command-free post-stop snapshot was race-inactive
  at base/status `65/3`, race start `-1`, finish `-1`, and `reset_sent=false`;
  no delayed Gate-3 credit appeared.
- N180 closed the intended lateral gap: full `-1.0` roll remained continuous
  at `5.237`, `4.571`, `3.345`, and `2.683 m`, including current right
  `+0.050 m` at the final sample. Terminal lateral projection was already
  centered (`-0.183`, `+0.020`, `+0.202`, `+0.235 m`), yet the collision
  still occurred. The closest clean comparison, Candidate 021, passed Gate 3
  while terminal leveling held roll `0` from `3.072 m` through `1.77 m`;
  its terminal right residual remained safely left at about `-0.33 m`.
  Therefore the missing one-sample continuity was not the root cause, and
  extending a fixed full-scale counter-bank further is rejected.
- Aggregate/attempt/smoke/CSV/passive/diagnosis SHA-256 values are
  `5b524071d3a1430b0b15d1491497e132022b9643d483750cf0dbe7fbcc331bb1`,
  `cd4502b968ca33500f6911ba69cdfc2705a83ac03f011796d48c1d6130742665`,
  `b90377055819ae0415a289fbe23d8ea544d5bac82ec362db26aac1406b273fcc`,
  `e67413f701eac3a881d6f6608a48d7ed0956768c90a0ae93c6b06f1e64202945`,
  `8c339d96f7a885f2fe4c8fc50b72d054ef2059ebd85c5caa0e03fee0d144f125`,
  and `170c9e771d31ed55123e36af6c52de95d691c1137fd4c80c81fbc6f52ad3d577`.
- Do not reset or fly again. Candidate 042 and FullLap remain unauthorized.
  Next only: offline design/source-hash N181 by replacing the fixed `-1.0`
  terminal rate brake with a continuous, physics-based stopping command. Use
  the observable right residual and rightward rate to estimate the lateral
  deceleration required to stop before center, map that acceleration through
  `atan(a/g)` into normalized roll, and retain the existing N179 activation
  envelopes. Bound the command in `[-1,0]`, preserve all non-roll channels and
  protected positive roll, prove all five clean Gate-3 passes unchanged, and
  show finite/moderate commands on Candidates 038/039/041 before any new
  passive shadow. Do not revive another fixed-threshold/full-scale branch.



## Candidate 040 Rejected at Gate 3; Cross-Center Brake Gap Isolated

- Candidate 040 ran exactly once and is rejected without unchanged retry. Its
  detected command-`31000` reset rolled boot `119237 -> 3394 ms`, published a
  fresh race start at `3303 ms`, was detected in `0.516 s`, and completed
  calibration. It passed official Gates 1 and 2 at `7.329575061 s`, then
  collision `1001`, threat `2`, impact `10.494649887084961 m/s` stopped
  control at official index `2`, finish `-1`.
- Delivery was healthy: `643/10.313 = 62.252 Hz`, duration `10.407 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `3276`
  MAVLink messages, `1654` IMU samples, `133` frames and `131` detections,
  plus one exit disarm. The command-free post-stop snapshot retained official
  index `2`, Gate-2 time `7.329575061 s`, race start `3303 ms`, finish `-1`,
  base/status `65/3`, and `reset_sent=false`; no delayed Gate-3 credit appeared.
- N179's separate high-rate tier did not activate in this run. At `6.358`,
  `5.783`, and `5.133 m`, rightward rate was only `1.568`, `1.638`, and
  `1.739 m/s`. The ordinary brake activated at `3.692 m` with current right
  residual `-0.145 m`, rate `2.006 m/s`, projection `+0.280 m`, and output
  `-1.0`. One sample later at `3.254 m`, the residual crossed only `0.040 m`
  right while rate remained `2.045 m/s` and projection `+0.365 m`; the strict
  `right_m < 0` guard ended the brake and the latched terminal level output
  `0`. The positive-residual guard did not engage until `1.655 m`, where
  residual/projection were already `+0.678 m`; its `-1.0` command was too late.
- Aggregate/attempt/smoke/CSV/passive/diagnosis SHA-256 values are
  `2642ca48bdc8e22f0c18856019b0a539d723acd172a74c85ed3e7ca8222bdc35`,
  `b9f1e214256da069c2f1bf99b02cbb14848bcd047e11f28eaf7b1297ef78ae82`,
  `d73ffbb2399a9a23dbb78758f8f0a6b45b6be1fd1c3257ed7649d58e0cdda039`,
  `2859720dd4f281c2fb11ca2174c3259c5e56b06bfbd13c124062a03d68669c06`,
  `49f4638f4c6641f6934cdb53804d2c4f1c618fcdfca940810411c93230596d71`,
  and `9d6c140c6155ed1121fba121b97fe0de7a17b0c236d8d2eb4b4021105fa929a5`.
- Do not reset or fly again. Next only: build/source-hash N180 by changing the
  existing Gate-3 rate brake's current-right condition from strictly negative
  to at most `+0.10 m`, preserving both N179 rate/projection/range tiers and
  every action-eligibility condition. Prove zero changes on all five clean
  Gate-3 passes, exactly the single `3.254 m` Candidate-040 continuity sample,
  preservation of every other law, then require a fresh passive shadow.
  Candidate 041 and FullLap remain unauthorized.


## N180 Cross-Center Gate-3 Brake Continuity — Offline Passed

- N180 changes only the existing Gate-3 rate brake's current-right bound from
  strictly negative to at most `+0.10 m`. Both N179 ordinary/high-rate range,
  rate, and terminal-projection tiers remain unchanged, as do visibility,
  closing-speed, forward-minimum, and action-eligibility requirements. The
  rule still touches roll only and protects every positive learned roll except
  the exact `+0.3` margin floor.
- Replay changes zero samples in all five archived clean Gate-3 passes
  (Candidates 016/018/020/021/022). Candidate 040 changes exactly its one
  preregistered cross-center sample at `3.254 m`: current right `+0.040 m`,
  rightward rate `2.045 m/s`, terminal projection `+0.365 m`, and roll
  `0 -> -1.0`. Failed Candidate 031 exposes one additional instance of the
  same gap at current right `+0.075 m`, rate `2.454 m/s`, projection
  `+0.195 m`; its expected target count becomes three. Failed Candidates
  030/034/037/038/039 retain expected target counts 3/2/2/2/3. No non-roll
  channel or protected positive roll changes, and the command-free
  Candidate-040 post-stop proof is embedded.
- Every retained Gate-1/Gate-2/Gate-3/Gate-4 preservation replay passes.
  Policy/verifier/evidence/Gate-1/Gate-2-early/Gate-2-late/Gate-2-stateless/
  Gate-3-early-vertical/Gate-3-counter/Gate-3-positive-residual/Gate-3-pitch/
  Gate-3-margin/Gate-3-terminal/Gate-4/wrapper SHA-256 values are
  `fea39e91787f32712e3d5313ce42c8d9edf383addb3db0b32c757e1ac44b5bdf`,
  `4df7097b1465883646b860b127f7b349f44d4f0d766c3fc83f42056718c9e0a7`,
  `302b7b9a353d743385600cf467f4f7b76a6a38eb60d4f5aa236518638252f8b5`,
  `3c070733fa4079065c74e9c643afbcbf8e4552a19e1d34670adf05de223cae4d`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `73dc320a6b39abdb534a51e944f913025a544baf77d5a268c76e6b2ba8373389`,
  `5481494f82b9e6a7a0d396765f8f6194ff29bf00936ccc2c27a10bdb07cc1126`,
  `3b0e8a136caedf44d3ab73dde06406e8232851e4d3c9a12b1f10d3ab6408ec65`,
  `fff2ad2d70e14e521bafa25b9c8ddba93c47a493fe81f52877f888a009f7eb46`,
  `012ffb5322be33efd66439ebaed7e2b09ac6cb5cc1cf9488913442168c3a819f`,
  `5b9ac139576ecea7eb585597bb98da78f1e56050977cc8f115f7057ceee9de17`,
  `c8286475c1971adc566c2fb664d8fe057c5b51b188e27a01e6b579efec0113ee`,
  `7bac794800dc2f29730da95a36a70ac4791ed1d93c1400a5a530c29dc4a56894`,
  `fe58a6b6a9c2963cb7d988551e60ab62749df04170ae6bfba0767806bf0c7606`,
  and `03e785c5c13e82b5cffafebb4996bc9aeceafcc497aabb5426c67dba34d3d169`.
  Linux and actual-Windows targeted suites pass `48/48`; compilation and both
  PowerShell parsers pass.
- No reset, setpoint, or new flight was issued during N180 offline validation.
  Candidate 041 and FullLap remain unauthorized. Next only: on the same
  responsive PID `24476`, use the collision post-stop's still-active race as
  command-`31000`-reset-ready; issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000`, confirmation `0`, parameters 1--7
  zero, after normal disarm. Require boot rollback greater than `1000 ms`, a
  fresh nonnegative race-start timestamp, official index `0`, finish `-1`,
  and visible Gate 1, then run zero-command passive
  `n180_six_gate_hybrid_shadow_043`. Only a passing shadow may authorize one
  bounded Candidate 041; FullLap remains unauthorized.


## N180 Reset-31000 Setup and Shadow 043 — Passed; Candidate 041 Authorized

- Reused responsive simulator PID `24476` without UI interaction or relaunch.
  Exactly one reset-only MAVLink `COMMAND_LONG` command `31000`, confirmation
  `0`, parameters 1--7 zero, and normal pre-reset disarm rolled the continuing
  boot epoch from at least the post-Candidate-040 proof's `29214 ms` to
  `4459 ms`. A fresh race start appeared at `3288 ms`; setup ended base/status
  `193/4`, official index `0`, finish `-1`, zero collision, with `460` Gate-1
  detections. Setup SHA-256 is
  `99a2a05cd588bd214f896c45c8905158acc29acf2953dfc917e86c65d8d3ea64`.
- Shadow 043 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `286/286`, maximum action error `2.1029205321543287e-7`, inference
  `28.466 Hz`, and every deployment hash. No-op cadence was
  `561/10.031 = 55.827 Hz`, with zero rate violation and zero MAVLink setpoints.
  Official index stayed `0`, finish `-1`, with zero collision, telemetry
  dropout, or drain-limit hit; telemetry contained `2429` messages and `1252`
  IMU samples, while vision contained `119` frames and `119` detections.
  Shadow/parity SHA-256 values are
  `b3fbbd46effc59dc8bc80505bc5b0a244cacf6e878f792868e0971dbfa6f60fd`
  and `1ec384ed1028dca8ad15a8009e7991d1d4044e08da6d781d01ba095e9b534584`.
- Authorize exactly one `n180_six_gate_hybrid_gate4_bounded_041`: pinned N180
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 042 Rejected at Gate 3; N181 One-Sided Brake Retired

- Candidate 042 ran exactly once and is rejected without unchanged retry. Its
  command-`31000` reset rolled official boot `180297 -> 3345 ms`, produced
  fresh race start `3278 ms`, and was detected in `0.516 s`; calibration was
  `60/60`. It passed official Gates 1--2 (last gate time `7.545225620 s`), then
  collision `1001`, threat `2`, impact `9.727025985717773 m/s` stopped at
  official index `2`, finish `-1`.
- Delivery was healthy: `649/10.594 = 61.167 Hz`, duration `10.719 s`, zero
  rate violation, telemetry dropout, or drain-limit hit, `3311` telemetry
  messages, `1670` IMU samples, `137` frames, `132` detections, and one exit
  disarm. The command-free post-stop snapshot is disarmed/race-inactive at
  base/status `65/3`, race start `-1`, finish `-1`, and `reset_sent=false`.
- N181 produced its intended bounded roll `-0.5` at forward/right/right-rate
  states `5.026/-1.076/+2.034`, `4.085/-0.613/+2.293`, and
  `3.168/-0.178/+2.467` (m and m/s). The next sample crossed center to
  `right=+0.295 m` at `+2.717 m/s`, and collision followed. N180's stronger
  continuous full-scale brake had already failed Candidate 041. By contrast,
  clean Candidate 021 passed Gate 3 after positive roll near `4.156 m`, then
  level roll from `3.067 m` through the plane. Another one-sided brake
  threshold, cap, or continuation is therefore rejected.
- Aggregate/attempt/smoke/CSV/post-stop/diagnosis SHA-256 values are
  `102664962967f79742262f989448dcec37ddcc4d2496c1e98b9b64c65b325386`,
  `fbf0a7c52bcbde0626c4a31fb731da18cf0cb4a78505386584a3857997e61b43`,
  `5cf26b182efa8fc5a1fec14e0435e7a748f23bfa2d379b57c430f06cc838afda`,
  `f32609e5681c78e08eb20cdb0d395afadc8bfd147640d9f993a173ef1853d763`,
  `0a6a7280b477d37a8304ec2edd4e9a29a8ac33c4a8a2280d6da2878b017ec385`,
  and `e8508c57c9f95fb16dd7b70b426146df4fdabff715bc1cb85029a923cd7dd953`.
- No Candidate 043 and no FullLap are authorized. Next only: offline N182, a
  genuinely new receding-horizon minimum-energy lateral optimizer for Gate 3.
  Model lateral motion as a double integrator and, from observable residual
  `r`, rate `v`, and time-to-terminal `T`, use the first action of the exact
  terminal solution `a0 = -6*r/T^2 - 4*v/T`, then map with `atan(a0/g)` and a
  bounded roll. This is bidirectional terminal steering, not another brake
  patch. Replay all clean and failure traces, preserve non-roll channels, and
  document boundedness/source hashes before any recovery, reset, or shadow.





## Candidate 029 Rejected at Gate 2; N169 Offline Passed

- Candidate 029 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` reset command `31000` rolled boot
  `179957 -> 3399 ms`, race start `3297 ms`, with first detection at
  `0.484 s` and calibration `60/60`. It passed official Gate 1 only, then
  collision `1001`, threat `2`, impact `8.849061965942383 m/s` aborted
  at official index `1`, finish `-1`.
- Delivery was `460/7.812 = 58.756 Hz`, duration `8.234 s`, zero rate,
  dropout, or drain issue; `2764` messages, `1396` IMU, `99` frames,
  `93` detections, and one exit disarm. Aggregate/attempt/smoke/CSV/passive
  SHA-256 values are
  `cdda54b363b0f1be786e508a5db8a0f853291e09bf2036cc4cafe5dcd7960934`,
  `f79515d177fad1e6594cd9b72684c811a7b2a8bc07349a612383821bfe24b9ec`,
  `7dd025749f90e4ea120623ce8a38be031888df6a53a5ec1b2e09b84759d123d0`,
  `cf736519a27a9faa6d6c9e3197dd8a17b92c923cd0e9799e0cd45eb3b76e08ee`,
  and
  `ce95d608ead0b97fcd0736eb61e42017f1dd69a2175025980bd52a3c5e04312b`.
  The command-free passive snapshot confirms index `1`, last-gate time
  `5.447367668 s`, finish `-1`, and `reset_sent=false`.
- Candidate 029 projected about `3.38 m` below the Gate-2 plane at
  `11.99 m`, while all archived Gate-2 pass paths remain above the
  `-2.5 m` trigger (worst clean trace about `-2.26 m`). N169 therefore adds
  only a latched Gate-2 normalized thrust floor of `0.5` when the visible
  projected-down error is below `-2.5 m` inside `12 m). It preserves pitch,
  roll, yaw, and stronger learned thrust and clears on official transition.
- Source-hashed replay activates only on the two archived low-path failures
  (014 and 029), changes zero samples in eleven archived Gate-2 pass/other
  paths, has zero non-thrust error, floor violation, or stronger-thrust
  reduction. N168 Gate-3 behavior and all earlier Gate-2/Gate-3/Gate-4 laws
  pass preservation replay.
- Policy/verifier/evidence/Gate-3-counter/Gate-3-pitch/Gate-3-vertical/
  Gate-3-terminal/Gate-2-brake/Gate-4/wrapper SHA-256 values are
  `6a28135d1010b9b2b7d127362a0995983c0ce3249fe58feaf2fb9b63fc7e34b8`,
  `67701b2bb7b4e297374baad4d9a5e696c927037756d5d6dfcf660cb00c1f1759`,
  `de0c453cb8cc7822336b5ec9511e9dbd38916bcd6e407cb7bb5950791eda586a`,
  `f1659d429872383cfb84fcb0588fd4be01227ecb72bf311be236b819b2db10fd`,
  `7fa6b9ea59b1424331bb9e1e871a0e0fba49adbdebc15f06ddc8518854d556b6`,
  `a445c72001c7b92c82430777fd8eb0d855c1fb7f95339402549c472b8c7636ad`,
  `06b93b871e01639c723c7e28527c10dd2d6422daf5704edc4401ef9d5a794aba`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `9e8eac8744d317f16920ce71dcee20d474797ced9642681d5b3a0881dbef68f1`,
  and `82d6c6c2cdb1b6e3a70b0e7ef98c2d107f449b4a5db222ea1f626ffdc8325f88`.
  Linux and actual-Windows targeted suites pass `65/65`; compilation and
  both PowerShell parsers pass.
- Next only: recover responsive PID `24476` without relaunch or control,
  issue exactly one separately logged reset-only MAVLink `COMMAND_LONG`
  command `31000` and require measured boot/race rollback plus visible
  Gate 1, then run passive zero-command `n169_six_gate_hybrid_shadow_032`.
  Candidate 030 and FullLap remain unauthorized until Shadow 032 passes.


## N169 Recovery, Reset Setup, and Shadow 032 Passed — Candidate 030 Bounded

- Same responsive simulator PID `24476` was recovered through the VQ1/R1 UI
  without relaunch or flight control to base/status `193/4`, index `0`,
  finish `-1`; recovery SHA-256
  `e6ca6f5aa723c3d2454f847d34ff62760c16969cdc3d7325069547fe69ebed96`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000`. It rolled boot from roughly `921482` to `5347 ms`,
  race start `3299 ms`, index `0`, finish `-1`, zero collision, and
  `230` Gate-1 detections. Setup SHA-256:
  `4bd826fdc80a2e6f3402b40eaae517ab01ed54dc7b9f82a0895cde383cb30d28`.
- Shadow 032 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `254/254`, maximum action error
  `2.5661010744082446e-7`, inference `25.281 Hz`, with all eleven
  deployment hashes matching. No-op cadence was
  `595/10.047 = 59.122 Hz`, zero rate violations and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, zero collision,
  dropout, or drain issue; `2438` messages, `1256` IMU, `121` frames
  and detections. Shadow/parity SHA-256:
  `c2c72f0758f898aa2b3aa431fd38383536592bb2cbc6f10b4372490587b15886`,
  `e245f91ced579c07374b6aaef9d5f305e85b9481087cf7bdde0956b6543ccf81`.
- Authorize exactly one
  `n169_six_gate_hybrid_gate4_bounded_030`: pinned wrapper, actual Windows
  Python, reused PID, one detected normal command-`31000` reset, requested
  `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/stop
  official index `4`. Accept only clean index `4`, finish `-1`, cadence
  `[50,100) Hz`, matching hashes, zero collision/invalid/dropout/drain/rate
  issue, exit disarm plus passive proof. Reject without unchanged retry.
  FullLap remains unauthorized.


## Candidate 030 Rejected for Gate-3 Frame Contact; Official Gate 3 Proven

- Candidate 030 ran exactly once and is rejected without unchanged retry.
  Normal command-`31000` reset rolled boot `109832 -> 3402 ms`, race start
  `3296 ms`, detection `0.516 s`, calibration `60/60`. It passed official
  Gates 1 and 2 before collision `1001`, threat `2`, impact
  `2.4179086685180664 m/s` stopped the controller on stale index `2`.
- The required command-free passive snapshot then proved official index `3`,
  last-gate race time `10.548810005 s`, finish `-1`, base/status `65/3`,
  and `reset_sent=false`. N169 therefore restored the Gate-2 path and passed
  Gate 3 while still contacting its frame; Gate-4 control never activated
  because collision abort preceded the delayed status transition. This is a
  milestone but not a clean accepted crossing.
- Delivery was `625/10.375 = 60.145 Hz`, duration `10.485 s`, zero
  command-rate violation, telemetry dropout, or drain hit; `3304` messages,
  `1669` IMU, `132` frames, `130` detections, and one exit disarm.
  Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `97c900c660336c38a96b96416e384d0cb0fe45fe79415d236ff03f003ade5a96`,
  `003ac40a9ce62478ac33bbfe4a7b274c4162d9e4421aea148971325cceaf0e86`,
  `df712bb7460556f3c719d10539a3d570f9340e66f9012e36132f4f85bb0aa8f4`,
  `e846954bbc75348ff1ffd2d552b6c5e6b63957b0e343545bf5f19333ca7f0689`,
  and `540ceb698c5bc65ba47c177e0a23ae84b3e34390b77f101b11cb0d6efeda95d7`.
- Freeze N169 and do not fly Candidate 031 yet. Diagnose Candidate 030 against
  archived clean Gate-3 paths using source-hashed replay; admit an N170 change
  only if a narrow observable condition separates this late, low-impact frame
  contact without disturbing the clean crossings. FullLap remains
  unauthorized.


## N170 Gate-3 Mid-Terminal Margin Bridge — Offline Passed

- Candidate 030 exposed a bounded gap between the strong Gate-3 intercept
  (projected-right trigger below `-0.8 m`) and safe terminal roll leveling
  (absolute projected-right at most `0.5 m`). Its action dropped to zero roll
  while the projected right position at the existing 2 m terminal point stayed
  between roughly `-0.74` and `-0.67 m`.
- N170 adds one stateless roll-only bridge: while Gate 3 is visible,
  `4 < forward <= 7 m`, closing speed is at least `1 m/s`, and projected
  right at the 2 m terminal point remains below `-0.5 m`, floor normalized
  roll at `0.3`. It preserves pitch, thrust, yaw, and any stronger roll; it
  stops before the existing close/terminal region.
- The initial broad `0--7 m` hypothesis was rejected offline because it
  changed three archived clean crossings. The admitted `(4,7]` window changes
  exactly four Candidate-030 samples at `6.853/5.714/4.948/4.102 m`, zero
  samples in all five archived clean Gate-3 paths, one rejected Candidate-019
  sample, and two rejected Candidate-028 samples. Non-roll error, roll-floor
  violation, and stronger-roll change counts are zero. Candidate 030's
  command-free official index-`3` proof is embedded in the evidence.
- Policy/verifier/evidence/counter/pitch/vertical/terminal/Gate-2-brake/
  Gate-2-vertical/Gate-4/wrapper SHA-256 values are
  `cc50f43681c5c59364b56aabf4a97ecd48b95abe72031285e95a756e3d993b6f`,
  `17e1fb0974d7595c3f1e0285893412f1368462de7dbbc5e149e8a622ace1a1d9`,
  `9ae68b8fb40a9a303f3ae9a2f2abe4339b7575bf8d8ae65913dc33600d19b173`,
  `d123fe43f1b6609f2565721623ccd419ed48f4b1d9cdccaa5ce4efc0c19d7581`,
  `1825217b849e6d37797ffdaf8c7438b0731dfa620b4597f2cead3ca05ad9b5ba`,
  `60f18a285d0f53fd4fda5c1e388e5e80b5d7455265fc9dd3c15bef640ccb64f8`,
  `dee90fdbf25db5238dd1457c536dd2ae7bb1ff11e43d36d3127436f9a3eab25a`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `397e3d81631d0e435b8b44e77feed5d67c5f339b36dbf81aa4cca7db16c76f02`,
  `fb10f3a153e3495b7a22ce617043a1c47acb178130df4398f6b613fd5809b875`,
  and `0202a8f8350e2a0f4861771d1b74a8412dcf03a87042e1c2232800d87c7a06b8`.
  Linux and actual-Windows targeted suites pass `67/67`; compilation and
  both PowerShell parsers pass.
- Do not fly N170 yet. Next only: recover responsive PID `24476` without
  relaunch or control, issue exactly one separately logged reset-only MAVLink
  `COMMAND_LONG` command `31000` with measured boot/race rollback and
  visible Gate 1, then passive zero-command
  `n170_six_gate_hybrid_shadow_033`. Candidate 031 and FullLap remain
  unauthorized until Shadow 033 passes.


## N170 Recovery, Reset Setup, and Shadow 033 Passed — Candidate 031 Bounded

- Same PID `24476` recovered without relaunch/control to base/status
  `193/4`, index `0`, finish `-1`; recovery SHA-256
  `78433b8d2e745447f3848987ab581dd98f1d7665104725f8b94d40f38b9ec908`.
  One reset-only command `31000` rolled boot from roughly `1000617` to
  `5342 ms`, race start `3361 ms`, zero collision, and `263` Gate-1
  detections; setup SHA-256
  `0f92a2ad35b17b46bac03ec3247c192839ac0c9a7614e0efe724b972a6c6ddde`.
- Shadow 033 sent zero reset/arm/setpoint/disarm. Replay passed `280/280`,
  maximum action error `2.2355041501276318e-7`, inference `27.913 Hz`,
  all eleven hashes matching; no-op cadence
  `579/10.015 = 57.713 Hz`, zero violation/setpoint. Index stayed `0`,
  finish `-1`, zero collision/dropout/drain issue; `2411` messages,
  `1242` IMU, `121` frames/detections. Shadow/parity SHA-256:
  `9759b266484affc8cf6223cd36f062e209fc63fa096774dcd4ab64a01fd6106e`,
  `092599b750532233f686b956dc91c1eabdd7dabd90140922f9d739fc6ef45e4c`.
- Authorize exactly one
  `n170_six_gate_hybrid_gate4_bounded_031`: pinned wrapper, actual Windows
  Python, reused PID, one detected command-`31000` reset, requested
  `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/stop
  official index `4`. Require clean index `4`, finish `-1`, cadence
  `[50,100) Hz`, matching hashes, zero collision/invalid/dropout/drain/rate
  issue, exit disarm plus passive proof. Reject without unchanged retry.
  FullLap remains unauthorized.


## Candidate 031 Rejected at Gate 3; N171 Offline Passed

- Candidate 031 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `102817 -> 3401 ms`, race start `3302 ms`, reset detection `0.515 s`,
  and calibration `60/60`. It passed official Gates 1 and 2, then collision
  `1001`, threat `1`, impact `0.40086275339126587 m/s` aborted on
  official index `2`, finish `-1`; Gate 4 never activated.
- Delivery was `617/10.187 = 60.469 Hz`, runner duration `10.282 s`,
  with zero command-rate violation, telemetry dropout, or drain hit;
  `3252` messages, `1642` IMU, `130` detector frames, `129`
  detections, and one exit disarm. Aggregate/attempt/smoke/CSV/passive
  SHA-256 values are
  `e22b378ecc45689ee7da58798b9166ed2ab3243f4749f3b510a1b9a356e000f7`,
  `e0c7e49747027a622d3c630addabfbc9006e0dae9b9a778f0ce0b4c857948665`,
  `a85993f5b100f8df5b8009703214ce914990a702374d5a92ec06609457ee9633`,
  `7a26af9644d8420b500a8c051d2d9e9acbb90854ee32e78d9e087c18ae49feac`,
  and `86ea9c52242d9e69d9d6ec38f218c85dbf1b0c17ece3fcbc2e29a2fd74f805ea`.
- The required command-free passive snapshot cannot prove Gate 3: after the
  collision the simulator made the race inactive at base/status `65/3`,
  race start `-1`, index `0`, finish `-1`, with `reset_sent=false`.
  Therefore no vision-local pass count is promoted over official status.
- Source-hashed reconstruction shows N170's margin bridge never engaged.
  At `4.119821 m`, right `-0.605744 m`, right rate
  `2.103262 m/s`, and projected terminal right `-0.064931 m`, roll had
  leveled to `0`; at `3.555615 m` the corresponding values were
  `-0.350734 m`, `2.221926 m/s`, `+0.065237 m`, and roll `0`.
  The remaining separator is high late lateral rate despite a centered
  terminal projection. Diagnostic SHA-256:
  `98660f4ee59272d711860c42468c9cc53c42b770a92a342a3e261b22543b7451`.
- N171 adds one bounded roll-only brake after terminal leveling: visible Gate
  3, `3 < forward <= 4.5 m`, closing speed at least `1 m/s`, right
  position negative, right rate at least `2 m/s`, absolute projected right
  at the existing 2 m terminal point at most `0.5 m`, and current roll no
  greater than zero. It sets normalized roll to `-0.3`, preserving pitch,
  thrust, yaw, and any positive roll.
- Replay changes exactly the two Candidate-031 samples above, zero
  Candidate-030 samples, and zero samples in all five archived clean Gate-3
  paths. N170's bridge and every retained Gate-2/Gate-3/Gate-4 law pass
  preservation replay.
- Policy/verifier/evidence/diagnostic/N170-margin/counter/pitch/vertical/
  terminal/Gate-2-brake/Gate-2-vertical/Gate-4/wrapper SHA-256 values are
  `e9fb75533044eaeb1db3dc7548e2617579b4a68fb6b9882d232082feb2d276ee`,
  `c43fc0a2526e8d6bea6090e6c7f7c70a379340e56938353fdd76279068cd7d21`,
  `69b6980dc1f013abae08f8b978c2e8c16b6e922837879ff5138bb596def815d5`,
  `98660f4ee59272d711860c42468c9cc53c42b770a92a342a3e261b22543b7451`,
  `0e381aa429b0e582ec07bf96e260468cebb421d629da73ef821162190e44d25d`,
  `a1b5386d49ee896de3680872a392e9220b3239d736d4b66c229dbf35cc8f5464`,
  `764a81728d669365eaf7d9198a1d89c8e09db69c14622b14f68c9737c64008eb`,
  `97f4883af3e1a87d38a203f7474266413cd5bad78462ae63a1cd5bb951bc28d7`,
  `39679583a6da32e247030b48ba70116a1c76ad20ad5124c2bf3df9ace0e5257c`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `c70c374a87deb5a549c519962365a23787abdab0e62a7e56b0ed76f4e21ede10`,
  `d1290d6bd7ad621de453c8cc4a2435fd48896715318029d6c6695972c865d414`,
  and `27d95e89081b4f3c441b9e57210975591b28d1ed609dcad33c09d2dd939d0807`.
  Linux and actual-Windows targeted suites pass `69/69`; compilation and
  both PowerShell parsers pass.
- Do not fly N171 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` and require measured boot/race
  rollback plus visible Gate 1, then run passive zero-command
  `n171_six_gate_hybrid_shadow_034`. Candidate 032 and FullLap remain
  unauthorized until Shadow 034 passes.


## N171 Recovery, Reset Setup, and Shadow 034 Passed — Candidate 032 Bounded

- Same responsive simulator PID `24476` recovered through the VQ1/R1 UI
  without relaunch or flight control to base/status `193/4`, index `0`,
  finish `-1`; recovery SHA-256
  `3d7d8f3f4d59207832cb7914c68e73005370de15541feb82f2273d5a1daf8a64`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000`. It rolled the continuing boot epoch from roughly
  `1137735` to `3344 ms`, published new race start `3273 ms`, index
  `0`, finish `-1`, zero collision, and `259` Gate-1 detections. Setup
  SHA-256:
  `2e334bfedfb71a30792cb8f64b4e04d6fbbeaa4e632d4873fe7bbb5f2321b41e`.
- Shadow 034 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `226/226`, maximum action error `1.864559173447855e-7`, inference
  `22.528 Hz`, with all eleven deployment hashes matching. No-op cadence
  was `557/10.0 = 55.6 Hz`, with zero rate violation and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, with zero collision,
  dropout, or drain issue; `2436` messages, `1253` IMU, `109`
  detector frames and detections. Shadow/parity SHA-256 values are
  `2f688fa1033f2f96d6b2648e5e219472f61fc7804820df92288ccf91ccee6109`
  and `0a4e2cb969ad800bdaa28e66d2ff811fe9a134f5686e36752b49155a4c1eec07`.
- Authorize exactly one
  `n171_six_gate_hybrid_gate4_bounded_032`: pinned wrapper, actual Windows
  Python, reused PID `24476`, one detected normal command-`31000` reset,
  requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target,
  minimum, and authoritative stop official index `4`. Accept only clean
  index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.


## Candidate 032 Rejected for Gate-2 Frame Contact; Official Gate 2 Proven

- Candidate 032 ran exactly once and is rejected without unchanged retry.
  Normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `116859 -> 3410 ms`, race start `3268 ms`, reset detection `0.500 s`,
  and calibration `60/60`. It passed official Gate 1 before collision
  `1001`, threat `2`, impact `8.27989387512207 m/s` stopped the
  controller on stale official index `1`, finish `-1`.
- The required command-free passive snapshot then proved official index `2`,
  last-gate race time `7.516726493 s`, finish `-1`, base/status `65/3`,
  and `reset_sent=false`. Candidate 032 therefore crossed Gate 2 while
  contacting its frame; Gate-3 control never activated because collision abort
  preceded the delayed official transition. Vision-local counts do not replace
  this judge proof.
- Delivery was `380/7.031 = 53.904 Hz`, runner duration `7.125 s`, with
  zero command-rate violation, telemetry dropout, or drain hit; `2522`
  messages, `1275` IMU, `73` detector frames/detections, and one exit
  disarm. Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `2d1170f22b0e4e1ef1fe49c06bdeede60c42371ff7e8bf060119547019b1337f`,
  `49428713cbc5af93c12fe846f83b5cc472b7e3afa015442a243e87da24ca4d37`,
  `c80dc0c0b878b39831f008eab22bc2d3840a6b06fdc145ba88495dc967a0cea4`,
  `edd8f94c858a55adc9db25c39d5c95832da6b3a77340e87d40ee0adb270ed319`,
  and `d84b2ac739bf948b714b8089e3406f5719aeaf94356932626ac4a4fd1353780c`.
- The last two Gate-2 samples remained on the positive/right projected-miss
  side: at `3.157895 m`, right `0.927632 m`, right rate
  `-0.381434 m/s`, projected right `0.798827 m`, and roll
  `-0.423077`; at `2.473223 m`, the corresponding values were
  `0.851450 m`, `-0.482944 m/s`, `0.722906 m`, and `-0.341840`.
  The PD derivative term weakened correction even though the projected
  crossing stayed outside the aperture.

## N172 Gate-2 Late Residual Roll Floor — Offline Passed

- N172 adds one stateless, one-sided roll-only floor after the existing Gate-2
  projected-aperture governor: while Gate 2 is visible, forward distance is
  in `(0,4] m`, and projected right at the gate plane remains above
  `0.65 m`, retain normalized roll at or below `-0.5`. It preserves pitch,
  thrust, yaw, and any stronger negative roll; all prior Gate-2 latches and
  thresholds remain unchanged.
- The threshold is evidence-separated: the largest residual in the fifteen
  archived clean Gate-2 paths is about `0.616 m`, while Candidate 032's two
  samples are `0.799/0.723 m`. Replay changes exactly those two Candidate-032
  samples, zero samples across all fifteen clean paths, and is independently
  exercised on prior positive-edge failures Candidate 015 (two), 017 (two),
  and 023 (one). Non-roll error, floor violation, and stronger-roll change
  counts are zero. Candidate 032's command-free official index-`2` proof is
  embedded in the evidence.
- N171's late Gate-3 brake, N170's margin bridge, and every retained
  Gate-2/Gate-3/Gate-4 law pass source-hashed preservation replay.
- Policy/verifier/evidence/Gate-3-late/margin/counter/pitch/vertical/terminal/
  Gate-2-early/Gate-2-vertical/Gate-4/wrapper SHA-256 values are
  `1d16f44f728730c4e1946992acf0d9dd14493b1d915a0a52a195236080eac683`,
  `d5508423df1448b246767525fe19bd9f4fd856406e7b0745edcb963e5bc17bd2`,
  `017c7fbdb5c5c18274c686da371ff3540ec6230ad32de3d6acdc86202bd29511`,
  `da5f3a13398167c8d1bdf336f8b3a6e4cead15374e8e99f057fb81d2fc35ccfe`,
  `776da728ffc15c57c7651557835d5c989c9214a30a327373e9a7ca7e35963006`,
  `8ded5507e7f2281fb5c8d4cdaf471c3aa0509e12ec2210965ca5123f5c69be56`,
  `c53fac50a807569aa25cb1ab183b14c5e832d922e2a142f2537e6f9fd114fbaf`,
  `8993a64f110250ba86e870daee083253dedfebf07da3fecab9cf53bbca003f0b`,
  `cafd31f4ce99fd25a1e024d1c66e991b1508aa546ed9458f2a6d1a54eec29fef`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `0a2f46e884422af26729852a3f7ce829b1490246a82ab310e34694c9a53cc24a`,
  `2f6d213fe127e83cc249428df4f48d27956e439f9d0ac54d6a6e3a118bc9546f`,
  and `dbcf6a21f2eac3596d9e6a5ef476e9cbfc81b486ef0af106b2c5de97f73a0143`.
  Linux and actual-Windows targeted suites pass `71/71`; compilation and
  both PowerShell parsers pass.
- Do not fly N172 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` and require measured boot/race
  rollback plus visible Gate 1, then run passive zero-command
  `n172_six_gate_hybrid_shadow_035`. Candidate 033 and FullLap remain
  unauthorized until Shadow 035 passes.

## N172 Live Lifecycle and Shadow 035 — Passed; Candidate 033 Authorized

- Recovered the already-running responsive simulator PID `24476` without
  relaunch or flight control. It reported base/status `193/4`, official index
  `0`, finish `-1`, boot time `729868 ms`, and race start `730607 ms`.
  Recovery SHA-256 is
  `fd6f62bb01cf58b435a79837d956c2d38c59e350a5cc13ee2075f483428b9c25`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with parameters 1--7 zero. The continuing boot epoch rolled
  from `729868` to `3491 ms`, a new race start appeared at `3318 ms`, and the
  passive setup ended at official index `0`, finish `-1`, zero collision, with
  `255` Gate-1 detections. Setup SHA-256 is
  `88fe19990038c2f6ecebf07370d8dabc2c5c6d7e093d32da73871fa67181ee6d`.
- Shadow 035 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `287/287`, maximum action error `2.2973365784717537e-7`, inference
  `28.654 Hz`, and all eleven deployment hashes. No-op cadence was
  `611/10.0 = 61.0 Hz`, with zero rate violation and zero MAVLink setpoints.
  Official index stayed `0`, finish `-1`, with zero collision, dropout, or
  drain issue; telemetry contained `2420` messages and `1246` IMU samples,
  while vision contained `123` frames/detections. Shadow/parity SHA-256 values
  are `363f2d25eb33f9d7ef85e5f2be6e33cca64a605ac6e966b2c4a3efc9f63b0bf8`
  and `272003923bcba947590175e79e2d305f15820c1a7749f72b5793fa1319704e62`.
- Authorize exactly one `n172_six_gate_hybrid_gate4_bounded_033`: pinned N172
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 033 Rejected for Gate-1 Frame Contact; No Official Gate Pass

- Candidate 033 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `279893 -> 3358 ms`, published race start `3279 ms`, was detected in
  `0.531 s`, and completed calibration `60/60`. Collision `1001`, threat `2`,
  impact `8.113656997680664 m/s` stopped the controller at official index `0`,
  finish `-1`.
- The required command-free passive snapshot also reports official index `0`,
  last-gate time `-1`, finish `-1`, base/status `65/3`, and
  `reset_sent=false`; the vision-local pass count of one is therefore not an
  official Gate-1 pass. The collision-truncated runner reported
  `185/4.688 = 39.249 Hz`, duration `4.813 s`, zero command-rate violation,
  dropout, or drain hit, `1980` messages, `1001` IMU samples, `43` detector
  frames/detections, and one exit disarm.
- The final observable crossing prediction is a separated positive/right miss:
  at forward `5.581 m`, right `0.776 m`, right rate `0.378 m/s`, projected
  right `0.970 m`, roll `-0.209`; at `3.764 m` the corresponding values are
  `0.953`, `0.477`, `1.083`, and `-0.273`; at `3.032 m`, `1.029`, `0.616`,
  `1.128`, and `-0.314`; and at `2.397 m`, `1.043`, `0.649`, `1.084`, and
  `-0.328`. The annotated closest frame shows the Gate-1 left upright filling
  the image while the opening remains to camera-right.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `e2e2adc0be31c1416d1341eda55ea0690856f2ef537466d91bc8703fcfd7ed9c`,
  `d38c28610004de7bc8c8d83ff8e4ee44cae7f88ff669e0053392c82a92e913a0`,
  `dd3be5f9ff3f65261e3629a7cd68c785af9a7efbbefaf2b627e6e2eaf356c199`,
  `41c14318d20c14c95be2fecaae829015d9ba04900e08a38403be3a29865630f0`,
  and `2ab3e27cf59414d0c61feedafb96232e0078d0a96a6b07e0fac3f6076fcbd08c`.
- Do not reset or fly again. Next only: build and source-hash an N173
  Gate-1-only late positive projected-miss roll floor, prove zero changes on
  every archived clean hybrid Gate-1 path, exercise Candidate 033 and the
  comparable Candidate-008 miss, preserve all N172 laws, then preregister a
  fresh passive shadow. Candidate 034 and FullLap remain unauthorized.

## N173 Gate-1 Late Residual Roll Floor — Offline Passed

- N173 adds one stateless, Gate-1-only, roll-only floor: while the gate is
  visible, forward distance is in `(0,6] m`, and the observed position/rate
  projects the opening more than `0.8 m` to camera-right at the gate plane,
  retain normalized roll at or below `-0.7`. It preserves pitch, thrust, yaw,
  stronger negative roll, the recurrent state, and every Gate-2/3/4 law.
- The boundary is evidence-separated. Candidate 033 activates on exactly six
  samples, with projected right values `1.078--1.319 m`; the comparable
  Candidate-008 miss activates once at `2.317 m`. It changes zero samples on
  all 21 archived clean hybrid Gate-1 paths (Candidates 011 and 013--032),
  whose largest projection in the same window is `0.194 m`. Non-roll error,
  floor violation, and stronger-roll change counts are zero, and the
  command-free Candidate-033 no-pass proof is embedded in the evidence.
- Every retained Gate-2/Gate-3/Gate-4 verifier passes source-hashed
  preservation replay. Policy/verifier/evidence/Gate-3-late/margin/counter/
  pitch/vertical/terminal/Gate-2-early/Gate-2-vertical/Gate-2-late/Gate-4/
  wrapper SHA-256 values are
  `a2835e54a58945de2f9cc1b2d026a13af495779dc4488bc85b5ba9b62d1a7a8e`,
  `2f7f23d77b244ebaab19d9c0fa68ff73cded6c7694a36e7ad561520aaf4445ff`,
  `dc0bc0d03c01198f98967b14dbe8f259689b3ba66030cfb857e941c659114aba`,
  `0f83890606fea015ab15d3131d37c3954cabe18683a42e954f3c000aff93d22c`,
  `aa9a6c3bb581826d38f639531dc8d95d8d5b63922fc1229e40fe699e3ce2d392`,
  `a64900ed42a25d2a875995d388b0381880d1e8394a91557150ef4547e8744901`,
  `8e1a840b588075a8ca902f29352cac8e5cbf74cb4fbae8d3a74aa7359db9ffb2`,
  `5a97dfd9792eb475d81d1e228b914675233d37fb0aa9fcfd6f59b61dd5481b35`,
  `bb56e9cc2283c2f1c3430c77f8298aeb139c4af38786d3c4735fc44ba1ffc628`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `cad75cec8a15e208b9b55851a4bba410eb8785b8f1d63944a474dbe14274d9e2`,
  `e698a22e582bddf6842ba3ee9dc2d24f4ba2046f4df8590be0502122391d89cd`,
  `dce02248b29b51dca4a3a4915377d32bad7c501806beda5098600fe4e61df5d3`,
  and `45110347517be9c987f24ba3aff3788901634512efec7cfa3e7cb0d3eccaf2ef`.
  Linux and actual-Windows targeted suites pass `40/40`; compilation and both
  PowerShell script parsers pass.
- Do not fly N173 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` and require measured boot/race
  rollback plus visible Gate 1, then run passive zero-command
  `n173_six_gate_hybrid_shadow_036`. Candidate 034 and FullLap remain
  unauthorized until Shadow 036 passes.

## N173 Live Lifecycle and Shadow 036 — Passed; Candidate 034 Authorized

- Recovered the already-running responsive simulator PID `24476` through its
  UI without relaunch or flight control. It reported base/status `193/4`,
  official index `0`, finish `-1`, boot time `1037640 ms`, and race start
  `1038137 ms`. Recovery SHA-256 is
  `6194921400917acf20881d960be4a2eeab4648e5596112443feca1cc57687685`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with parameters 1--7 zero. The continuing boot epoch rolled
  from `1037640` to `4302 ms`, a new race start appeared at `3280 ms`, and the
  passive setup ended at official index `0`, finish `-1`, zero collision, with
  `241` Gate-1 detections. Setup SHA-256 is
  `82fb98c4f4ee015abb7b5c49491ed31efdfc575ae8b6e2ec3ac0f7264652a0ce`.
- Shadow 036 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `274/274`, maximum action error `2.426536560018455e-7`, inference
  `27.4 Hz`, and all eleven deployment hashes. No-op cadence was
  `580/9.969 = 58.08 Hz`, with zero rate violation and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, with zero collision,
  dropout, or drain issue; telemetry contained `2404` messages and `1238` IMU
  samples, while vision contained `120` frames/detections. Shadow/parity
  SHA-256 values are
  `db02cc26134afe3e98e76472da88de87ed8650475cd1adfeb19cdfeae96ceeb8`
  and `37c0b663a2841c8f3950daa9b3a61d9003fc80360a11647728e8938fdf2efd52`.
- Authorize exactly one `n173_six_gate_hybrid_gate4_bounded_034`: pinned N173
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 034 Rejected at Gate 3; Clean Official Gates 1--2

- Candidate 034 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `149397 -> 3339 ms`, published race start `3283 ms`, was detected in
  `0.516 s`, and completed calibration `60/60`. The controller ran its full
  `45.015 s` without collision, invalid-run indication, or limp mode, but the
  command-free official result stopped at index `2`, last-gate time
  `7.677293777 s`, finish `-1`. The vision-local count of three is not an
  official Gate-3 pass.
- Delivery was healthy: `2774/44.875 = 61.794 Hz`, zero command-rate
  violations, telemetry dropouts, or drain-limit hits, `11062` MAVLink
  messages, `5585` IMU samples, `572` detector frames, `199` detections, and
  one exit disarm. The passive snapshot independently retained index `2`,
  last-gate time `7.677293777 s`, finish `-1`, base/status `65/3`, zero
  collision, and `reset_sent=false`.
- Gate 3 was a late positive/right residual miss. At forward/right/right-rate/
  projected-right values `4.034/-0.170/2.429/0.462 m`, N171 applied only its
  `-0.3` roll brake; the next reliable observations were
  `3.623/0.039/2.564/0.570 m`, `2.388/0.672/2.899/0.815 m`, and
  `2.434/1.103/3.071/1.277 m`, while roll had returned to zero. The associated
  current-frame confidences were `0.20625`, `0.3125`, and `0.546875`.
  Thereafter false far detections caused the motion filter to retain the last
  close pose while current confidence fell to `0.01875`; any correction must
  therefore require current-frame confidence and must not act indefinitely on
  the retained pose.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `93c9872c016dd073a9af9b368f5054cc3fb0450c2d2cac8b48eb7442deaed076`,
  `dd28309dba48324e9fccf98b13884d93145e050d34b12eb4d27c964f02648011`,
  `4643b82dc188c195243937dba9400825dcf6b8a2307e56ffc9e84e331762c6da`,
  `0c4b444a0102481417b3b9b1ef519dfe6551312f53957071849f7e9995613a8a`,
  and `ad0a629600daafd405f8f726822e1bd3d1b2ec96472ff4cba6812ccf7e16125f`.
- Do not reset or fly again. Next only: build and source-hash an N174
  Gate-3-only, confidence-gated, late positive-residual roll floor; prove it
  changes only the three reliable Candidate-034 samples and zero samples on
  every archived clean Gate-3 path, preserve all N173 laws, then preregister a
  fresh passive shadow. Candidate 035 and FullLap remain unauthorized.

## N174 Gate-3 Late Positive-Residual Roll Floor — Offline Passed

- N174 adds one stateless, Gate-3-only, confidence-gated roll floor after the
  existing N171 late brake. It acts only with a current visible frame, current
  confidence at least `0.15`, forward distance in `(0,4] m`, right position
  nonnegative, right rate at least `2.0 m/s`, and projected right at the 2 m
  terminal point above `0.5 m`; normalized roll is retained at or below
  `-1.0`. Pitch, thrust, yaw, stronger negative roll, recurrent state, and all
  other gate laws are unchanged.
- Replay changes exactly Candidate 034's three reliable samples at elapsed
  `10.578/10.703/10.812 s`, current confidence
  `0.20625/0.3125/0.546875`, and projected right
  `0.570/0.815/1.277 m`. It changes zero samples in every other archived
  Candidate 013--033 trace, including all five clean Gate-3 paths
  (Candidates 016/018/020/021/022). The confidence condition rejects `309`
  later Candidate-034 samples whose cached close geometry still met the
  kinematic trigger after current confidence fell below threshold. Non-roll
  error, floor violation, and stronger-roll change counts are zero; the
  command-free official index-`2`/finish-`-1` proof is embedded.
- All eleven retained Gate-1/Gate-2/Gate-3/Gate-4 source-hashed preservation
  replays pass. Policy/verifier/evidence/Gate-1/Gate-2-early/Gate-2-late/
  Gate-2-vertical/Gate-3-early-vertical/Gate-3-counter/Gate-3-late/
  Gate-3-pitch/Gate-3-margin/Gate-3-terminal/Gate-4/wrapper SHA-256 values are
  `c8fecda034d802880f02431d53a9551ff0952f7572574bd709e55fd55fa521d5`,
  `ecde313adb5b3f66b7be5d09e2e007f1881f3d1f8ae13610aa56d09b0b6a2411`,
  `ddc43f13d5ece3654ce9fb5348ac15c40ce98d18b4330eaa7a35971e81abc185`,
  `8320a64c06001748c45a49d35ac895aa210a43a84804287662104c000c3cc662`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `ec8cdf979aba8a01e513d140b8f821cd6f4710035870a17886a13c51f41c2330`,
  `95ffdffd9fd702adc4ee5e4e20bd22350b793f566b868e62285bc131264c1636`,
  `c698b8b5c2ad25a2686075fa159cced20816cdc9c105532e2e83e1f668203767`,
  `cfd3092f28066f4a8f296898ed928911a66a847bbb3edf8876a475f2315359cb`,
  `62fcd111cf19ff65a18cd67542c872bbb172fe0615d78b4d43e12391d054deba`,
  `46f82405947d51964e158079879abe48b111f3f5b4380399a45941211d526a08`,
  `a822e8494fbbcf20f3b56c9bae262df09d463992c31049f70dc1688a10d8b89d`,
  `6ca855058f22ad2952dc9c55e866909648d18e14f0e341c35b01f85fd5f086e0`,
  `fefd70bc974066fcd99e9bd91a1e4fd2c2d8e8a2d3570aeee73dacece2e1186d`,
  and `5c92e2c9362e0c26f38996954b176cd0f96816122a1d12aea880eb7e76fc09b1`.
  Linux and actual-Windows targeted suites pass `42/42`; compilation and both
  PowerShell script parsers pass.
- Do not fly N174 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` with confirmation `0` and parameters
  1--7 zero, require boot rollback over `1000 ms`, a fresh nonnegative race
  start, official index `0`, finish `-1`, and visible Gate 1, then run passive
  zero-command `n174_six_gate_hybrid_shadow_037`. Candidate 035 and FullLap
  remain unauthorized until Shadow 037 passes.

## N174 Live Lifecycle and Shadow 037 — Passed; Candidate 035 Authorized

- Reused responsive simulator PID `24476` without relaunch or flight control.
  The passive recovery probe found the session already command-`31000`
  reset-ready at base/status `193/4`, official index `0`, finish `-1`, boot
  `301776 ms`, and race start `3301 ms`. Recovery SHA-256 is
  `0d68c23a56e861f673bd37a7eeec0efc32fe8f54a28e9f68ab71f152932eb46a`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with confirmation `0` and parameters 1--7 zero. Boot rolled
  `301776 -> 3020 ms`, a new race start appeared at `3266 ms`, and the passive
  setup ended at official index `0`, finish `-1`, zero collision, with `193`
  Gate-1 detections. Setup SHA-256 is
  `74f269a3e3c723860ceef4dc6d5ebc476c01fa3d125c584d92d9c21ce87eb23c`.
- Shadow 037 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `228/228`, maximum action error `2.0781478882181403e-7`, inference
  `22.764 Hz`, and all eleven deployment hashes. No-op cadence was
  `587/10.0 = 58.6 Hz`, with zero rate violation and zero MAVLink setpoints.
  Official index stayed `0`, finish `-1`, with zero collision, telemetry
  dropout, or drain-limit hit; telemetry contained `2432` messages and `1253`
  IMU samples, while vision contained `115` frames and `112` detections.
  Shadow/parity SHA-256 values are
  `e884a08d0b99ec9ec599554f3da68576008d19b5ed020e34406b7ee254ef759b`
  and `4cc0fef8a109ac411ca32adf337360e50a136e30024f2f5d03289495d4e419a2`.
- Authorize exactly one `n174_six_gate_hybrid_gate4_bounded_035`: pinned N174
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 035 Rejected for Gate-2 Frame Contact; Official Gate 1 Proven

- Candidate 035 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `181444 -> 3484 ms`, published race start `3286 ms`, was detected in
  `0.516 s`, and completed calibration `60/60`. It passed official Gate 1 at
  `5.496647834 s`, then collision `1001`, threat `2`, impact
  `9.815644264221191 m/s` stopped control at official index `1`, finish `-1`.
- Delivery was healthy: `441/7.375 = 59.661 Hz`, duration `7.969 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `2750`
  MAVLink messages, `1390` IMU samples, `93` detector frames, `92`
  detections, and one exit disarm. The required command-free passive snapshot
  independently retained official index `1`, last-gate time `5.496647834 s`,
  finish `-1`, base/status `65/3`, and `reset_sent=false`.
- This is the first deployed N169-or-later run to enter the Gate-2 projected-
  vertical latch. From forward `11.707 m` through the last pre-impact sample
  at `3.139 m`, it held normalized thrust at exactly `+0.5` for ten
  consecutive trace samples. The final annotated frame places the camera ray
  near the opening's top-left margin. Its lateral state overlaps archived
  Gate-2 passes, while all archived clean Gate-2 paths—including Candidate
  034—leave the vertical latch inactive and retain learned thrust. The two
  historical traces that would have activated the N169 rule (Candidates 014
  and 029) were also Gate-2 failures; the rule has no clean live activation.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `03a4e908e5688fb5431789149ba05cbcee9422bac0c95d1af4f15a5aa800c9cd`,
  `e31dd99eeef672cd4ca5466ce5bb12696ee59c674606c730cf4ae9622b6bf638`,
  `850b242cb29c967e7227b9ac765c435a3c094534ff3d7e03b45b6a3afb0f0427`,
  `f9b0fac876861e6e21578fe0bec1aa0f494a0544dd257590a35059a9a6a5d4d1`,
  and `9b2f313e7637cf3f000c8b51349db81dd9cb45be3c9fc37ecef98ce2e66d8ae0`.
- Do not reset or fly again. Next only: build and source-hash an N175
  retirement of the unproven Gate-2 projected-vertical thrust floor, prove
  all clean Gate-2 paths remain unchanged, prove the Candidate-035 latch would
  release learned thrust rather than force `+0.5`, preserve every N174 law,
  then preregister a fresh passive shadow. Candidate 036 and FullLap remain
  unauthorized.

## N175 Gate-2 Projected-Vertical Floor Retirement — Offline Passed

- N175 retires the N169 Gate-2 projected-vertical thrust floor. The helper
  remains as an explicit no-op for deployment compatibility, clears its former
  latch, and preserves all four learned action channels unconditionally. The
  historical `12 m`/`-2.5 m`/`+0.5` constants remain only for forensic
  classification; the lateral aperture governor and every Gate-1/3/4 law are
  unchanged.
- Replay finds zero historical latch samples on all 16 archived clean Gate-2
  paths (Candidates 011/013/016/018--022/024--028/030/031/034). Only failed
  Candidates 014, 029, and 035 historically trigger, for `7`, `11`, and `10`
  trace samples respectively. N175 changes zero recorded action channels on
  every trace, leaves the latch false, and passes a synthetic sub-floor learned
  thrust unchanged at all ten Candidate-035 latch samples. The command-free
  official index-`1`/finish-`-1` proof is embedded.
- All eleven remaining N174 Gate-1/Gate-2/Gate-3/Gate-4 source-hashed
  preservation replays pass. Policy/verifier/evidence/Gate-1/Gate-2-early/
  Gate-2-late/Gate-3-early-vertical/Gate-3-counter/Gate-3-late/
  Gate-3-positive-residual/Gate-3-pitch/Gate-3-margin/Gate-3-terminal/Gate-4/
  wrapper SHA-256 values are
  `1c683b0e81b0366e8262e510f7a016d45e3db659d8afa53dbf272301434c4d29`,
  `c9a0c6101dd04fb673edaabf9a3f357698020277c83055be72bab74283303be9`,
  `fa30d83da0609c4f237fac6838588b1b535f13644c94c4cfff34395006a3cebd`,
  `68f14c0da2e02bc5977ab5f3b768fa5529eb41406d12450a2ffd0543304e56c5`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `d240bdfda6ba55225f83c575bb1a510953c1e71ef7db9e15d0f44966f76d851c`,
  `aec3f02e61e1ae6e21bc6d7e1a4307080ec9eb2c7a477cfe0eb22cd400c7a4aa`,
  `a0207b4197461222a11785dacbd8cbc20181da64855c7d277a1bd3988e5f784b`,
  `3810da44abc781445ac156b5916123a0e0b1e919595299002aa1333779ff3630`,
  `59f29636bd491096b12a016af06761c168b4a71a86a7a04d5fab9d9532583a64`,
  `344abf70f78ed0ef53d821346b1a156f05dcd32b3337f2e6b6ed28160e081501`,
  `5c58aa11bdb4337a6186a49364fc81848d176961f16c92d6288c1d6bfb6b5ca9`,
  `655c03a4fd519320797f91c62e4968155d8aee5a516017949131410bfe96f253`,
  `1b88115c99ed63d2c2f9148d679863ac108703480966a7f289c3be8fe22f5955`,
  and `31b81ca17a54311c59f91d8377ca70fc9e967f4dd67af9b08452dfdf73483e15`.
  Linux and actual-Windows targeted suites pass `42/42`; compilation and both
  PowerShell script parsers pass.
- Do not fly N175 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` with confirmation `0` and parameters
  1--7 zero, require boot rollback over `1000 ms`, a fresh nonnegative race
  start, official index `0`, finish `-1`, and visible Gate 1, then run passive
  zero-command `n175_six_gate_hybrid_shadow_038`. Candidate 036 and FullLap
  remain unauthorized until Shadow 038 passes.

## N175 Live Lifecycle and Shadow 038 — Passed; Candidate 036 Authorized

- Recovered the same responsive simulator PID `24476` through the VQ1/R1 UI
  without relaunch or flight control. It reached base/status `193/4`, official
  index `0`, finish `-1`, boot `822663 ms`, and race start `813083 ms`.
  Recovery SHA-256 is
  `05b2b77fd4c7df2d95b8acf4384899774932e358c2845738753a57d6bcd8f63d`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with confirmation `0` and parameters 1--7 zero. Boot rolled
  `822663 -> 3045 ms`, a new race start appeared at `3316 ms`, and the
  passive setup ended at official index `0`, finish `-1`, zero collision,
  with `207` Gate-1 detections. Setup SHA-256 is
  `82ff5ebcfcec3bd5450d350d3b0d917ea693137761a299f2463c522a1df63cb3`.
- Shadow 038 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `125/125`, maximum action error `2.401035308907673e-7`, inference
  `12.5 Hz`, and all eleven deployment hashes. No-op cadence was
  `550/9.985 = 54.982 Hz`, with zero rate violation and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, with zero collision,
  dropout, or drain-limit hit; telemetry contained `2432` messages and `1252`
  IMU samples, while vision contained `102` frames/detections. Shadow/parity
  SHA-256 values are
  `56fb08683b60677a1021903ee94fecf093232f054022e3522349896d2d62e092`
  and `2272f6b344d133c3e731cb3d90d307a6e51f0c1aa903ae3de343b8f4b3461660`.
- Authorize exactly one `n175_six_gate_hybrid_gate4_bounded_036`: pinned N175
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 036 Rejected for Low Gate-2 Frame Contact; N175 Falsified

- Candidate 036 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `152403 -> 3388 ms`, published race start `3287 ms`, was detected in
  `0.531 s`, and completed calibration `60/60`. It passed official Gate 1 at
  `5.081851482 s`, then collision `1001`, threat `2`, impact
  `9.508764266967773 m/s` stopped control at official index `1`, finish `-1`.
- Delivery was healthy: `441/7.484 = 58.792 Hz`, duration `7.797 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `2690`
  MAVLink messages, `1360` IMU samples, `97` detector frames, `89`
  detections, and one exit disarm. The command-free passive snapshot retained
  official index `1`, last-gate time `5.081851482 s`, finish `-1`,
  base/status `65/3`, and `reset_sent=false`.
- N175 falsifies full retirement: the learned Gate-2 path projected down
  `-2.79` to `-3.48 m` inside about `9.8--9.1 m` and still projected about
  `-2.11 m` at the last coherent `2.89 m` pose. Learned thrust rose from
  `0.176/0.301/0.245/0.151` in the severe early window to
  `0.437--0.673` close, but too late; the last annotated image is almost
  entirely orange frame, consistent with low/bottom-bar contact. Candidate
  035's persistent `+0.5` floor instead ended near the top-left aperture
  margin. The evidence therefore bounds the correction between those extremes.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `bf48cefd28b1f8e4f3cd1bee802da2bdb655f36edaa5d48961539a5e2de97df1`,
  `a804be863ada626ec858321b54d025ae9bd6ea50b00f2e91a30396c736c0164d`,
  `37c1217f92c90cb6e38e0f73c14fc0ae252da5e527e364ec2c3a5d31e8406a00`,
  `902fc64528677d7d41f9a12b1afcaaecb25fe1a6ff62fa7aa43cb93f3cdea91a`,
  and `3e815c665628fbc33811ee956d4f3250d3efa8f05d484300e604a5dfd585ba9d`.
- Do not reset or fly again. Next only: build and source-hash an N176
  stateless moderate Gate-2 vertical floor of `+0.3`, active only while the
  current visible plane projection remains below `-2.5 m` inside `12 m`, with
  no latch after recovery. Prove zero clean-path changes, exactly three
  Candidate-036 learned-action changes, bounded Candidate-035 activation and
  recovery, preserve every N175 nonvertical law, then preregister a fresh
  passive shadow. Candidate 037 and FullLap remain unauthorized.

## N176 Stateless Moderate Gate-2 Vertical Floor — Offline Passed

- N176 replaces both N169's persistent `+0.5` floor and N175's full
  retirement with one stateless current-sample rule: while Gate 2 is visible,
  forward distance is in `(0,12] m`, and the current plane projection is below
  `-2.5 m`, retain normalized thrust at or above `+0.3`. It never latches,
  releases immediately after projection recovery, preserves pitch/roll/yaw
  and stronger learned thrust, and changes no lateral or later-gate law.
- Replay changes zero samples on all 16 archived clean Gate-2 paths. Candidate
  035 is severe for five samples and recovered for the next five; synthetic
  sub-floor thrust is raised to `+0.3` only on the five severe samples and
  passes through unchanged on all five recovered samples. Candidate 036 is
  severe for seven samples but only three weak learned thrust commands
  (`0.176`, `0.245`, and `0.151`) rise to `+0.3`; its four stronger commands
  are untouched. Non-thrust error, stronger-thrust change, and residual-latch
  counts are zero; the command-free index-`1`/finish-`-1` proof is embedded.
- All eleven N175 nonvertical Gate-1/Gate-2/Gate-3/Gate-4 source-hashed
  preservation replays pass. Policy/verifier/evidence/Gate-1/Gate-2-early/
  Gate-2-late/Gate-3-early-vertical/Gate-3-counter/Gate-3-late/
  Gate-3-positive-residual/Gate-3-pitch/Gate-3-margin/Gate-3-terminal/Gate-4/
  wrapper SHA-256 values are
  `1380d654b208daba7a9e0b84d9e0ad4eda680bf52068db86871009dc5483ce60`,
  `bc647ac85817a3a4ff07cbc853bbe39d7b50761c3555869e5e70aa2e519a43ff`,
  `1d54a25e45794c15a56b0f877360cf9080985c5a62c66b8e13d4bbb32de953e3`,
  `e202be8ee3f9b6519251f1a5a81e2b712021616b97a8a59ce44d23d24f436763`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `543be05bffadba395086d5f6fec6b045adc08ebf0588df5776419126a4f42156`,
  `b5daa546254ea60680973171ba41ed529a7c6e8cb5f56320887df5d111bf57c1`,
  `fd9cab28093debb3e2403665a5ce4d29adbe3c5bd2f6cf25fd55811ac30711e6`,
  `45743d15c5037f98dfcbee2735ac425fd30569ca173e2896b4f819327d952efa`,
  `f6a981629c74da04358513042212369bdf6811762a2f03fafa5d485fcc59c712`,
  `39c6d252208d314011339e3e959c2818322831fd86cfd1181cbb61ca37b3d787`,
  `4205007d2c5b8944b4a083a1796871e9ad0b6d72dc175db158ebb2b5c27083f3`,
  `fff65d1c651c2a60636098db98f5343030951929bf7fc9c63fd07bf3dc184e8d`,
  `a2a47421fad2b8e580e3a85c3b86b403986eab50e476d75fa8315cea638494cf`,
  and `3a6bd8db4c23bc9557d388a58f9c300a85b85977c46cdd7a5abdc9d7acdda2e4`.
  Linux and actual-Windows targeted suites pass `42/42`; compilation and both
  PowerShell script parsers pass.
- Do not fly N176 yet. Next only: recover responsive PID `24476` without
  relaunch or flight control, issue exactly one separately logged reset-only
  MAVLink `COMMAND_LONG` command `31000` with confirmation `0` and parameters
  1--7 zero, require boot rollback over `1000 ms`, a fresh nonnegative race
  start, official index `0`, finish `-1`, and visible Gate 1, then run passive
  zero-command `n176_six_gate_hybrid_shadow_039`. Candidate 037 and FullLap
  remain unauthorized until Shadow 039 passes.

## N176 Live Lifecycle and Shadow 039 — Passed; Candidate 037 Authorized

- Recovered the same responsive simulator PID `24476` through the VQ1/R1 UI
  without relaunch or flight control. It reached base/status `193/4`, official
  index `0`, finish `-1`, boot `533568 ms`, and race start `523603 ms`.
  Recovery SHA-256 is
  `43230f1d4c1ffae7cabe465939085ef1006eb6d89c2ca7553cf0ebdc3faf85b1`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with confirmation `0` and parameters 1--7 zero. Boot rolled
  `533568 -> 3295 ms`, a new race start appeared at `3267 ms`, and the
  passive setup ended at official index `0`, finish `-1`, zero collision,
  with `249` Gate-1 detections. Setup SHA-256 is
  `63ba373be57d6f680717ce022e1ddeba0f575b93c6f750971959c880ad8745ba`.
- Shadow 039 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `298/298`, maximum action error `2.746414184406909e-7`, inference
  `29.708 Hz`, and all eleven deployment hashes. No-op cadence was
  `605/9.984 = 60.497 Hz`, with zero rate violation and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, with zero collision,
  dropout, or drain-limit hit; telemetry contained `2448` messages and `1260`
  IMU samples, while vision contained `123` frames and `122` detections.
  Shadow/parity SHA-256 values are
  `b2b693e3a965b4401ef083acb0e8ccac6ba25a0135c5b702caefc709d5f5e251`
  and `8e5f638bedcc6137c2d250fd9739f08f66d91e724420a1c52af2e581489e1eaa`.
- Authorize exactly one `n176_six_gate_hybrid_gate4_bounded_037`: pinned N176
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 037 Rejected at Gate 3; N176 Gate-2 Floor Validated

- Candidate 037 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` reset command `31000`, confirmation `0`, and
  parameters 1--7 zero rolled simulator boot `152386 -> 3377 ms`; the fresh
  race start was `3286 ms`, reset detection took `0.500 s`, and calibration
  completed `60/60`. It passed official Gates 1 and 2, reaching official index
  `2` at `7.647600650 s`, before collision `1001`, threat `2`, impact
  `4.530916213989258 m/s` stopped control at the Gate-3 frame. Finish remained
  `-1`.
- Delivery was healthy: `622/10.469 = 59.318 Hz`, duration `10.766 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `3323`
  MAVLink messages, `1678` IMU samples, `134` detector frames/detections, and
  one exit disarm. The required command-free passive post-stop snapshot
  retained official index `2`, last-gate time `7.647600650 s`, finish `-1`,
  base/status `65/3`, and `reset_sent=false`; there was no delayed Gate-3
  credit.
- N176 achieved its intended bounded correction: unlike Candidates 035 and
  036, Candidate 037 cleared Gate 2 cleanly, so the stateless `+0.3` severe-
  projection thrust floor is retained. The new failure is isolated to Gate 3.
  During the last Gate-3 approach the rightward rate grew from about
  `+1.70` to `+2.92 m/s`; at forward `3.52 m` the existing centered-rate brake
  issued only `-0.3` roll, then at `2.24 m` the terminal-level rule returned
  roll to `0` despite the still-growing rate. The final reliable projected
  right residuals (`-0.36`, then `-0.17 m`) cleanly separate these samples
  from all archived successful Gate-3 paths under the existing `0.5 m`
  centered-projection bound.
- Aggregate/attempt/smoke/CSV/passive SHA-256 values are
  `c8187457ffd4b9c22e93c61881e2e3e537dafc6443384e52a4a24b23240c3222`,
  `d5a71f4fa68204fad710a54393793db9e2f131f63bbb6e183a4bd1b3940dd570`,
  `2bfeb1bf627aec1d7bfac0d833b9dc15074482934e08771111cc56e9fdf8e3d8`,
  `d806a533ab23a9cc40db20340f7dc99738b63df17b2e54dd5a112887269b860f`,
  and `2c99d73f21de17fdc6925a21b30e88c21da47ce0a566b667b41304d1b0bb3750`.
- Do not reset or fly again. Next only: build and source-hash an N177 change
  that strengthens the already isolated Gate-3 centered lateral-rate brake
  from `-0.3` to `-1.0` and extends its lower forward bound from `3.0` to
  `2.0 m`. It must preserve every clean Gate-3 trace, positive learned roll,
  all non-roll channels, N176 Gate 2, and all other laws, then pass a fresh
  passive shadow. Candidate 038 and FullLap remain unauthorized.

## N177 Strong Extended Gate-3 Lateral-Rate Brake — Offline Passed

- N177 changes only the existing Gate-3 centered lateral-rate brake: normalized
  roll is strengthened from `-0.3` to `-1.0`, and its exclusive lower forward
  bound is extended from `3.0` to `2.0 m`. The rule still requires visible
  official Gate 3, forward at most `4.5 m`, closing speed at least `1 m/s`,
  negative current right residual, rightward rate at least `2 m/s`, terminal
  projection within `0.5 m` of center, and nonpositive learned roll. It takes
  the minimum with the current roll so it cannot weaken an already stronger
  negative command; pitch, thrust, yaw, and positive learned roll are
  preserved.
- Source-hashed replay changes zero samples in all five archived clean Gate-3
  paths (Candidates 016/018/020/021/022). It changes exactly two samples in
  failed Candidate 030, two in failed Candidate 031, one in Candidate 034's
  missed Gate-3 approach, and two in failed Candidate 037. Candidate 037 spans
  both the old and extended windows: `3.518 m` changes `-0.3 -> -1.0`, and
  `2.240 m` changes `0 -> -1.0`. Non-roll error, positive-roll changes, and
  roll-output violations are all zero; the command-free post-stop official
  index-`2`/finish-`-1` proof is embedded.
- Every remaining N176 Gate-1/Gate-2/Gate-3/Gate-4 source-hashed preservation
  replay passes. Policy/verifier/evidence/Gate-1/Gate-2-early/Gate-2-late/
  Gate-2-stateless/Gate-3-early-vertical/Gate-3-counter/Gate-3-positive-
  residual/Gate-3-pitch/Gate-3-margin/Gate-3-terminal/Gate-4/wrapper SHA-256
  values are
  `0249a94a28dca4811c0471aa96ec5c1de385af9ae91445a2d80181fbacab8b8e`,
  `be5c2414d981d3927dfdc420cc9c94faa93ce46bd3d6217a6ea71847a3cd5ecc`,
  `8471ab52bd10c03ce707f5572c62b4e55f57e2794842e7d705b0650e81d35899`,
  `f3ebaa7b3a25dc966d94986b9eebb325ef6f76707ff04a6e5d6dfd52db27d6c2`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `1f065cde5909dbabf22094d2d8ab739ce4ed5b95315ee5a80173c820c9575d9c`,
  `5211e599152ed9c6c664ba6eba2ac7ac50d3c612acc111e40d73e1cfcbeb54ed`,
  `f8a9515217f72e2d0e3ff8a4d974b41c1a2ec47fa989d064d15d8746c45cccdd`,
  `923c796a39a66286ee19c65740798966db0f576ef853f8ca7538816ab4b1ac3d`,
  `c93d333b153b885d0d6dc032dc067748cfac16ad8905fa5e6e9b67223a8f401a`,
  `4631e4a4c56849280be83451fe823df78c1a82fe050dcc6a16d9b4d743cee8c4`,
  `aa4b2261aed553d40170c5b201c6f4da7c948ac6b3beed1dbc49e98c0bd68da2`,
  `e7745c347cdb0bc67010072097b7920a3cfb5c2ccda6fcbd0d51b8015ba6b368`,
  `93b8ef992d3f2a7015be20b00a42beb411677d11398750fefe0c79a8715c861c`,
  and `abde9c280fb40f715306c56ec1c80bb54955087339de7e4f5e158e72f90269d9`.
  Linux and actual-Windows targeted suites pass `44/44`; compilation and both
  PowerShell parsers pass.
- Do not fly N177 yet. Next only: recover responsive simulator PID `24476`
  without relaunch or flight control, issue exactly one separately logged
  reset-only MAVLink `COMMAND_LONG` command `31000` with confirmation `0` and
  parameters 1--7 zero, require boot rollback over `1000 ms`, a fresh
  nonnegative race start, official index `0`, finish `-1`, and visible Gate 1,
  then run passive zero-command `n177_six_gate_hybrid_shadow_040`. Candidate
  038 and FullLap remain unauthorized until Shadow 040 passes.

## N177 Live Lifecycle and Shadow 040 — Passed; Candidate 038 Authorized

- Reused the same responsive simulator PID `24476` without relaunch or UI
  mutation. The passive recovery probe found the collision session reset-ready
  at base/status `65/3`, official index `2`, finish `-1`, boot `816019 ms`, and
  race start `3286 ms`. Recovery SHA-256 is
  `88ffd01cd4194b6234afec3cf131135c9f025e0028d806026a40e1c17b567e3f`.
- Exactly one separately logged reset setup sent MAVLink `COMMAND_LONG`
  command `31000` with confirmation `0` and parameters 1--7 zero after normal
  disarm. Boot rolled `816019 -> 4182 ms`, a fresh race start appeared at
  `3282 ms`, and the passive setup ended at base/status `193/4`, official index
  `0`, finish `-1`, zero collision, with `434` Gate-1 detections. Setup SHA-256
  is `4c033ae2f4e6c5ac6ec7958cd50ee006fe78d6ec18abfe2181a2e923e8ac15a9`.
- Shadow 040 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `269/269`, maximum action error `2.692115783631266e-7`, inference `26.9 Hz`,
  and every pinned deployment hash. No-op cadence was
  `569/9.984 = 56.891 Hz`, with zero rate violation and zero MAVLink
  setpoints. Official index stayed `0`, finish `-1`, with zero collision,
  dropout, or drain-limit hit; telemetry contained `2415` messages and `1244`
  IMU samples, while vision contained `118` frames/detections. Shadow/parity
  SHA-256 values are
  `9757f0cf9966954a51fb6fc24b6bfd7ddb69d39d84605fdda2be41c8a51b198b`
  and `96c035f62acd9f001c6fca587c09b45877a96d78af18af3bff6dde9fc5b23d20`.
- Authorize exactly one `n177_six_gate_hybrid_gate4_bounded_038`: pinned N177
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 038 Rejected at Gate 3; N177 Braked Too Late

- Candidate 038 ran exactly once and is rejected without unchanged retry. Its
  normal MAVLink `COMMAND_LONG` command-`31000` reset rolled boot
  `159795 -> 3378 ms`, published a fresh race start at `3297 ms`, was detected
  in `0.500 s`, and completed calibration `60/60`. It passed official Gates 1
  and 2, reaching official index `2` at `7.408710002 s`, before collision
  `1001`, threat `2`, impact `10.444109916687012 m/s` stopped control at Gate
  3. Finish remained `-1`.
- Delivery was healthy: `603/9.797 = 61.447 Hz`, duration `10.328 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `3215`
  MAVLink messages, `1624` IMU samples, `121` detector frames/detections, and
  one exit disarm. The required command-free passive post-stop snapshot found
  the simulator's collision-inactive state at base/status `65/3`, race start
  `-1`, finish `-1`, and `reset_sent=false`; there was no delayed Gate-3
  credit.
- N177 did issue full negative roll at `3.057` and `2.463 m`, but the rightward
  rate was already `2.476--2.486 m/s`. Two immediately earlier samples expose
  the remaining priority gap: at `5.304 m`, right `-1.326 m`, rate
  `+2.200 m/s`, projected residual `-0.538 m`, the terminal-margin law changed
  the counter command to `+0.3`; at `4.636 m`, right `-1.046 m`, rate
  `+2.279 m/s`, projected residual `-0.386 m`, terminal leveling changed it to
  `0`. N177's `4.5 m` maximum then delayed full braking until only about
  `3.1 m`. All five clean Gate-3 paths remain outside a proposed `0.55 m`
  projection bound at their comparable late zero-roll samples.
- Aggregate/attempt/smoke/CSV/passive/diagnosis SHA-256 values are
  `698f5262976e50ccaff86963338e6665a07cb4e0aa61ce7df140adf96d45cb28`,
  `f1f4ab72158236e7de33bd67b6d1d539b09c3a8de8f6b49865ce70c8457ff028`,
  `1ceefd8ad28b9c715c64f5dd0471103b90a54ede13fb339563df4130ce75e34e`,
  `777bf28d2860ff3f2b094dc05ea65131e31e1b0d8b6807e803741a70f35a2ea0`,
  `2b775bc414f5bd5b884acae1044dc3d2d80ac180770e1510f343b1c06d513996`,
  and `20ec90fa844c0a37f2060a47e5122d08fae2843cc6c4b69da0cdcad5e484ad50`.
- Do not reset or fly again. Next only: build and source-hash N178 by extending
  the existing Gate-3 rate brake's maximum forward bound from `4.5` to
  `6.0 m`, widening its centered terminal-projection bound only from `0.50`
  to `0.55 m`, and allowing it to override the exact `+0.3` terminal-margin
  floor as well as nonpositive roll. Prove zero changes on every clean Gate-3
  pass, exactly the two intended new Candidate-038 samples, preservation of
  all other laws, then require a fresh passive shadow. Candidate 039 and
  FullLap remain unauthorized.

## N178 Earlier Gate-3 Margin-Priority Brake — Offline Passed

- N178 retains N177's full `-1.0` Gate-3 rate brake and `2.0 m` exclusive
  lower bound, but extends its maximum forward distance from `4.5` to `6.0 m`
  and widens the centered terminal-projection bound only from `0.50` to
  `0.55 m`. The rule may now override the exact `+0.3` output of the terminal-
  margin floor; all other positive roll remains protected. Visibility,
  negative current right residual, closing speed at least `1 m/s`, and
  rightward rate at least `2 m/s` are still required. Pitch, thrust, yaw, and
  already stronger negative roll remain untouched.
- Replay changes zero samples in every archived clean Gate-3 path (Candidates
  016/018/020/021/022). Candidate 038 changes exactly the two preregistered
  earlier samples: at `5.304 m`, the exact margin floor changes
  `+0.3 -> -1.0`; at `4.636 m`, terminal level changes `0 -> -1.0`. No other
  positive roll or non-roll channel changes. Failed Candidate 030 has three
  target samples, failed Candidate 031 two, Candidate 034's missed Gate 3 two,
  and failed Candidate 037 two. The command-free collision-inactive
  Candidate-038 proof is embedded.
- Every remaining N177 Gate-1/Gate-2/Gate-3/Gate-4 source-hashed preservation
  replay passes. Policy/verifier/evidence/Gate-1/Gate-2-early/Gate-2-late/
  Gate-2-stateless/Gate-3-early-vertical/Gate-3-counter/Gate-3-positive-
  residual/Gate-3-pitch/Gate-3-margin/Gate-3-terminal/Gate-4/wrapper SHA-256
  values are
  `f5566fe001961e7a74b81ca4d06fb36a4ba7b0a4801935bad260800f884129db`,
  `1ad163ddf18938b2b2c51c3f40c0a491684042e991694bcbe446dab0aae9f9ac`,
  `d2b995b7c42d585a0d3d847a9942dae9909ec075dab7d8dff425cc880563d985`,
  `82edd0971f794c30d95dde410b4d919e49125f086ddcfbb96b1838ce0efdd28a`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `e1c389173a519033b26b39aca94c4acd9568dfb564003299cadb31b02d693aed`,
  `9bc6c9e1338192693f8ae9a4f16d2061129a30368b2432da78014f40aa1a6a7a`,
  `587c325a1ee6c111803794e70d23231abde3bf9b030335b0edc5ab8ca61e8a14`,
  `9daa1eaec475421bef04bb7bac091e481cf8fbe0275abb163cb1f7df8a97a04d`,
  `79e192ac24948c372594574be468ae6a71187566316c64417559f61d1661fd12`,
  `0fa2535ed03d0aca9ba4482a4635a5134a525afaa0b42f9e0ffb108bc050d01c`,
  `f05c0886933d29e50018f06e6215086e8985af9a084fd45aed176f40aac36d13`,
  `b2ce4de3d4f7f17d1d223394e8a21d624b266fc0f107785a99a52ed01980a905`,
  `36ed8266bc6f83f9654d9a8a8652479c66516a2ea85d2232862723f9b4dc72cc`,
  and `87291793fd106d3a1556c2682c05b12fd568ca0d6c0d5dfb60cc7a8f3d7c491e`.
  Linux and actual-Windows targeted suites pass `45/45`; compilation and both
  PowerShell parsers pass.
- Do not fly N178 yet. Next only: recover responsive PID `24476` through the
  VQ1/R1 UI without relaunch or flight control because Candidate 038 left the
  race inactive; then issue exactly one separately logged reset-only MAVLink
  `COMMAND_LONG` command `31000` with confirmation `0` and parameters 1--7
  zero. Require boot rollback over `1000 ms`, a fresh nonnegative race start,
  official index `0`, finish `-1`, and visible Gate 1, then run passive zero-
  command `n178_six_gate_hybrid_shadow_041`. Candidate 039 and FullLap remain
  unauthorized until Shadow 041 passes.

## N178 Live Lifecycle and Shadow 041 — Passed; Candidate 039 Authorized

- Recovered the same responsive simulator PID `24476` through the VQ1/R1 UI
  without relaunch or flight control after Candidate 038 left the race
  inactive. The ready proof reached base/status `193/4`, official index `0`,
  finish `-1`, boot `591064 ms`, race start `578115 ms`, and nonnegative race
  age. Recovery/ready SHA-256 values are
  `569c0d481490d0de5138131691bf0e7a5d3cd227d3436bde6d1de1db4d015f94`
  and `3cc6934dc3ee12d3991468278bf89ac025b960c7e9c9270605a3483436b7aed2`.
- Exactly one reset-only MAVLink `COMMAND_LONG` command `31000`, confirmation
  `0`, parameters 1--7 zero, and normal pre-reset disarm rolled boot
  `591064 -> 4172 ms`. A fresh race start appeared at `3302 ms`; setup ended
  base/status `193/4`, official index `0`, finish `-1`, zero collision, with
  `437` Gate-1 detections. Setup SHA-256 is
  `affbe58559c69c8705988d8c178655732e96a428ef26286cc4358ffa25f4978d`.
- Shadow 041 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `292/292`, maximum action error `2.5335693359940237e-7`, inference
  `29.2 Hz`, and every deployment hash. No-op cadence was
  `591/9.968 = 59.189 Hz`, with zero rate violation and zero MAVLink setpoints.
  Official index stayed `0`, finish `-1`, with zero collision, dropout, or
  drain-limit hit; telemetry contained `2414` messages and `1242` IMU samples,
  while vision contained `124` frames and `122` detections. Shadow/parity
  SHA-256 values are
  `0144b25861d36213c62d0f353383c9c6f4324063e1ae96857f1a072a594bf447`
  and `94695f339920dd1eadcdd564a8f569034c2388875f26341c8b7c2f2d0b41354e`.
- Authorize exactly one `n178_six_gate_hybrid_gate4_bounded_039`: pinned N178
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 039 Rejected at Gate 3; High-Rate Margin Gap Isolated

- Candidate 039 ran exactly once and is rejected without unchanged retry. Its
  detected command-`31000` reset rolled boot `152525 -> 3377 ms`, published a
  fresh race start at `3266 ms`, was detected in `0.531 s`, and completed
  calibration `60/60`. It passed official Gates 1 and 2 at `7.813352584 s`,
  then collision `1001`, threat `2`, impact `6.272382736206055 m/s` stopped
  control at official index `2`, finish `-1`.
- Delivery was healthy: `655/11.015 = 59.374 Hz`, duration `11.109 s`, zero
  command-rate violation, telemetry dropout, or drain-limit hit, `3411`
  MAVLink messages, `1720` IMU samples, `138` frames and `135` detections,
  plus one exit disarm. The command-free post-stop snapshot was collision-
  inactive at base/status `65/3`, race start `-1`, finish `-1`, and
  `reset_sent=false`; no delayed Gate-3 credit appeared.
- N178 did not activate early because three consecutive high-rate samples at
  `6.223`, `5.926`, and `5.905 m` projected `-0.628`, `-0.553`, and
  `-0.636 m`, just outside its `0.55 m` ordinary centered bound. Their
  rightward rates were exceptionally high at `3.144`, `2.986`, and
  `2.949 m/s`, yet the margin floor output `+0.3`; full negative braking began
  only at `3.854 m`. The closest comparable clean late zero-roll path is
  Candidate 021 at projected `-0.596 m` but only `2.520 m/s`, which provides a
  separate high-rate threshold without changing that pass.
- Aggregate/attempt/smoke/CSV/passive/diagnosis SHA-256 values are
  `29403e97517779175a517e52f34779a35fa81c1db57c8d3668a1680aa22ec6db`,
  `d1daa3272f6acad911da6b16d1f0cf9d8ee41f08f97ed48dbbd355e0babd415b`,
  `9fb9fc3a569488b986604b6e1642831fce237d8202f256e01d090165e0ede518`,
  `ac8761cc3317728677a472fde306faf9250f2b6a7acedc29a96eaf4b943bfc19`,
  `9f7d8922c129f67c78efc94f9400971cf90ceeccbe292de958d97945ab4355ce`,
  and `e82a1d38ef4f2d23a38c7f158dd7528cf5f12d2c98a7d77c81e5c11744792b4c`.
- Do not reset or fly again. Next only: build/source-hash N179 with the ordinary
  N178 condition retained, plus a separate high-rate tier allowing terminal
  projection within `0.65 m` and forward distance through `6.5 m` only when
  rightward rate is at least `2.8 m/s`. Prove all five clean Gate-3 passes
  unchanged, exactly three new Candidate-039 samples, and every other law
  preserved before a new passive shadow. Candidate 040 and FullLap remain
  unauthorized.


## N179 High-Rate Gate-3 Margin Brake — Offline Passed

- N179 retains N178's ordinary Gate-3 brake tier unchanged: forward distance
  through `6.0 m`, rightward rate at least `2.0 m/s`, and terminal projection
  within `0.55 m`. A separate tier extends only to `6.5 m` and `0.65 m`
  when rightward rate is at least `2.8 m/s`. Both tiers require current Gate-3
  visibility, negative current right residual, closing speed at least `1 m/s`,
  and forward distance above `2.0 m`; they may override nonpositive roll or
  the exact `+0.3` terminal-margin floor only. Pitch, thrust, yaw, stronger
  negative roll, and every other positive learned roll remain unchanged.
- Replay changes zero samples in all five archived clean Gate-3 passes
  (Candidates 016/018/020/021/022). Candidate 039 changes exactly the three
  preregistered `+0.3 -> -1.0` samples at `6.223`, `5.926`, and `5.905 m`;
  their measured rightward rates are `3.144`, `2.986`, and `2.949 m/s` and
  projected residuals are `-0.628`, `-0.553`, and `-0.636 m`. No non-roll
  channel or non-margin positive roll changes. Failed Candidates
  030/031/034/037/038 retain their expected 3/2/2/2/2 target samples, and the
  command-free collision-inactive Candidate-039 proof is embedded.
- Every retained Gate-1/Gate-2/Gate-3/Gate-4 preservation replay passes.
  Policy/verifier/evidence/Gate-1/Gate-2-early/Gate-2-late/Gate-2-stateless/
  Gate-3-early-vertical/Gate-3-counter/Gate-3-positive-residual/Gate-3-pitch/
  Gate-3-margin/Gate-3-terminal/Gate-4/wrapper SHA-256 values are
  `50c78eeffa6de4c0971dc1ba38b8d0bf941a7045e518d6ed0ee670a5937196a9`,
  `3db19fea9c6266a255ca33f195259ee9419335cb21bc747957c4850af41c8b46`,
  `b22ab4f8a3120cc6c534769ed2a75ade0922c2c2623644fd9e0df763026f05e4`,
  `ce70a5eefc2f8ed535d74931f64029aafdc3f4e9865125c62ab456f08b0afc04`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `632fa0179720b3a39b682ecb91a8d8259575bd4998a7046daa98273fa29ca38d`,
  `09c21a10e55adc5b95dd3204ff8a3a063b0f4a2ddce0b85187b9659f63236721`,
  `711af5ad28eaea71285cfd481841b8c87aca85a26ff07bda5e01335622617932`,
  `5b147ff697c5a42f8a5769966558c017f8f4b41792156e14d234285dc1c89727`,
  `c4f7150f9a84becb0127fca68b0e6dc33a8c4d03a9c31a55b864007108ce49db`,
  `fe07509fa4576cf14ff199769736975a9decaecdc48f08a0d84b61bcbb1e2af3`,
  `fd27c1b39e19835607cdd33dec0f304b9ae7d0ccc492bee5f9dbe95fc8b5cba4`,
  `b4767c0a989b133362dd67d4c32104f1142187931ade27a6003f0535d865f0eb`,
  `9ec20d347d3de610b1a661bba275a48d27fc071d788e277c1d986628b23a0d93`,
  and `06add70bee933ae0872155f33f1d2bbcc49c54f1cf619a6117b315b0477f47b2`.
  Linux and actual-Windows targeted suites pass `47/47`; compilation and both
  PowerShell parsers pass.
- The existing simulator is still responsive at reused PID `24476`; no reset,
  setpoint, or new flight was issued during offline validation. Candidate 040
  and FullLap remain unauthorized. Next only: recover VQ1/R1 on the same PID
  without relaunch or flight control if the collision-inactive session needs
  UI recovery; issue exactly one separately logged reset-only MAVLink
  `COMMAND_LONG` command `31000`, confirmation `0`, parameters 1--7 zero,
  after normal disarm. Require boot rollback greater than `1000 ms`, a fresh
  nonnegative race-start timestamp, official index `0`, finish `-1`, and
  visible Gate 1, then run zero-command passive
  `n179_six_gate_hybrid_shadow_042`. Only a passing shadow may authorize one
  bounded Candidate 040; FullLap remains unauthorized.


## N179 Live Lifecycle and Shadow 042 — Passed; Candidate 040 Authorized

- Recovered VQ1/R1 through the UI on the same responsive simulator PID
  `24476`, with no relaunch or flight control. A follow-up ready proof showed
  base/status `193/4`, official index `0`, finish `-1`, boot `738159 ms`,
  race start `715824 ms`, and nonnegative race age. Recovery/ready SHA-256
  values are
  `06cac541f03fc149de6b88b52a75ab0baad3375c3e403fd302479e54ed08fa1c`
  and `0524e514817549a2dbb1204ffd1cd22d4a26f192bee5de4c8637479cbbca7bb0`.
- Exactly one reset-only MAVLink `COMMAND_LONG` command `31000`, confirmation
  `0`, parameters 1--7 zero, and normal pre-reset disarm rolled boot
  `738159 -> 4239 ms`. A fresh race start appeared at `3282 ms`; setup ended
  base/status `193/4`, official index `0`, finish `-1`, zero collision, with
  `419` Gate-1 detections. Setup SHA-256 is
  `e8b4761762dae4ad1ff4b96d8fab12bd7a396017ef60dfbf6d110e413e6ac645`.
- Shadow 042 sent zero reset, arm, setpoint, or disarm commands. Replay passed
  `278/278`, maximum action error `2.3177242278182852e-7`, inference
  `27.756 Hz`, and every deployment hash. No-op cadence was
  `609/9.984 = 60.897 Hz`, with zero rate violation and zero MAVLink setpoints.
  Official index stayed `0`, finish `-1`, with zero collision, telemetry
  dropout, or drain-limit hit; telemetry contained `2418` messages and `1246`
  IMU samples, while vision contained `125` frames and `124` detections.
  Shadow/parity SHA-256 values are
  `9e10800a3dad409ee161c25111f7337a989d818be33410e4cb6badb3a4afd49f`
  and `288a246c0d0405bb46e9210f3af8ec238975f0244e7df43e9c7e08e961c9a0a0`.
- Authorize exactly one `n179_six_gate_hybrid_gate4_bounded_040`: pinned N179
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.


## Current Execution State — 2026-07-18 After Candidate 041

- This end-of-file checkpoint supersedes any older authorization text above.
  Candidate 040 (N179) and Candidate 041 (N180) were each run exactly once and
  rejected at official Gate 3; neither may be retried unchanged. Full details
  and hashes are recorded in the Candidate-040, N180, Shadow-043, and
  Candidate-041 sections above.
- The current frozen policy source hash is
  `fea39e91787f32712e3d5313ce42c8d9edf383addb3db0b32c757e1ac44b5bdf`;
  the N180 wrapper hash is
  `03e785c5c13e82b5cffafebb4996bc9aeceafcc497aabb5426c67dba34d3d169`.
  N180 passed all offline preservation contracts, Linux/Windows `48/48`, and
  passive Shadow 043, but its fixed full-scale Gate-3 brake still failed live.
- The reused v1.0.3385 simulator process remains responsive at PID `24476`.
  Candidate 041's command-free post-stop proof is disarmed and race-inactive
  (base/status `65/3`, race start `-1`, finish `-1`, `reset_sent=false`).
  Because the race is inactive, command `31000` may be ignored until VQ1/R1 is
  recovered through the UI on the same PID; do not relaunch the process.
- The only valid simulator reset is learned-target MAVLink `COMMAND_LONG`,
  command ID `31000`, confirmation `0`, parameters 1--7 zero, after normal
  disarm. Count it only when simulator boot time rolls back by more than
  `1000 ms` and a fresh nonnegative race-start timestamp appears.
- No Candidate 042 and no FullLap are authorized. Next only: offline N181,
  replacing the fixed `-1.0` Gate-3 terminal brake with a bounded continuous
  physics-based stopping command using `atan(a/g)`; preserve the five clean
  Gate-3 passes and all other laws before any recovery, reset, or shadow.

## N181 Offline Freeze — 2026-07-18

- N181 replaces only the eligible Gate-3 fixed full-scale brake with an
  observable stopping command: `a = right_rate^2 / (2 * max(-right, 0.5))`,
  `roll = -atan(a / 9.80665)`, capped at `-0.25 rad`, then normalized by the
  existing `0.5 rad` policy limit. The previously validated activation
  envelopes and protected positive learned-roll cases are unchanged.
- Replay evidence
  `logs/sitl/n181_gate3_continuous_stopping_brake_20260718.json` has SHA-256
  `4bed5726cba9b81c0a5a3a7105b563e085ac1fe5f91f7981548a01c08cd56fd2`.
  All five clean Gate-3 passes are action-identical. Intended roll-only changes
  occur on failing Candidate traces 030, 031, 034, 037, 038, 039, 040, and 041;
  there are no protected-positive-roll or non-roll changes. Candidate 041's
  four eligible commands become approximately `-0.432806, -0.5, -0.5, -0.5`.
- All eleven retained preservation replays pass with zero blockers. Their
  SHA-256 values, in Gate1-late, Gate2-early, Gate2-late, Gate2-stateless,
  Gate3-early, Gate3-counter, Gate3-positive, Gate3-pitch, Gate3-margin,
  Gate3-safe, and Gate4-bounded order, are `03b73fefe9253f48`,
  `768a7d135670c159`, `51821795ab6eaaec`, `c9963ca0537158f5`,
  `5d99d70c95a2ef7a`, `3dd7f91dc6188e4a`, `61a6bcf2ff6e7e43`,
  `7bb30d21dcacc275`, `38dfb3af963debd2`, `86a487431454729d`, and
  `cc83836e153ff025` (prefixes; full values are in the run ledger).
- Frozen SHA-256 values are policy
  `afaf45535548c69d6549ef12ac91efa47e180ff8a6a5bcd55fa85805ad17b6fe`,
  verifier
  `6635115f69baa505706587922bab2b766f974fe995ad1d8c9607de3787e7cf57`,
  and deployment wrapper
  `534701f2ca3dd5ada9e4dc48297c49e8d83669dbb02b679d9ba9c0c531f98d04`.
  Compile checks and PowerShell parsing pass; Linux and actual-Windows test
  suites each pass `48/48`.
- No simulator reset, arm, setpoint, or flight occurred during N181 development.
  Candidate 042 and FullLap remain unauthorized. Next only: recover VQ1/R1
  through the UI on reused PID `24476` without relaunch, prove one exact
  command-`31000` reset by boot rollback plus fresh race start, then run a
  command-free Shadow 044 before considering one bounded Candidate 042.

## N181 Live Setup and Shadow 044 — 2026-07-18

- Reused simulator PID `24476` was responsive and VQ1/R1 was recovered through
  the UI with `-NoRelaunch`; the process was not restarted. Passive readiness
  evidence `logs/sitl/n181_recovery_ready_20260718.json` has SHA-256
  `fb47a2ca340512db1a6b922a390c833b0abeef755fdf5b9cf2fa85883d8906ff`.
- Exactly one reset-only setup sent a normal disarm followed by learned-target
  MAVLink `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 zero. The
  official boot time rolled back from `968833 ms` to `4373 ms` (`964460 ms`),
  and fresh race start `3264 ms`, index `0`, finish `-1`, detections, and zero
  collision/dropout prove the reset. Snapshot SHA-256 is
  `1741688ae40989d4ef02ca096a1fc8b8e48380be1d0bb6a69aee1405628b0f05`.
- Shadow 044 sent zero reset, arm, disarm, or MAVLink setpoint commands. It
  replayed `295/295` samples with maximum action error
  `1.8615045548064924e-7`, inference `29.5 Hz`, all deployment hashes matched,
  and no blockers. Its no-op cadence was `588/9.985 = 58.788 Hz`, with zero
  rate violations. Official index remained `0`, finish `-1`, with zero
  collision, dropout, or drain-limit hit; telemetry had `2434` messages and
  vision `121` frames/detections. Shadow and parity SHA-256 values are
  `994e0814be1fdc2e49da228503d95e9e805fe799da92bf23614b23c1feaa3d68`
  and `61d2ab8d035f573856aa0c192d90744185019698704a5257c17e59439422aabb`.
- Authorize exactly one `n181_six_gate_hybrid_gate4_bounded_042`: pinned N181
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Current Execution State — 2026-07-18 After Candidate 042

- This physical end-of-file checkpoint supersedes every older authorization.
  Candidate 042 ran once and is rejected at official Gate 3; it may not be
  retried unchanged. N181's bounded one-sided brake is retired. The detailed
  Candidate-042 evidence and hashes are recorded above.
- The reused simulator remains PID `24476`; the command-free post-stop state is
  disarmed and race-inactive (base/status `65/3`, race start `-1`, finish `-1`,
  `reset_sent=false`). Do not relaunch it. The only valid reset remains normal
  disarm then learned-target MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero, proven by boot rollback over `1000 ms` plus fresh race
  start.
- No Candidate 043 and no FullLap are authorized. Next only: offline N182,
  replacing one-sided Gate-3 braking with the bidirectional receding-horizon
  minimum-energy terminal controller `a0=-6*r/T^2-4*v/T`, bounded through
  `atan(a0/g)`. Require trace replay, non-roll preservation, tests, and hashes
  before any UI recovery, reset, passive shadow, or flight.

## N182 Minimum-Energy Gate-3 Controller — Offline Freeze

- The initial hard center/zero-rate terminal objective was rejected offline
  because it saturated across nearly every clean pass. The admitted controller
  instead targets Candidate 021's proven competitive handoff at forward
  `3.3 m`, right `-0.9 m`, and right-rate `+2.5 m/s`, then commands level roll
  through the aperture. For horizon `T`, its exact minimum-energy first action
  is `a0=6*(r_target-r)/T^2-(4*v+2*v_target)/T`; roll is
  `atan(a0/9.80665)`, capped at `0.25 rad` (`0.5` normalized).
- The controller owns roll only for visible/current-confidence `>=0.1`, closing
  Gate 3 in `(2.0,6.5] m`. It is bidirectional and overrides the retired
  one-sided brake/residual branches only inside that envelope. Pitch, thrust,
  yaw, recurrent state, and all other gate laws remain untouched.
- Source-hashed replay covers clean Candidates 016/018/020/021/022 and failed
  030/031/034/037/038/039/040/041/042 with expected changed-sample counts
  `3/5/5/2/2/3/4/4/2/4/5/4/4/2`. Clean Candidate 021 receives positive
  approach steering; Candidate 042 receives earlier negative steering and its
  banked `3.17 m` crossing becomes level. All changes have trustworthy current
  frames, roll is finite in `[-0.5,+0.5]`, and maximum non-roll error is zero.
- All eleven retained Gate-1/Gate-2/Gate-3/Gate-4 preservation verifiers pass
  with zero blockers. Their SHA-256 values, in Gate1-late, Gate2-early,
  Gate2-late, Gate2-stateless, Gate3-vertical, Gate3-counter,
  Gate3-positive-residual, Gate3-pitch, Gate3-margin, Gate3-safe-coast, and
  Gate4-dropout order, are `098264cebfac428f`, `768a7d135670c159`,
  `30552d5b3f991b14`, `fdedf807f531c5d5`, `01ef2ceacdb0623d`,
  `214f3f17d279b559`, `0deea70b0bbf2c5a`, `d5fcbe1b65c8cbff`,
  `ac6b5cbf38192fcf`, `fe7249b52b600d2f`, and `6b6978b9ae8fc8b7`.
- Frozen policy/verifier/evidence/wrapper SHA-256 values are
  `3fc363876bef3cea67036bd7bb216c899b8358004b2c780ae40a1c1d2bee2585`,
  `9b9abfd9703e1deb682f8e00fcde3d5f64c2827541856c39eb5bb81000ca0959`,
  `54c0bc2c17d66861e083359b317c153e547fc0b1ec76e5b2fd10d53f87b02963`,
  and `e3c0aa01a59bd60b31dd0aea64e4d78f456ecee9cba591b4bbfe2100021046ae`.
  Linux and actual-Windows suites each pass `45/45`; compile checks and both
  PowerShell parsers pass.
- No simulator command occurred during N182 development. Candidate 043 and
  FullLap remain unauthorized. Next only: recover VQ1/R1 on reused PID `24476`
  without relaunch, prove one exact command-`31000` reset by boot rollback and
  fresh race start, then run command-free Shadow 045.

## N182 Live Reset Proof and Shadow 045 — Passed

- Reused responsive PID `24476` and recovered VQ1/R1 through the UI with
  `-NoRelaunch`. Passive readiness SHA-256 is
  `1d19571cd91688259d4319e27bd928e04b708d7721b4fed07a5307d3c00f27b3`.
- One reset-only setup sent normal disarm then learned-target MAVLink command
  `31000` with confirmation/parameters zero. Official boot rolled
  `1556574 -> 3793 ms` (`1552781 ms`), with fresh race start `3315 ms`, index
  `0`, finish `-1`, visible Gate 1, and zero collision/dropout. Setup SHA-256
  is `a7625a14a0dcec7284e1c5a19b7b32adbd13fd9fe1140b4e488ff1c7dee24a30`.
- Shadow 045 sent zero reset, arm, disarm, or MAVLink setpoints. Replay passed
  `302/302`, maximum action error `2.391401290879891e-7`, inference `30.2 Hz`,
  every deployment hash matched, and no blockers. No-op cadence was
  `608/9.984 = 60.797 Hz` with zero violation. Official index stayed `0`,
  finish `-1`, zero collision/dropout/drain hit, `2415` telemetry messages,
  `126` frames, and `124` detections. Shadow/parity SHA-256 values are
  `5e92dd3d7daf1cfde03d471b6c0c6c56f70b7869d654586d30b7b59f123a4581`
  and `cc516caa9dd189d85adac2543c02685e8d05ed779b22578f804811218f187b50`.
- Authorize exactly one `n182_six_gate_hybrid_gate4_bounded_043`: pinned N182
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive proof. Reject without unchanged retry. FullLap remains unauthorized.

## Candidate 043 Rejected at Gate 3; Receding N182 Roll Oscillation Retired

- Candidate 043 ran exactly once and is rejected without unchanged retry. Its
  detected command-`31000` reset rolled boot `167666 -> 3364 ms`, produced
  fresh race start `3300 ms`, detection `0.485 s`, and calibration `60/60`.
  It passed official Gates 1--2 (last gate time `7.846203804 s`), then
  collision `1001`, threat `2`, impact `10.669703483581543 m/s` stopped at
  official index `2`, finish `-1`.
- Delivery was healthy: `630/10.188 = 61.739 Hz`, duration `10.813 s`, zero
  rate violation, telemetry dropout, or drain-limit hit, `3321` telemetry
  messages, `1676` IMU samples, `131` frames, `130` detections, and one exit
  disarm. The command-free post-stop is disarmed at base/status `65/3`, retains
  official index `2`, race start `3300 ms`, finish `-1`, and
  `reset_sent=false`; no delayed Gate-3 credit appeared.
- N182's final reliable roll sequence was `+0.5,+0.5,-0.5,0` at forward
  `5.65,5.14,3.53,2.27 m`. The terminal position was geometrically safe at
  right/down `-0.181/+0.206 m`, but the last negative bank was commanded only
  one sample before level, too late for attitude settling. Re-solving the
  short-horizon problem every sample caused sign reversals and is retired.
- Aggregate/attempt/smoke/CSV/post-stop/diagnosis SHA-256 values are
  `1bd02554451926f190ffcf0ea87c886f8152562b7f89ba3f7e4284bc4b7d8574`,
  `3b2d27ca1ced6daec1b15669fd35dfd7bfa23a22b7722432afee091af49fb5f5`,
  `c6fa0529bb3443bd9ab02d4f8b65dedc0c9e8e8a9435948d00d261e413777141`,
  `d4dff9162a989c144ed2e5d3096a156901fc3f9cb3c43c3a7f847b7161bad2f0`,
  `6e200db81b57dbf38dc1530977e40a87dd7529517386f7ff399e22bd98e2e933`,
  and `b3c9090ee0e8f14c4010178350884b6a662a893cb950c45c39eccb65308a94eb`.
- No Candidate 044 and no FullLap are authorized. Next only: offline N183,
  replacing receding re-optimization with one committed minimum-energy
  intercept. At the first trustworthy Gate-3 sample in `(4.0,6.5] m`, solve
  once for Candidate 021's proven approximately `4.0 m` handoff state
  (`right=-1.46 m`, `right-rate=2.34 m/s`), latch the bounded roll through
  `4.0 m`, then latch level roll for the remaining crossing. Clear both latches
  on reset/gate transition; preserve all non-roll channels and other laws.

## N183 Committed Minimum-Energy Gate-3 Controller — Offline Freeze

- N183 replaces N182's per-frame re-solve with one stateful decision. At the
  first current, visible, confidence-`>=0.1` Gate-3 sample in `(4.0,6.5] m`
  with closing speed `>=1 m/s`, it solves the same minimum-integral-`a^2`
  double-integrator intercept once for Candidate 021's proven `4.0 m`
  handoff (`right=-1.46 m`, `right-rate=2.34 m/s`). It maps
  `a0=6*(r_target-r)/T^2-(4*v+2*v_target)/T` through `atan(a0/g)`, caps bank
  at `0.25 rad` (`0.5` normalized), holds that committed roll to `4.0 m`,
  then latches roll `0` for the crossing. Reset and gate transitions clear
  both latches. Pitch, thrust, yaw, recurrent state, and every other gate law
  are unchanged.
- Stateful, source-hashed replay passes all `15` retained live traces:
  clean Candidates 016/018/020/021/022 and failed
  030/031/034/037/038/039/040/041/042/043. Expected changed-sample counts are
  `4/5/8/3/3/3/3/3/3/4/7/5/4/2/3`. Candidate 021 commits `+0.5` at
  `6.418 m` and levels at `3.072 m`; Candidate 043 commits `-0.5` at
  `5.650 m`, holds it through the next observation, and levels at `3.528 m`
  instead of reversing bank one sample before level. All controller-owned
  rolls are finite in `[-0.5,+0.5]`; maximum non-roll error is exactly zero.
- All eleven retained Gate-1/Gate-2/Gate-3/Gate-4 preservation verifiers pass
  with zero blockers. Their SHA-256 values are
  `73070ee75dc3e797069cdfac5b1b8b555387f9134921894001e68230dbd3616d`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `be66f4e141feac8bcf85e39f41078c9a156c2555a6c207540027dd02b414a5a0`,
  `8a3630a73e7bc2f22a8191ed0dda4ff4dd78ecaba12bb838b174a096b9ddf3f7`,
  `d9edc125d7267a005706a8fead5db1d1b30f8aa47f3bc1f68157b2d5a029afea`,
  `514e2ce5cfbeb09dd7952e27939590130405e81af95eb44d2570229818996dcd`,
  `6d08e229ce8175eb47ee40619e1e579b06ffa866156bf76043adcf3e92be1190`,
  `9f2edd95454c8294dfe16a9fca1e3df974dc74704baefcdb897b659f96f8c461`,
  `f61aa20265fd41a64d1ded8093b3b8221e70d10553358fd07e262b3d62c5d011`,
  `c0d9901ca23bb4dc4cc5799bbe51fe41a6a813a90b433525cfe384f9cc7ef09b`,
  and `37ddcaedc8ec391f7b836e48b3ee40ec141cd0b6a9298bd7c5d711750fa3bca5`.
- Frozen policy/verifier/evidence/wrapper SHA-256 values are
  `b077c853659735366c5ebda2f132770e3ffeb89e2df2998c7a476e4d724008ac`,
  `454b6ae16ca1177806abb5dbaf89f4972eca26814ad5116e5fce8c165fccf3b9`,
  `0a59e987450451e4ec92030751a87eb0c659ca7ddfcde5581ae1fb1d55d9a839`,
  and `f9c0a8b1d2413bd64233f33a927147645b1cd39f56e85193ad72f1578e0f47d8`.
  Linux and actual-Windows focused suites each pass `45/45`; compile checks
  and both PowerShell parsers pass.
- No simulator command occurred during N183 development. Candidate 044 and
  FullLap remain unauthorized. Next only: reuse responsive simulator PID
  `24476` without relaunch; recover VQ1/R1 through the UI only if required;
  send normal disarm followed by exactly one learned-target MAVLink
  `COMMAND_LONG 31000` with confirmation and parameters 1--7 zero; prove
  reset by boot rollback `>1000 ms` plus a fresh nonnegative race-start
  timestamp; then run command-free Shadow 046. No flight is authorized until
  that reset and shadow evidence is appended.

## N183 Live Reset Proof and Shadow 046 — Passed

- Reused responsive simulator PID `24476` throughout and performed no process
  restart. Candidate 043's active but disarmed session was not command-`31000`
  ready, so VQ1/R1 was recovered through the UI with `-NoRelaunch`; readiness
  evidence SHA-256 is
  `77c6df0235ec260399a0a28fca9ed03597946bf4f21e0c5eff6dced807a35881`.
- The reset-only setup sent normal disarm followed by exactly one learned-target
  MAVLink `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 zero.
  Official boot rolled `1207294 -> 3312 ms` (`1203982 ms`), with fresh race
  start `3326 ms`, index `0`, finish `-1`, armed base/status `193/4`, visible
  Gate 1, and zero collision/dropout. Setup SHA-256 is
  `8d100071a3f1d1d7d7d897cabead78f80c4d0b66fee72a46ff8ec45e90e5b3af`.
- Shadow 046 sent zero reset, arm, disarm, or MAVLink setpoints. Replay passed
  `244/244`, maximum action error `1.9448356630702435e-7`, inference
  `24.363 Hz`, all deployment hashes matched, and no blockers. No-op cadence
  was `560/10.0 = 55.9 Hz` with zero violation. Official index stayed `0`,
  race start `3326 ms`, finish `-1`, zero collision/dropout/drain hit, `2419`
  telemetry messages, `118` frames, and `115` detections. Shadow/parity
  SHA-256 values are
  `38ef8d0e4cc3f8036ddd52906b0e9d7c24cb5ba7ed422cf11aa3eb5f6e0ec906`
  and `8b78fd2f6e267c2c516c2f4803d092a56088b643d8b1c7a2806b410eeaf23d78`.
- Authorize exactly one `n183_six_gate_hybrid_gate4_bounded_044`: pinned N183
  wrapper, actual Windows Python, reused PID `24476`, one detected normal
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, exit disarm, and command-free
  passive post-stop proof. Reject without unchanged retry. FullLap remains
  unauthorized.

## Candidate 044 Rejected at Gate 3; N184 Exact N160 Gate-3 Rollback Next

- Candidate 044 ran exactly once and is rejected without unchanged retry. Its
  detected command-`31000` reset rolled boot `140620 -> 3407 ms`, produced
  fresh race start `3300 ms`, detection `0.500 s`, and calibration `60/60`.
  It passed official Gates 1--2 (last-gate time `7.372447013 s`), then
  collision `1001`, threat `2`, impact `10.105694770812988 m/s` stopped at
  official index `2`, finish `-1`.
- Delivery was healthy: `633/10.360 = 61.004 Hz`, duration `10.453 s`, zero
  rate violation, telemetry dropout, or drain-limit hit, `3296` telemetry
  messages, `1664` IMU samples, `133` frames, `130` detections, and one exit
  disarm. The command-free post-stop is disarmed at base/status `65/3`, retains
  index `2`, race start `3300 ms`, finish `-1`, and `reset_sent=false`; no
  delayed Gate-3 credit appeared.
- N183 removed sign oscillation but did not reproduce the proven trajectory.
  Candidate 044's reliable normalized roll sequence was
  `+0.905,+0.900,+0.902,-0.382,-0.382,0` at forward
  `9.143,8.907,7.007,5.714,5.220,3.609 m`; coarse estimator samples still had
  actual roll `+0.4506 rad` at `9.672 s` and `+0.2013 rad` at `10.188 s`.
  Offline recurrent replay establishes a simpler authority: Candidate 021's
  complete Gate-3 roll trace is reproduced exactly by the historical N160
  live path—only `tail._promoted_gate3_intercept` followed by
  `_gate3_close_severe_boost` with the original `4 m` counter-suppression
  bound. It includes full counter-bank at `8.639/8.275 m`, positive bank at
  `7.653/6.418/4.154 m`, and level at `3.072 m`; Candidate 021 passed Gate 3.
- Aggregate/attempt/smoke/CSV/post-stop/diagnosis SHA-256 values are
  `bd3b705c11efcf9bd0225d26f869ccfc9ea0eda2c51601d533955c0c0be26fee`,
  `2030ccf00f3d9d055b5e5d589b5d6ab228949449d0b2ba368b0c06ae2f54a8e8`,
  `bb041587feafb3037e2ab55085f107535a2b51c79c84a5dd415c838800d47922`,
  `421bab7d5ad7659b21445cd86922aea953695fd8862ebabbbed3aa7741de1007`,
  `cdea488a0656a9da352ded95b3265a167d7a82b6e9c67102961f8c5fa3d0821f`,
  and `9807f26582e36c51b027a39db297ec2fd75be43e478fc6809b0a7ecf9790673a`.
- Candidate 045 and FullLap are unauthorized. Next only: build N184 by setting
  Gate-3 counter suppression back to the N160 `4 m` bound and restoring the
  live Gate-3 action pipeline to exactly promoted intercept then close-severe
  boost. Retire terminal-margin, vertical-floor, terminal-level,
  late-positive-residual, and minimum-energy calls from the live path while
  preserving their historical replay helpers. Require sequential recurrent
  replay over all five clean Gate-3 traces plus Candidate 044, preservation of
  every non-Gate-3 law, tests, and pinned hashes before any UI recovery, reset,
  shadow, or flight.

## N184 Exact N160 Gate-3 Live-Path Rollback — Offline Freeze

- N184 restores the only Gate-3 action pipeline with direct official pass
  authority: unchanged recurrent v6c action, then exactly
  `tail._promoted_gate3_intercept`, then `_gate3_close_severe_boost` with the
  original N160 `4.0 m` counter-suppression bound. The five later Gate-3
  terminal-margin, projected-vertical, terminal-level, late-residual, and
  minimum-energy helpers remain in source for historical direct replay but are
  unreachable from live `infer`. Gate 1, Gate 2, Gates 4--6, the recurrent
  bridge/state, and all checkpoint files are unchanged.
- Source-checked action-law replay covers historical Gate-3 pass Candidates
  016/018/020/021/022 plus failed Candidate 044. Candidates
  018/020/021/022 are exact fixed points. Candidate 016 changes only three
  late left-side counter commands at `3.404/2.794/2.386 m` from `-1` to the
  N160 level output `0`. Candidate 044 changes only two samples at
  `5.714/5.220 m` from N183's `-0.3816846` to N160 full counter-bank `-1`.
  Maximum pitch/thrust/yaw error is exactly zero and all roll outputs are
  finite/legal. Candidate 021's complete proven terminal sequence remains
  `-1,-1,+0.905,+0.907,+0.906,+0.901,0` at
  `8.639/8.275/7.653/6.418/6.423/4.154/3.072 m`; it passed official Gate 3.
- The immutable prefix checkpoint SHA-256 remains
  `ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760`.
  All five retained non-Gate-3 Gate-1/Gate-2/Gate-4 preservation verifiers
  pass with zero blockers; their SHA-256 values are
  `e2554feb00951def799dd3ee470f733f12b3f3aab0cf93244bd32d2a7a0401d1`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `8303805500bfbaabfadf61bc4837f35f62497efa3c811303bdd6ceff6f684cba`,
  `f78782ddc821656b0a3ef4c58f0d84dd399656936dce7cc6a3f5a83f80519d93`,
  and `3d097a1b3846e8195ef88f94a03a3009714366e0945c040fef6c25e7a72d9796`.
- Frozen policy/verifier/evidence/wrapper SHA-256 values are
  `134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514`,
  `428f01d938714e0a8278f74001537e3579cccaf3ffef1944b18996f52ea12ef7`,
  `43e2947032d6b0c4a28a2196634c0987d4308b7d00b259f85baf720f74283242`,
  and `9f8ada9f72fe6f02ef734ff21a1da802e62e3b535f7a9f7f8e510f8a624b24dc`.
  Linux and actual-Windows focused suites each pass `46/46`; compilation and
  both PowerShell parsers pass.
- No simulator command occurred during N184 development. Candidate 045 and
  FullLap remain unauthorized. Next only: reuse PID `24476` without relaunch,
  recover sole VQ1/R1 through the UI if necessary, send normal disarm followed
  by exactly one learned-target MAVLink `COMMAND_LONG 31000` with confirmation
  and parameters 1--7 zero, require boot rollback `>1000 ms` and a fresh
  nonnegative race-start timestamp, then run command-free Shadow 047. No
  flight is authorized until both proofs are appended.

## N184 Live Reset and Shadow 047 Passed — Candidate 045 Authorized Once

- The existing responsive simulator shipping PID `24476` was reused without
  relaunch. UI recovery restored sole VQ1/R1 at base/status `193/4`, official
  index `0`, finish `-1`, with pre-reset sim boot `1127189 ms` and race start
  `1127854 ms`. The ready snapshot SHA-256 is
  `9accf90b9a4020eb6e2e2bf41cc4eb1cb3445127dd2a44874280ac124e6924c7`.
- One learned-target MAVLink `COMMAND_LONG 31000` reset was sent only after
  normal disarm, with confirmation and parameters 1--7 zero. It was accepted:
  sim boot rolled back `1127189 -> 3146 ms` (`1124043 ms`), a fresh race start
  appeared at `3267 ms`, and the state remained official index `0`, finish
  `-1`, base/status `193/4`, with zero collisions or telemetry dropouts. The
  reset snapshot SHA-256 is
  `d7b3a6bf59d227d9a76d28131b13e35e1231e66cb56c12cdee040340337760f6`.
- Passive Shadow 047 passed with zero setpoint, arm, reset, or disarm commands.
  It replayed `297/297` visible samples, maximum action error
  `2.0943222045488596e-7`, inference `29.653 Hz`, cadence probe
  `603/10 = 60.2 Hz`, `2427` telemetry messages, `118` frames, `117`
  detections, and zero collision, dropout, drain-limit, or rate issue. Shadow
  and parity SHA-256 values are
  `8af0e72e22403ebe048b745d0b96b23f8e85b68fe43c71a98e1c97345713e276`
  and `cc5f14f6be48bf7997944c358b7e6d5f159f2a324d62228ce05ae16c8710d8b9`.
- Exactly one bounded Candidate 045 is now authorized under tag
  `n184_six_gate_hybrid_gate4_bounded_045`: 80 Hz, 45 s maximum, one attempt,
  one detected reset, stop immediately at clean official index `4`, require
  finish `-1`, matching hashes, no collision/invalid/dropout/drain/rate issue,
  exit disarm, and a command-free passive post-stop snapshot. Reject without
  unchanged retry. FullLap remains unauthorized.

## Candidate 045 Rejected at Gate 3 — Offline Robust-Control Screen Next

- Candidate 045 ran exactly once and is rejected without unchanged retry. Its
  normal command-`31000` reset rolled boot `215871 -> 3416 ms`, produced fresh
  race start `3300 ms`, was detected in `0.500 s`, and calibrated `60/60`.
  Official Gates 1--2 passed (last-gate time `7.415724277 s`), then collision
  `1001`, threat `2`, impact `2.798445463180542 m/s` stopped control at
  `10.437 s`, official index `2`, finish `-1`.
- Delivery was healthy: `626/10.312 = 60.609 Hz`, duration `10.437 s`, zero
  rate violation, telemetry dropout, or drain-limit hit, `3273` telemetry
  messages, `1654` IMU samples, `133` frames, `132` detections, and one exit
  disarm. The command-free post-stop is disarmed at base/status `65/3`, retains
  index `2`, race start `3300 ms`, finish `-1`, and `reset_sent=false`; no
  delayed Gate-3 credit appeared.
- The local vision detector declared its third pass at `10.297 s` and then
  selected the small downstream gate visible through the current aperture, but
  official progress correctly remained index `2`. The final reliable state at
  `2.078 m` had right/rate `-0.140/+2.778 m/s`; plane projection was
  right/down `+0.555/-0.378 m`, radial `0.671 m`. Collision followed only
  `0.140 s` after the local vision event. Historical pass/failure projections
  overlap near this nominal radius, so point-center geometry is insufficient;
  a tighter clearance margin must include attitude lag and terminal rate.
- Aggregate/attempt/smoke/CSV/post-stop/diagnosis SHA-256 values are
  `49a51a44c176bf597ec071032fcbdc246dc95c0e191199f0d2bda45fdc2d0e38`,
  `1484210c03ee4e724d314231546f3f9e0041822d9318339d0c44e18625ad8755`,
  `d4d05e7591c88e2d98e67d49ceea743eca94468d60d85905595473eae5816ee9`,
  `06fd425c62a39998f65e5161276a38f08638a8982af26d592e1c7ce46724a743`,
  `e1e935b5c246fa2bcf3dddff9a5a4ad1daeead424539a9f015a5743ba64d6953`,
  and `bcfb8f31da15acedce57caeb943dc3d3d2d5870146e3705f8d217e6eec1d0050`.
- The existing official-trace dynamics fitter now accepts the current 32-field
  ABI while preserving its first-23-field contract. Across Candidates 008--045
  it found `116` continuous segments and `2390` acceleration windows; measured
  roll/pitch lags are `0.566/0.522 s`, consistent with the native plant. Fit and
  fitter SHA-256 values are
  `a41967ca58c2a063dcc4bb9fa97f48f5ce6fd70430b585dc871bb42c6db7fcfc`
  and `b90b5ab13f47c9636dbf7145d1b3aac901acd47c342f438cdecc14c90222114b`.
- No Candidate 046 and no FullLap are authorized. Next only: offline
  preregistration and screening of a smooth phase-based Gate-3 controller with
  a small named parameter vector against the complete official pass/failure
  corpus. Require a tighter robust terminal margin, no non-Gate3 change,
  focused tests, and frozen hashes before any UI recovery, reset, shadow, or
  flight. Do not reopen the retired pitch-brake, terminal-coast, vertical-floor,
  minimum-energy, or one-sample counter-threshold branches.

## N185 Projected-Path PD Surrogate — Rejected Offline

- The 32-field dynamics audit additionally fit `511` Gate-3 lateral samples:
  acceleration gain `4.4573 m/s^2` per `tan(roll)`, rate term
  `-0.1735 /s`, bias `-0.7893 m/s^2`, and RMSE `0.6470 m/s^2`. A bounded
  projected-path PD search then evaluated all `5400` preregistered combinations
  of path slope, terminal offset, position/rate gains, roll cap, and slew cap.
  Candidate 045 was held out from parameter selection.
- The best vector was slope `0.30`, terminal right `-0.10 m`, gains
  `1.0/0.7`, normalized roll cap `0.9`, and slew cap `2.0/s`. It projected the
  Candidate-045 holdout from observed `+0.554 m` to `+0.059 m`, but the model
  failed its own authority gates: recorded-action replay median/p90/max terminal
  error was `0.716/1.699/2.268 m`, and the best controller's worst development
  projection was `2.493 m` against a `0.55 m` limit. Therefore no controller
  change is admitted and N185 is rejected without live shadow or flight.
- Search script/report SHA-256 values are
  `33e5ec1f73037e9986aefc39296fc831c5e6fb3640c21316f33047cda2e3f999`
  and `e0512e57bbfb590082af182ee125903bf81d2b88e6e8cf96717c177c80aee0cf`.
  Candidate 046 and FullLap remain unauthorized. Do not fit around the failed
  surrogate or fly its attractive holdout prediction. The next mechanism must
  first obtain a replay-valid counterfactual model or use a bounded native/live
  evaluator with a small interpretable controller vector; no simulator command
  is authorized by this offline result.

## N186 Native Recorded-Action Replay — Rejected as Authority

- A bounded native evaluator reconstructed the first visible official Gate-3
  state inside `12 m`, replayed recorded pitch/roll/thrust/yaw through the
  official `0.75 m` aperture, and used the frozen official dynamics fit only
  for evaluator plant parameters. Across the preregistered 24-trace corpus at
  four native replicates per trace, it classified `19/24` outcomes but predicted
  every trace as a failure: official-pass recall was `0/5` and official-failure
  recall was `19/19`. Balanced recall gates therefore rejected it even though
  class imbalance put raw accuracy just below `0.80`.
- N186 is not a counterfactual admission authority and selected no controller.
  No FlightSim command, reset, shadow, or flight was sent from this result.
  Evaluator/test/report SHA-256 values are
  `8b58597122eb807c4bec5fad9cd3a864134ee448cb5d22847075ddda1e1a5ae4`,
  `c743ea9082c21fbb00654a88a542317f7ac5db01973b7a3d9b1a229b4729a58f`,
  and `6641dc38edef51efc1d52edd17464d0633c8cdba02ad22ec91b0ff69be4658d0`.

## N187 Bounded Live Evaluator Frozen — Passive Shadow 048 Only

- The exact N184 six-gate hybrid remains the base. A new stateful wrapper may
  change only normalized roll while official Gate 3 is active, visible, closing
  at least `1 m/s`, and inside `12 m`; pitch, thrust, yaw, Gates 1--2, and
  Gates 4--6 remain delegated unchanged. Its single named vector is cubic path
  power `3.0`, terminal right `-0.10 m`, position/rate gains `0.20/1.0`,
  roll cap `1.0`, slew cap `4.0/s`, and nominal control rate `60 Hz`.
- The 24-trace invariant verifier passed with no non-roll or out-of-phase
  changes, no invalid output, one activation per usable trace, and measured
  controller slew no greater than `4.0/s`. On held-out Candidate 045 the
  requested counter and governed nonpositive roll both begin at `5.189 m`;
  governed roll is negative by `3.982 m` and reaches `-1.0` by `2.652 m`,
  advancing unloading relative to the measured `0.522--0.566 s` attitude lag.
  Seven controller/evaluator tests and six runner tests passed.
- Policy/verifier/offline-report/full-runner/deployment-runner SHA-256 values are
  `9f63f47ee3132c76dfb0a987fb97e9d697a50ce990cdb39a6df3ff69a9bbb883`,
  `9a06e2bbbc9628f1712c7975f81536f02e4f06c1f29f0c1a83fd351f0869176f`,
  `1b7900325927106d39da75cfd627f2485f967d9c53c6ff4a27d35af8c3a744ed`,
  `296c6e303a2c88d44fa8844773b23e36f216525ced41af35be3a787e0ec259df`,
  and `7b34dbcd336755d90d74fe63e9d73173c333093f8cfc622ca1a4cdda814266a6`.
- Authorization is observation-only: reuse responsive shipping PID `24476`
  without relaunch; if inactive, recover sole VQ1/R1 through the UI on that same
  PID. After normal disarm, send exactly one learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 zero. Require boot
  rollback greater than `1000 ms`, a fresh nonnegative race start, official
  index `0`, and finish `-1`; then run only passive zero-command Shadow 048
  with the hash-pinned N187 wrapper. The shadow itself must send zero reset,
  arm, setpoint, and disarm commands and must pass parity/cadence/telemetry/
  camera/fault checks. Candidate 046 and FullLap remain unauthorized until that
  shadow and its parity artifact are appended.

## N187 Reset and Passive Shadow 048 — Passed

- Reused the already-running responsive shipping process, PID `24476`; no
  simulator process was launched. The read-only pre-reset snapshot retained
  boot time `3367318 ms` and race-start time `3322 ms`. After normal disarm,
  exactly one learned-target MAVLink `COMMAND_LONG 31000` was sent with
  confirmation `0` and parameters 1--7 zero.
- Reset validation passed: boot time rolled back by `3363713 ms` to `3605 ms`,
  a fresh race-start time of `3296 ms` appeared, official index was `0`, finish
  was `-1`, and base/status remained `193/4`. Pre-reset and reset-summary
  SHA-256 values are
  `e8303bb1590363f0062bc29bd6b735b54156d9f87b4f93bb40066ae7af9657b2`
  and `09000fe72c0f55f2a93a6fea6914f861e0c9c52fee3d4fef0217fc89b5077ed5`.
- Passive Shadow 048 passed. It sent zero setpoints, reset, arm, and disarm
  commands; sustained `60.2 Hz` policy cadence with zero cadence violations;
  observed `2424` telemetry messages, `125` camera frames/detections, and zero
  telemetry, camera, collision, or malformed-message faults. Replay covered
  all `302` inference samples with maximum action error
  `2.2151412965e-7`; parity passed with no blockers. Shadow and parity SHA-256
  values are
  `41e487a8b54ac9a6953c2ce748716f5dfb094288b252b3fd6da52419682a05f0`
  and `c2cf1955c5764d20b5a8e541c34fac26b8aacf359aa748b3e8b95a1a966ac8a2`.
- Exactly one Candidate 046 `BoundedGate3` attempt is now authorized with the
  frozen N187 hashes and parameter vector. Reuse PID `24476`, use the same
  disarm-then-`31000` reset contract, stop and disarm immediately upon official
  index `3`, collision, fault, or timeout, and do not retry. Require a
  command-free post-stop snapshot. FullLap remains unauthorized; the bounded
  result must be appended before any later flight decision.

## N187 Candidate 046 — Failed Without Controller Activation

- The sole authorized `BoundedGate3` attempt reused PID `24476` and sent one
  policy-ready learned-target MAVLink reset. Boot time rolled back from
  `279958 ms` to `3359 ms`; a fresh race-start time of `3270 ms` appeared.
  Gates 1 and 2 passed at `4.594 s` and `7.360 s`, but the run remained at
  official index `2` until the `45 s` bound. It recorded no collision, no
  telemetry dropout, no malformed MAVLink message, and no command-rate
  violation; effective command rate was `60.637 Hz`.
- Replay diagnosis found `329` Gate-3 samples and zero N187 controller
  activations. The reacquired gate began near `28.24 m` forward and about
  `-15.93 m` right; the closest continuous approach remained just outside the
  controller's admission boundary near `12.80 m`, then the target estimate
  jumped back beyond `21 m`. Thus the sustained negative roll was inherited
  from the unchanged N184 base, not produced by the new projected-path PD.
  Candidate 046 is a failed, non-authoritative test and must not be retried.
- The command-free post-stop snapshot sent no reset or control command and
  confirmed disarmed/standby base/status `65/3`, official index `2`, finish
  `-1`, and no collision or telemetry fault. Smoke/attempt/summary/diagnostic/
  post-stop SHA-256 values are
  `db864cf6648c76af64b3b4d81d8b96c052eb935c223d67e2c784edfbb490eeb3`,
  `b1da86aa1aa72d2ce3ce7ee7d646a8182cbbeec0b89329cfdb8a6754f15e7748`,
  `076a530828184406bbee98a1f3cda3f04c43492d5b9457a7d552acbd0cdbafa3`,
  `21278e48d3c7642981267f942afdb6b79f8887a714ab12249f54de99571908e0`,
  and `677ca85aa5ba5b0ebe507274411088d60d994c10139b2469ff7e03a52495546c`.
- No Candidate 047 and no FullLap are authorized. Retire the N187 close-range
  intervention as the next mechanism: it cannot act on this failure mode.
  Next work is offline only and must compare the complete observable
  Gate-2-to-Gate-3 transition in Candidate 045 versus Candidate 046, then freeze
  the smallest causal acquisition/transition change with tests, trace
  invariants, and hashes before any further reset, shadow, or flight.

## N188 Severe Gate-3 Acquisition Guard — Offline Freeze

- A deployed-observation comparison covers all ten Candidate-129 reliability
  traces plus Candidates 045 and 046: eight official Gate-3 passes and four
  failures. Candidate 046 alone entered the evidence-separated severe
  transition state: official Gate 3 visible `15--35 m` forward with at least
  `8 m` absolute lateral displacement. None of the eight passes or the other
  three failures entered it. All eight passes reached the prior close-range
  admission; Candidate 046 did not.
- N188 is the exact N184 base plus one bounded Gate-3-only acquisition guard.
  Normal approaches remain delegated exactly. On the severe state it reuses
  the bounded acquisition primitive: normalized pitch/roll/thrust
  `-0.24/0/0`, absolute yaw toward the visible opening limited to a `0.35 rad`
  step. It releases only within `4.5 m` lateral and `0.12 rad` yaw error, and
  bounds a missing-detection carry to `30` inference ticks. Candidate 046's
  recorded path exercises two acquisition epochs and `244` changed samples;
  the other eleven traces have zero activation and bit-exact unchanged actions.
  There are zero non-Gate3 changes, invalid outputs, or invariant blockers.
- Sixteen focused controller, diagnostic, verifier, and runner tests pass; the
  PowerShell deployment script parses. Comparison-script/comparison-report/
  policy/verifier/offline-report/deployment-runner SHA-256 values are
  `e2a9aef2cb1bf6cee094017a0a106a3e7ff71366a8e33511d3f1b7e6c27cfc6c`,
  `c922773cbd929ad8fb6a73d122425ba023b212bb2a0a69632a037e91de155dbc`,
  `fbbf16d8fc61610f180e914658dac3fd504ef856e4a456b2ba64e9dc394097bd`,
  `5a6b3be0aa922d3a217e3cccdadeb2acd70f027bb360d02b7cacd19d424774e3`,
  `cc8fa216f3e3d7db22d4c107b515f35710a692bf07d9e0edc81f1cee4dc3a880`,
  and `d9642d21148931ed93009cc69b45758008de3b6a5debfd31b0f5298cae34324e`.
- This is not counterfactual flight proof. Authorization is limited to reusing
  responsive PID `24476`, normal disarm, exactly one learned-target MAVLink
  `COMMAND_LONG 31000` reset with confirmation and parameters 1--7 zero, and
  passive zero-command Shadow 049 with the hash-pinned N188 callable. Require
  boot rollback greater than `1000 ms`, fresh nonnegative race start, index
  `0`, finish `-1`, then exact manifest/parity, cadence, telemetry, camera, and
  fault checks. Shadow itself must send zero reset, arm, setpoint, and disarm
  commands. Candidate 047 and FullLap remain unauthorized until the reset and
  shadow artifacts are appended.

## N188 Reset and Passive Shadow 049 — Passed

- Reused responsive shipping PID `24476`; no simulator process was launched.
  The read-only pre-reset state was active at index `0`, finish `-1`, boot
  `709104 ms`, and race start `3355 ms`. Normal disarm followed by exactly one
  learned-target MAVLink `COMMAND_LONG 31000` reset rolled boot back by
  `706333 ms` to `2771 ms` and produced fresh race start `3283 ms`, index `0`,
  finish `-1`, base/status `193/4`.
- Passive Shadow 049 passed and sent zero reset, arm, setpoint, and disarm
  commands. No-op cadence was `578/10 = 57.7 Hz` with zero violations. It
  observed `2428` telemetry messages and `123/123` detected camera frames with
  zero collision, dropout, drain hit, malformed message, decode failure, or
  missing quad. All `303` inference samples replayed at `30.3 Hz` with maximum
  action error `1.9858703615e-7`; every deployment manifest hash matched and
  parity had no blocker.
- Pre-reset/reset/shadow/parity SHA-256 values are
  `032b0069d89bce198ce349124149ba28a13c963a4446c2ade68403aa667dd9e1`,
  `3f7b0b7bd186a3e68f45f39c46ba4f7cf3dc964d63809a63de7b05c966c3f8f5`,
  `083d2019e9b2f11ca5b10d7ab43b9089ac6cabd529d4c7545bb57b834939e733`,
  and `eeb854c10b91c54ffb821641ac8ad4fa832c80a1cbbcf9620ab9e4d621dcd0a6`.
- Exactly one Candidate 047 `BoundedGate3` attempt is authorized with the
  hash-pinned N188 callable. Reuse PID `24476`; use the same normal-disarm then
  command-`31000` reset proof; stop and disarm immediately on official index
  `3`, collision, fault, or timeout; do not retry. Capture a command-free
  post-stop snapshot. FullLap remains unauthorized until the bounded result is
  appended and evaluated.

## N188 Candidate 047 — Invalid Pre-Treatment Gate-2 Contact

- Candidate 047 ran once on reused PID `24476`. Its policy-ready reset was
  detected (`157889 -> 3394 ms`, fresh race start `3297 ms`). It passed Gate 1,
  then contacted frame ID `1001` while official Gate 2/index `1` was active at
  about `7.3 s`; impact was `6.287686`. The run aborted immediately, was marked
  crash/invalid, and had no telemetry, camera, or rate fault (`448` commands,
  `61.528 Hz`, zero violations/dropouts/drain hits/malformed messages).
- The policy trace contains `22` Gate-2 samples and zero Gate-3 samples, so the
  N188 severe-acquisition guard never activated. The command-free post-stop
  snapshot sent no reset/control command and confirmed disarmed/standby
  base/status `65/3`, finish `-1`; official index advanced to `2` only after
  the damaging contact/abort and is not a valid Gate-2 pass.
- Smoke/attempt/summary/post-stop/recent-prefix-audit SHA-256 values are
  `a89f19871e0ceadad0393194f1aed9480adcb531a4854cb45ecb32de2c0e2e2f`,
  `e3cd9b876178200f76b2f68614a5d8c6c9b752ef6a6fa11b80b9f466ac18fdc9`,
  `26a24637147d95bac12086767308c459dbb4f15c0986677c28b1f3946c937cf8`,
  `13ea2a54877394f36b2ceba5a299ffe6950af325d401c1c6542b40b78b6328d7`,
  and `3999ee861864d4bfe2ebab39a18b51fddb87e94c76decf708ce8e678d30bad1a`.
- Candidate 047 is rejected and must not be replayed as a valid result. Because
  it ended before treatment exposure, while each of Candidates 037--046 reached
  official index `2`, exactly one replacement Candidate 048 `BoundedGate3`
  attempt is authorized with the unchanged N188 policy/runner hashes and the
  already-passing Shadow 049. Reuse PID `24476`, perform the exact disarm then
  command-`31000` reset proof, stop/disarm on index `3`, collision, fault, or
  timeout, and do not retry. FullLap remains unauthorized.

## N188 Candidate 048 — Rejected at Gate 3 Without N188 Activation

- Candidate 048 ran once on reused PID `24476`; the simulator was not
  relaunched. After same-process VQ1/R1 recovery, normal disarm followed by the
  learned-target MAVLink `COMMAND_LONG 31000` reset rolled official boot time
  `258464 -> 3419 ms` and produced fresh race start `3297 ms`. Gates 1 and 2
  passed, but collision ID `1001`, impact `5.980898857`, stopped the run at
  about `10.65 s`, official index `2`, finish `-1`.
- Transport remained healthy: `647` commands over `10.532 s` (`61.337 Hz`),
  with zero command-rate violations, dropouts, drain hits, or malformed
  messages; the run observed `3304` telemetry messages and `135` camera
  detections. The command-free post-stop snapshot sent no reset or control
  command and confirmed disarmed/standby base/status `65/3`, index `2`, and
  finish `-1`.
- This was a normal close Gate-3 approach, not the severe far-offset state.
  The N188 diagnostic reports exactly zero activations. The last accepted pose
  before collision was approximately `1.875 m` forward, `-0.117 m` right,
  `-0.135 m` down, with yaw error `-0.062 rad`. The unchanged base held full
  normalized roll through about `4.44 m` forward and unloaded only near
  `2.56 m`; the likely remaining failure is late bank/vehicle-frame clearance,
  not centerline acquisition.
- Smoke/attempt/summary/post-stop/activation-diagnostic SHA-256 values are
  `d96784ca515ba1bfc286408e2e8f190f3635d93c8720dfc04f145140109f07ab`,
  `c20e265b6c3450117126d3771a3b453b3ca8158fb43aecb9fb93055328834fff`,
  `9a411f6562bf31a2124096f23bee66ac2cbffbbbbef6b5272eca8e87867bc141`,
  `0cf86ee823694c21522c1848ffa6f760275bca7e139521f97684acbd655c7896`,
  and `07712276de79e4862ac70d8d5ff8b62426f2ee68f96cf38593fd01014e4a19b5`.
- Candidate 048 is rejected with no unchanged retry. N188 is retained only as
  the evidence-separated severe acquisition branch; it is retired as the sole
  next live mechanism because it did not treat Candidate 048. No Candidate 049
  and no FullLap are authorized. Next work is offline only: compose the frozen
  N188 far-offset branch with the frozen N187 close projected-path roll PD,
  prove branch precedence and bounded behavior on recorded traces, then require
  a separately logged command-`31000` reset and zero-command shadow before any
  bounded flight.

## N189 Gate-3 Acquisition/PD Composition — Offline Freeze

- N189 is the smallest composition supported by the two observed Gate-3
  failure modes. It evaluates the exact N184 recurrent base once. The frozen
  N188 far-offset acquisition branch has strict precedence; after a coherent
  release, the frozen N187 projected-path PD may replace roll only on the next
  visible, closing Gate-3 sample inside `12 m`. Loss of visibility or passage
  behind the gate resets PD and delegates to the base, preventing an indefinite
  stale-roll hold. No scalar in either frozen controller was retuned.
- Replay covers all ten Candidate-129 reliability traces plus Candidates 045,
  046, and 048: eight official Gate-3 passes and five failures. Candidate 046
  has exactly two severe activations and zero PD activations. Candidate 048 has
  zero severe activations and one PD activation; PD first requests counter-bank
  at `4.949 m`, reaches nonpositive roll at `4.444 m`, and stays within the
  `4.0 normalized-roll/s` slew cap. Candidate 045 likewise takes PD once. The
  corpus has zero non-Gate3 changes, invalid output, PD non-roll change,
  pre-admission change, branch-overlap blocker, or slew violation.
- Linux and actual Windows each pass `18/18` focused controller, composition,
  verifier, and runner tests; the PowerShell deployment runner parses. The
  Candidate-048 standalone replay/composite policy/composite verifier/offline
  report/deployment-runner SHA-256 values are
  `d97ac996d16fdfa4f2ab1ae50bf8c88145fa7c7224407fcc578653b934031226`,
  `ce0bdde0ec4d2cb3b46df0ee35d1ede3f805391899fa24ba19b2df756d42b267`,
  `085b671a625aabee2ce51847b73046cf2987ce1f595c64b00390b6bdc21a2bb1`,
  `be4d9d7bd13be22eca34285e21539417642fed1270ae83e72702d57743ab5c3e`,
  and `9d8fe0f2a93d22a855c52c86734aefd82d9d17f6ae7f94dd040d4ff8ba218a34`.
  Frozen N187/N188 child hashes remain `9f63f47e...` and `fbbf16d8...`.
- No simulator command occurred during N189 development. Candidate 049 and
  FullLap remain unauthorized. Next only: reuse responsive shipping PID
  `24476` without relaunch; recover sole VQ1/R1 through the UI only if the
  inactive state ignores reset; send normal disarm followed by exactly one
  learned-target MAVLink `COMMAND_LONG 31000` with confirmation and parameters
  1--7 zero; require boot rollback greater than `1000 ms`, fresh nonnegative
  race start, index `0`, and finish `-1`; then run passive zero-command Shadow
  050 with `-Gate3CompositeAcquisitionPD`. The shadow itself must send zero
  reset, arm, setpoint, and disarm commands. Only a fully passing hash/parity,
  cadence, telemetry, camera, and fault report may authorize one separately
  preregistered bounded Candidate 049.

## N189 Live Reset Proof and Passive Shadow 050 — Passed

- The preserved simulator remained responsive at PID `24476`; no process was
  launched. A command-free snapshot confirmed Candidate 048's inactive black
  screen at base/status `65/3`, index `2`, finish `-1`. The startup helper used
  `-NoRelaunch -SkipLoginClick` to recover sole VQ1/R1 on the same PID, yielding
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, and old boot/race
  epoch `1005540/989043 ms`.
- One diagnostic invocation from the stale Windows clone sent command `31000`
  without the required preceding disarm. v3385 ignored it: boot continued to
  `1034858 ms` and race start stayed `989043 ms`. It is retained as a failed
  precondition artifact and was never counted as a reset. The current tracked
  reset path then learned the target IDs, sent normal disarm, waited `100 ms`,
  and sent one `COMMAND_LONG 31000`, confirmation/parameters zero. That exact
  sequence succeeded: boot rolled back by more than one million milliseconds
  to `3078 ms`, with fresh race start `3285 ms`, index `0`, finish `-1`,
  base/status `193/4`, visible Gate 1, and healthy telemetry.
- Passive `n189_gate3_composite_acquisition_pd_shadow_050` passed and sent zero
  reset, arm, MAVLink setpoint, and disarm commands. No-op cadence was
  `601/9.969 = 60.187 Hz` with zero violations. All `298/298` visible inference
  samples replayed at `29.8 Hz` with maximum action error
  `2.1009979248e-7`; all deployment hashes matched. It observed `2413`
  telemetry messages and `120/120` camera detections with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- Pre-inactive/recovery-ready/stale-reset/corrected-reset/shadow/parity SHA-256
  values are
  `80d9700e3756bacebed4db4811a53dcfb51a505d477ef29e49bf326f084fc554`,
  `6c2485fae6a8608040f40b0530984e9932d9895ac3bcf1f65c5e0bb1b82fe48b`,
  `736c969013f0c34bf36dfd4af11fa0c88c193ca866a99696f068e4942a74013e`,
  `faa9de5a308a026cd5316768fd702046098cbfba842d975fdeb3550211e472e9`,
  `24fbff3a711768fb52a84ab44754a7189855d32cb1614a831bb3264cf41476aa`,
  and `7398aa2be25bfd990cc69baad9bba51d68eeda26dec34ed805804e3cb9e3149f`.
- Authorize exactly one Candidate 049 `BoundedGate3` under tag
  `n189_gate3_composite_acquisition_pd_bounded_049`: hash-pinned N189 callable,
  actual Windows Python, reused PID `24476`, one detected policy-ready
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official index
  `3`. Accept only a clean index-3 stop with finish `-1`, actual cadence in
  `[50,100) Hz`, exact hashes, zero rate/collision/invalid/dropout/drain issue,
  one exit disarm, and command-free passive post-stop proof. Any failure rejects
  Candidate 049 without unchanged retry. FullLap remains unauthorized.

## N189 Candidate 049 — Invalid Pre-Treatment Gate-2 Contact

- Candidate 049 ran once on reused PID `24476`; no simulator process was
  launched. Its exact disarm-then-command-`31000` reset was detected
  (`215831 -> 3407 ms`, fresh race start `3307 ms`, detection `0.516 s`,
  calibration `60/60`). It passed Gate 1, then contacted frame ID `1001` at
  about `7.313 s` while official Gate 2/index `1` was active. Impact was
  `3.207263947`; the runner stopped and disarmed immediately with finish `-1`.
- Transport was healthy: `433` commands over `7.219 s` (`59.842 Hz`), zero
  rate violations, telemetry dropouts, drain hits, or malformed messages. The
  command-free post-stop snapshot sent no reset/control command and confirmed
  disarmed/standby base/status `65/3`, finish `-1`. As in Candidate 047,
  official index changed to `2` only after the damaging contact/abort; this is
  not accepted as a valid Gate-2 pass.
- The policy trace contains `63` usable samples and zero Gate-3 samples. The
  N189 diagnostic confirms zero severe and zero projected-path activations, so
  Candidate 049 supplied no treatment evidence about N189. Smoke/attempt/
  summary/post-stop/activation-diagnostic SHA-256 values are
  `60af26d4c9d3be10a26100f780ea5739dc96e776921e513a785bb40b5f721e89`,
  `c18b4430a5273effb3b6a41dae8e8a0a775fd972c8921fc0d187f978f8201cd1`,
  `936a6019fa4000ba07c80850b6cad587b11f018ab3815d076fb7b1bac6d19c23`,
  `a91ed8443d0db0add681b18027503aa824995e12848db698a6c2a463c58926d7`,
  and `dc513300f03284dd2e2ca0167acd2467d2083ead3b46e281d7717b4925195a00`.
- Candidate 049 is rejected and must not be retried unchanged. No Candidate
  050 and no FullLap are authorized. Next work is offline only: compare the
  complete observable Gate-2 approaches in the two recent contact runs
  (Candidates 047 and 049) against Candidate 048 and recent clean transitions,
  then freeze the smallest Gate-2 reliability mechanism while retaining N189
  unchanged. No reset, shadow, or flight may occur before tests, trace
  invariants, hashes, and a new authorization are appended.

## N190 Stateless Close Gate-2 Vertical Margin — Offline Freeze

- N190 retains N189 source-hash exact and adds one Gate-2-only stateless rule.
  On a visible Gate-2 sample inside `6 m`, it projects current down position to
  the gate plane using a `3 m/s` closing-speed floor. If the projection is
  below `-1.1 m`, normalized thrust is floored at the already validated
  moderate value `+0.3`. It changes no pitch, roll, or yaw, preserves stronger
  thrust, has no latch, and releases immediately when projection recovers.
- Evidence covers Candidates 037--049: eleven clean Gate-2 transitions and the
  two recent contact paths. Both Candidates 047 and 049 receive exactly two
  earlier weak-thrust changes; Candidate 048 remains bit-exact. Five clean
  transitions enter the conservative margin rule, but only weak thrust is
  raised. Across the corpus there are zero non-Gate2 changes, non-thrust
  changes, stronger-thrust changes, invalid outputs, or invariant blockers.
  This is bounded trace evidence, not counterfactual pass proof.
- Linux and actual Windows each pass `23/23` focused N187--N190 controller,
  verifier, composition, and runner tests; the PowerShell runner parses.
  Recent-transition-comparison/policy/verifier/offline-report/deployment-runner
  SHA-256 values are
  `ddcb9d434e6801f4132c00e48d3bba83aca8dc7b2722ce79cc64ba510c00caa0`,
  `70292925b0748e959eb654febf994afd623143b319bb241ec2e93312ef9e24fb`,
  `eebfb83561a85f2cd7342486b19fb19f49d08417499fce399811834da1a2d1ca`,
  `4841d3375a59df6a0037f354b6865201c89c3f3af897f4fd72d7544777dc740b`,
  and `8f4a654ccdc5a8fdb3db94729c6555293d0d00fe6ad0d4713a0ad80a39bf1921`.
- No simulator command occurred during N190 development. Candidate 050 and
  FullLap remain unauthorized. Next only: reuse PID `24476` without relaunch,
  recover sole VQ1/R1 through the UI because the post-contact state is inactive,
  then use the current tracked normal-disarm plus learned-target MAVLink
  `COMMAND_LONG 31000` path with confirmation/parameters zero. Require boot
  rollback over `1000 ms`, fresh nonnegative race start, index `0`, finish
  `-1`, and visible Gate 1; then run passive zero-command Shadow 051 with
  `-Gate2VerticalMarginN189`. Shadow must send zero reset/arm/setpoint/disarm
  and pass exact hashes, parity, cadence, telemetry, camera, and fault checks
  before any separately preregistered bounded flight.

## N190 Live Reset Proof and Passive Shadow 051 — Passed

- Reused responsive PID `24476` throughout; no process launch. The inactive
  Candidate-049 state was recovered through the sole VQ1/R1 UI with
  `-NoRelaunch -SkipLoginClick`, producing base/status `193/4`, index `0`,
  finish `-1`, and old boot/race epoch `848754/849437 ms`. The current tracked
  disarm-first learned-target `COMMAND_LONG 31000` sequence then rolled boot to
  `3005 ms` and produced fresh race start `3268 ms`, index `0`, finish `-1`,
  visible Gate 1, and healthy telemetry.
- Passive `n190_gate2_vertical_margin_n189_shadow_051` passed and sent zero
  reset, arm, MAVLink setpoint, and disarm commands. No-op cadence was
  `560/9.703 = 57.611 Hz`, zero violations. All `238/238` visible samples
  replayed at `23.616 Hz`, maximum action error `2.7079429626e-7`, with every
  deployment hash exact. Telemetry/camera totals were `2676` messages and
  `118/118` detections, with zero collision, dropout, drain hit, malformed
  message, decode failure, or missing quad.
- Recovery/reset/shadow/parity SHA-256 values are
  `a8e387728798e4b2c83e532a8919b4cdb287881c8c26b0917d9d544861eb7191`,
  `29b647c75ec5a9d25849fa460e98df81b9c501738741c05cd1bab5637835e06e`,
  `16306c0c5b376c5e16b91bb2ac56e00c6dcb563d5edf7e0ed6e888611de4f6d3`,
  and `0de820a3290be8aef4b6a3d6e70b92d0c8d03d95df2a390302ea633ffd28751e`.
- Authorize exactly one Candidate 050 `BoundedGate3` under tag
  `n190_gate2_vertical_margin_n189_bounded_050`: frozen N190 hashes, Windows
  Python, reused PID `24476`, one detected disarm-then-command-`31000` reset,
  requested `80 Hz`, maximum `45 s`, repeats/min-valid `1/1`, target/minimum/
  authoritative stop official index `3`. Accept only clean index `3`, finish
  `-1`, cadence `[50,100) Hz`, exact hashes, zero rate/collision/invalid/
  dropout/drain issue, one exit disarm, and command-free post-stop proof.
  Reject failure without unchanged retry. FullLap remains unauthorized.

## Superseded N196 Shadow 057 Authorization Snapshot

- Supersede every earlier next-action note. Candidate 056 passed official
  Gates 1--2 but collided at Gate 3 after the N192 terminal-level branch
  latched at `5.367 m`; Gate 4 never activated. Candidate 055 remains the clean
  official Gates-1--3 milestone. Candidate 056 must not be retried unchanged.
- N196 retains exact N195 except for moving the Gate-3 terminal-safe-level
  admission from `6.0 m` to `3.7 m`. Offline replay preserves all protected
  paths and Candidate 051's three corrections, leaves Candidate 055 inactive,
  and reduces Candidate 056's latch from five samples to two. Policy/verifier/
  evidence/runner hashes are `3e2fb526ae01...`, `f0978810ba51...`,
  `c53ddc068a47...`, and `82668d40b44f...`.
- Same-PID recovery, the exact disarm-first MAVLink `COMMAND_LONG 31000`
  reset, and passive zero-command Shadow 057 have passed. The single next live
  interaction is Candidate 057 `BoundedGate3` under the frozen N196 wrapper,
  with target/minimum/stop official index `3`. Candidate 058 and FullLap remain
  unauthorized.


## N196 Reset-31000 Proof and Passive Shadow 057 — Passed

- Responsive shipping PID `24476` was preserved throughout and never
  relaunched. A command-free precheck found Candidate 056's inactive
  base/status `65/3` state with race start `-1`. The documented
  `-NoRelaunch -SkipLoginClick` recovery restored the sole VQ1/R1 session on
  the same PID to base/status `193/4`, official index `0`, finish `-1`,
  visible Gate 1, and old boot/race epoch `1118326/1096556 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `1118326 -> 3276 ms`; a fresh race start
  appeared at `3282 ms`. A later command-free proof showed boot `31091 ms`,
  base/status `193/4`, index `0`, finish `-1`, `136` Gate-1
  detections, and no collision.
- Passive `n196_gate3_later_terminal_level_n195_shadow_057` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `589/10.016 = 58.706 Hz`, with zero rate violations. All `251/251`
  visible samples replayed at `25.022 Hz` with maximum action error
  `2.3526687623065534e-7`; every deployment hash matched. It observed
  `2426` telemetry messages, `1248` IMU samples, and `115/115` camera
  detections, with zero collision, telemetry dropout, drain hit, malformed
  message, decode failure, or missing quad.
- Inactive-precheck/UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity
  SHA-256 values are
  `0295072468ac6270eaca45fe75341dd28475630018a333bf6f7d786148a65477`,
  `fca195732348c4b796579a73e4520ff1e9c6fddbd9b32303b4664ab014527796`,
  `a93f6259adede0a7b2ee9f4414c5dc9fab6cb705d8a597fd1c743fd0affb0940`,
  `5ad9b57579de8d94bf7a49cdff33c3638387b1bbbdcac475e73c29db5e439cfe`,
  `46cb97682a12cdaaac0948c3ed20b1943fed7427f26381d4984f1ae27a8cc2e4`,
  `58b49e1f3c94231eda5bc6d67c328ad64fc02691609ba8b80e6a013e92cdbb28`,
  and `28a174a92a47507cf7ae65dee865e25f7bf9c5b9ab33abe3ec622b8d1686a7d2`.
- Authorize exactly one Candidate 057 `BoundedGate3` under tag
  `n196_gate3_later_terminal_level_n195_bounded_057`: frozen N196 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean official index `3`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 058 and FullLap remain unauthorized.


## N196 Candidate 057 — Collision-Free Gate-3 Vision-Loss Timeout

- Candidate 057 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.516 s`: boot rolled `179671 -> 3410 ms`, fresh race
  start was `3284 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially; authoritative last-gate race time was
  `7.482846260 s`. Gate 3 did not pass. The attempt reached its exact
  `45 s` bound at official index `2`, finish `-1`, with no collision or
  invalid state. Delivery was healthy: `2714/44.844 = 60.499 Hz`, zero rate
  violations, `11094` telemetry messages, `5602` IMU samples, `569`
  frames, `169` detections, zero telemetry dropout/drain/malformed/decode
  fault, and one exit disarm.
- The last detection aged to `30.938 s` by run end. The legal raw-confidence
  observation slot 30 became exactly zero at `14.313 s` and remained zero
  for `281` consecutive eligible Gate-3 samples. Its fifth sample was
  `14.750 s` at forward/closing/right/down
  `2.822/7.052/+0.995/-0.859 m` and yaw error `+0.339 rad`; deployed
  action remained approximately pitch `+0.093`, roll `-1.0`, thrust
  `+0.761`, yaw `-0.00052`.
- N196 later terminal-level admission had zero activations, so Candidate 057
  does not falsify the N196 threshold. N194's frozen-observation recovery also
  had zero activations: sub-micro estimator drift repeatedly exceeded its
  `1e-7` equality test even though detector confidence stayed exactly zero.
  This separates raw confidence persistence as the simpler observable fault
  signal.
- The command-free post-stop proof sent no reset or control command and showed
  base/status `65/3`, official index `2`, finish `-1`, and no collision.
  Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `96d3c52ebdb5e4d1d7b4a4ce516fa56529ff0feca417821c163a8a62f6bc543b`,
  `a3d8e9ffb763a8893ea14ec10c22a957d58d862557ded128281e3f4236ecbf5d`,
  `f12418f224a9f02140b7cc6e502e4905d42fde9315c9d9543a1ed5aed8905e3a`,
  `34058c04566520e7d2e962008f66ae43c29b78371d0b2c94dd3f2f63ebaa0e16`,
  `aaa255b024ca128b3d36be2ab61357d15563ae4dd348c58b371249113fe21e29`,
  `915a22b622d9f1bd1ecacc952102f850ff472d4648689958b4c2f999f17ee6b8`,
  and `c8db2071ff882096255c3803f81c3ee3697b2f0842d8b6ee44a05c1ef2d7567f`.
- Candidate 057 is rejected without unchanged retry. Candidate 058 and FullLap
  remain unauthorized pending a source-hashed, offline-separated raw-
  confidence recovery, focused tests, exact reset proof, and passive shadow.

## N197 Gate-3 Persistent Zero-Confidence Recovery — Offline Freeze

- N197 evaluates exact source-pinned N196 once, then adds one Gate-3-only
  controller. While official Gate 3 is visible, within `3.0 m`, closing at
  least `1.0 m/s`, and legal raw-confidence field 30 is exactly zero for five
  consecutive inference samples, it sets only normalized roll to `0.0` and
  commands the existing bearing-signed `0.25 rad` yaw scan. Pitch and thrust
  remain exact. Any positive confidence, ineligible pose, visibility loss, or
  official transition releases the recovery immediately.
- Source-hashed replay covers every Candidate 016--057 trace. Only the three
  long Gate-3 vision-loss stalls activate: Candidate 034 once for `259`
  samples beginning `16.437 s`, Candidate 053 once for `272` samples
  beginning `15.047 s`, and Candidate 057 once for `277` samples beginning
  `14.750 s`. Every other trace remains inactive, including protected clean
  Gate-3 paths 016/018/020/021/022/055 and Candidates 045/048/051/056.
  Every active sample changes only roll/yaw, preserves pitch/thrust exactly,
  and remains finite/in range. Candidate 057's command-free official post-stop
  proof is embedded.
- Linux focused N189--N197 controller/verifier/runner tests pass `42/42`;
  actual Windows focused controller/runner tests pass `34/34`; compilation
  and the PowerShell deployment parser pass. Policy/verifier/offline-report/
  runner SHA-256 values are
  `ca2ceb12ecbc62a721bf3f34b7cd6b2675592626604899153c3ab7b92754b210`,
  `d2fc0816f3fe7e615d7d56e263e3368e8fbbf7ec0b221760928adcd7845c0ac9`,
  `a54361e1d2108cf9171086abde3e75d3f68ac8d5213dd2f73cf1a6c7794974d8`,
  and `7365cd84565d52f225834cc5252c79bf5ae6a9ce04a0325dd6e25ef8d409f526`.
- No simulator command occurred during N197 development. Candidate 058 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` because Candidate 057
  stopped at base/status `65/3`; then send normal disarm followed by exactly
  one learned-target MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Require boot rollback greater than `1000 ms`, fresh
  nonnegative race start, index `0`, finish `-1`, and visible Gate 1. Then
  run passive zero-command
  `n197_gate3_zero_confidence_recovery_n196_shadow_058` with
  `-Gate3ZeroConfidenceRecoveryN196`. Only a fully passing hash/parity/
  cadence/stream/fault report may authorize a separately preregistered bounded
  Candidate 058.


## N197 Reset-31000 Proof and Passive Shadow 058 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 057's base/status `65/3` post-stop was recovered through the sole
  VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` on the same PID. Command-free
  readiness showed base/status `193/4`, index `0`, finish `-1`, visible
  Gate 1, and old boot/race epoch `537019/514210 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `537019 -> 3077 ms`; fresh race start
  was `3302 ms`. A later command-free proof showed boot `10092 ms`,
  base/status `193/4`, index `0`, finish `-1`, `126` Gate-1
  detections, and no collision.
- Passive `n197_gate3_zero_confidence_recovery_n196_shadow_058` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `597/9.968 = 59.791 Hz`, zero violations. All `308/308` samples
  replayed at `30.8 Hz` with maximum action error
  `2.2283401487910304e-7`; `307` were visible and every deployment hash
  matched. It observed `2434` telemetry messages, `1254` IMU samples, and
  `124` frames/`121` detections, with zero collision, dropout, drain hit,
  malformed message, or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `1b5925a554846b13a41effbbeb381f3f24c9872fe4f7db48bb83108a7dba68e7`,
  `d70708e4b7636dd0d61e752bd65fa4cabeaf4056494fb615bfa094812ac6ad71`,
  `907af66f4fb19c556a0be5ac109203794e56acb2b126404c724ef66ea089f57e`,
  `75a663bfd9465378090231d854ebc1f9d97a918ece9e069c4d792b604410f4e9`,
  `704f52f37238e3888ff584e823d0c2247e556400a4d1ea154af25d012c8f6cfd`,
  and `080938e3a37bbe6bb8655efcb2dd47c5562d21585015cb39a82b1cee8eb997fb`.
- Authorize exactly one Candidate 058 `BoundedGate3` under tag
  `n197_gate3_zero_confidence_recovery_n196_bounded_058`: frozen N197 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 059 and FullLap remain unauthorized.


## N197 Candidate 058 — Clean Official Gates 1--3 Milestone

- Candidate 058 ran exactly once on reused responsive PID `24476`; no process
  launch. Its disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `127600 -> 3412 ms`, fresh race
  start was `3299 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.867674827 s`; the
  configured index-3 stop fired at controller elapsed `10.812 s`, and finish
  remained `-1`. There was no collision or invalid state.
- Transport and perception were healthy: `648/10.718 = 60.366 Hz`, zero
  command-rate violations, `3341` telemetry messages, `1688` IMU samples,
  `137` frames/`131` detections, and zero telemetry dropout, drain hit,
  malformed message, or decode failure. Offline replay shows zero N197
  zero-confidence activations, so this clean run used the unchanged N196 path
  and supplies no live treatment evidence.
- The command-free post-stop proof sent no reset/control command. After exit
  disarm, v3385 cleared the race epoch to start `-1`/index `0`, retained
  finish `-1`, and reported no collision. Reuse-check/probe/smoke/attempt/
  summary/CSV/post-stop SHA-256 values are
  `d3d10646cc68698a18bd454a1280816bad4e688333ef8d0fea9871b651f03b6b`,
  `5adf98828160a043b08afc3dd9472fd9c968403f1893f152e7016845c2c8d932`,
  `94651873c6db9abbe4ea02fce7755115ae02f22f1f61585a2bc8e31c9965f6d9`,
  `cffcdaa916ebe2ff79fa517c8d9d60110b30b1e648e55d0a3ed8da659ec14ce3`,
  `192059e1849366260cca0c1220a33e0c81f852eaa8d528a543fbb57b38d15606`,
  `a8ed667531fc0465f540df3a6d514d0efa7035e22aa11dda57c23c5d1d1287b1`,
  and `059b103ea3450793be326e1367bbe4da9928b7e80a45b68502f628c4d55606c8`.
- Candidate 058 is accepted only as a bounded official Gates-1--3 milestone.
  Authorize exactly one Candidate 059 `BoundedGate4` under tag
  `n197_gate3_zero_confidence_recovery_n196_gate4_bounded_059`: no policy,
  checkpoint, code, or deployment-hash change from passed Shadow 058/Candidate
  058; actual Windows Python, reused PID `24476`, one detected normal-disarm-
  then-command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/
  min-valid `1/1`, and target/minimum/authoritative stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 060 and FullLap remain unauthorized.


## N197 Candidate 059 — Rejected at Official Gate 3

- Candidate 059 ran exactly once as the authorized `BoundedGate4` attempt on
  reused responsive shipping PID `24476`; no process launch. Its disarm-first
  learned-target MAVLink `COMMAND_LONG 31000` reset was detected in `0.5 s`:
  boot rolled `122560 -> 3423 ms`, fresh race start was `3304 ms`, index
  was `0`, finish was `-1`, and calibration completed `60/60` samples.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.355759143 s`. A camera-side detection did not become an official Gate-3
  pass: authoritative index remained `2`, and collision `1001`/threat `2`
  occurred at controller elapsed `9.776438713 s`. Finish remained `-1` and
  Gate 4 was never active. Candidate 059 is rejected with no unchanged retry.
- Transport remained healthy: `640` commands over `10.375 s = 61.59 Hz`,
  zero rate violations, `3289` telemetry messages, `1661` IMU samples, and
  `133/133` frames/detections, with zero dropout, drain hit, malformed
  message, decode failure, or no-quaternion sample. Exit disarm count was one.
- N197 zero-confidence recovery never activated. N196 terminal roll leveling
  activated for exactly two samples: first at `10.250 s`, forward `3.221 m`,
  projected right/down `+0.0048/-0.444 m`; then at `10.375 s`, forward
  `2.232 m`, projected right/down `+0.0873/-0.236 m`. Both forced roll
  to zero. The lower controller's exact counterfactual output was not logged,
  so no stronger causal claim is made.
- Command-free post-stop sent no reset/control command and showed base/status
  `65/3`, race start `-1`, index `0`, finish `-1`. Its small passive ground
  contact after disarm does not replace the in-run impact. Reuse-check/probe/
  smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `3b9fa82c0103c917cb42fb4662d167f054186552fc152f1a07f03bc7b039f3f8`,
  `3832c67eb3e59e9736e3e273036b21f49522865d895cc66306f1ac47173aa071`,
  `40f7ee4a867b6207e35d3eefedd6da6f7319ee26dfd07f2a96195dffb0486aa0`,
  `f21c652b5debfa808d4e28540b1d1e7b9fce7331f9f31caac988bb1d546b0d6f`,
  `71debead7f6ce41d00d3ca1869446d4268623e086fce31a4d74134cb6b1bf017`,
  `1a3bf9f78ff3d97e094c09c961db0b86cf9e24d45485238fd5fc9976de2a4548`,
  and `06aa6932eabb3ab3866826448a79bf2f69d768b56d6da2c5acba38d2fc3be161`.

## N198 Gate-3 Final-Only Terminal Level — Offline Freeze

- N198 is an exact source-pinned re-composition of N197 with one change:
  Gate-3 terminal roll leveling may first latch only at forward distance
  `<= 2.0 m`, instead of `<= 3.7 m`. Projection bounds, minimum closing
  speed, level-roll value, latch behavior, all other controllers, checkpoints,
  and the six-gate course contract remain unchanged.
- Source-hashed replay over every Candidate 016--059 flight trace passes with
  no blockers. The clean Candidate 021 behavior remains active for three
  samples beginning at `1.771 m`; clean Candidates 055 and 058 remain
  inactive. Failed Candidates 051, 056, and 059 change from `3/2/2` terminal
  samples to `0/0/0`. New activations are a strict subset of old activations,
  all protected clean recorded actions remain exact, only roll can change,
  and every output is finite/in range.
- Focused dependency/policy/runner tests pass `30/30` on Linux and `30/30`
  under the actual Windows Python. The PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `cfd9308910dbe63a857a67a3b96d53204a2ad929c9ad9b1a93787ba9392e0724`,
  `48ceb6b6e36a6c56501b799e27e9689e016cf274a2ab06c75bdcbad3a10f2f58`,
  `8c31a2e2463dd14afc8c1fd3bb8a422890c07943086abe7c7041e4cb1b9ef278`,
  and `3b446631b4cce0a5eb9ed2c8264fcb33a5f10a56c844484ac8f0466d18e7e770`.
- No simulator command occurred during N198 development. Next only: preserve
  PID `24476`; recover the sole VQ1/R1 UI on that PID with
  `scripts/start_windows_aigp_race.ps1 -NoRelaunch -SkipLoginClick`; send
  normal disarm, wait `100 ms`, then exactly one learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 all zero. Require
  boot rollback greater than `1000 ms`, a fresh nonnegative race start,
  official index `0`, finish `-1`, and visible Gate 1. Then run passive
  zero-command `n198_gate3_final_terminal_level_n197_shadow_059` with
  `-Gate3FinalTerminalLevelN197`. Only a fully passing hash/parity/cadence/
  stream/fault report may authorize one separately preregistered Candidate
  060 `BoundedGate3`. Candidate 060 and FullLap remain unauthorized.


## N198 Reset-31000 Proof and Passive Shadow 059 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 059's inactive base/status `65/3` state was recovered through the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` on the same PID.
  Command-free readiness showed base/status `193/4`, index `0`, finish
  `-1`, visible Gate 1, and old boot/race epoch `822084/798447 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `822084 -> 3192 ms`; fresh race start
  was `3297 ms`. A later command-free proof showed boot `16217 ms`,
  base/status `193/4`, index `0`, finish `-1`, `234` Gate-1 detections,
  and no collision.
- Passive `n198_gate3_final_terminal_level_n197_shadow_059` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `599/10.016 = 59.704 Hz`, zero violations. All `253/253` visible
  samples replayed at `25.219 Hz` with maximum action error
  `1.9466705319937105e-7`; every deployment hash matched. It observed
  `2425` telemetry messages, `1249` IMU samples, and `122/122` frames/
  detections, with zero collision, dropout, drain hit, malformed message,
  no-quad result, or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `511e7864d82d0f73ac1348a8f8555ad8b3d23b56f899d07e0576be84ee4858f2`,
  `e22f409e6429532f8db54570cc0bcd65414fe11f08d9093d143c7967a264b302`,
  `190cfd6e11cac446d8e1d013f738f8cbb3d3fbd058af5945657f4333ce1c71d1`,
  `2caf105ac2843e18c814cc0b8c02385e9059d8e08e0ba5844dc1432d1d0367ca`,
  `b522ec24b6c8f0bfe7af8e1e0bb73f76b1f1136da692a63ad60ca249e74ec1de`,
  and `339eda10bb544565184a815da96b16d1f90cda3243d815ca68128d5ff52405d9`.
- Authorize exactly one Candidate 060 `BoundedGate3` under tag
  `n198_gate3_final_terminal_level_n197_bounded_060`: frozen N198 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 061 and FullLap remain unauthorized.


## N198 Candidate 060 — Clean Official Gates 1--3 Milestone

- Candidate 060 ran exactly once on reused responsive shipping PID `24476`;
  no process launch. Its disarm-first learned-target MAVLink `COMMAND_LONG
  31000` reset was detected in `0.516 s`: boot rolled `191126 -> 3399 ms`,
  fresh race start was `3296 ms`, index `0`, finish `-1`, and calibration
  completed `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.836339950 s`; the
  configured index-3 stop fired at controller elapsed `10.782 s`, and finish
  remained `-1`. There was no collision or invalid state.
- Transport and perception were healthy: `641/10.672 = 59.970 Hz`, zero
  command-rate violations, `3351` telemetry messages, `1693` IMU samples,
  and `136/136` frames/detections, with zero telemetry dropout, drain hit,
  malformed message, no-quad result, or decode failure. Trace replay shows
  N198 terminal leveling active on the final recorded Gate-3 sample at
  `1.849711 m`; N197 zero-confidence recovery remained inactive.
- The command-free post-stop proof sent no reset/control command and retained
  base/status `65/3`, official index `3`, last-gate time `10.836339950 s`,
  finish `-1`, and no collision. Reuse-check/probe/smoke/attempt/summary/CSV/
  post-stop SHA-256 values are
  `2db194c60686ef6c8e464ce69d5d46092584f4ae1da995c7076fe5ede9e583cc`,
  `db78ed65ec16bb6794c6639f63b3a7b30cee5788f8693755e5d49035bd87161b`,
  `55e9ef64a6147d8b1613c5a2ba5e4d9970916e87eb791631dc18e012718cde4f`,
  `f680b61973621dc43fd95d2cd972885e9de01f801d5c078a3d764571c9a63543`,
  `32cfbcbdff5ef286f09de3e78d8dd30a1c2d78f7754a9950755584dc6e94e38c`,
  `fa890d6152edf5d22a048657d45275035656ba6b2b4db500b9ba93b5e6afa40b`,
  and `a038b3ff236e2527b4c0f63ce5178c7fb2ae7cac78c235ffaeff8357ea1f940d`.
- Candidate 060 is accepted only as a bounded official Gates-1--3 milestone.
  Authorize exactly one Candidate 061 `BoundedGate4` under tag
  `n198_gate3_final_terminal_level_n197_gate4_bounded_061`: no policy,
  checkpoint, code, or deployment-hash change from passed Shadow 059/
  Candidate 060; actual Windows Python, reused PID `24476`, one detected
  normal-disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `4`. Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 062 and FullLap remain unauthorized.


## N198 Candidate 061 — Rejected at Official Gate 3

- Candidate 061 ran exactly once as the authorized `BoundedGate4` attempt on
  reused responsive shipping PID `24476`; no process launch. The runner
  recovered the inactive VQ1/R1 UI on the same PID without relaunch. Its
  disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset was detected
  in `0.500 s`: boot rolled `145737 -> 3428 ms`, fresh race start was
  `3286 ms`, index `0`, finish `-1`, and calibration completed `60/60`.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.537002563 s`. The camera-side pass counter recorded a third crossing at
  controller elapsed `10.453 s`, but official index remained `2`. No collision
  occurred; the run timed out at the preregistered `45 s` cap with finish
  `-1`, so Candidate 061 is rejected without unchanged retry.
- The failure is a stale-association/weak-detection stall, not a transport
  fault. After the camera-side crossing, the motion track stayed near
  `2.44--2.50 m` while new detections jumped tens of metres away at raw
  confidence around `0.02--0.03`; detections ended near `14.53 s` and final
  detection age was `30.453 s`. The exact-zero N197 watchdog therefore
  repeatedly released on weak nonzero false detections; trace replay shows
  `13` exact-zero activation episodes beginning only at `15.266 s`.
- Transport remained healthy: `2742/44.890 = 61.060 Hz`, zero rate
  violations, `11072` telemetry messages, `5593` IMU samples, and `564`
  frames, with zero collision, dropout, drain hit, malformed message, or
  decode failure. Exit disarm count was one.
- Command-free post-stop sent no reset/control command and showed base/status
  `65/3`, cleared race start `-1`/index `0`, finish `-1`, and no collision.
  Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `e1529747303c1ad67b18eb48df078892476ace1a29cbbc26f1af84e3b1f586b8`,
  `c6424ff6d604609db9c22625e1d01048435f7ffaad703e9b4a9169b089bdec2d`,
  `1042407989c2321d57300740ee18e636f38f8ed671b21dea64586695106bf845`,
  `6f0e91bb854de5c81b887a977d9ac408d03101072aef846d8d9ec103c01504b4`,
  `51ae6294cc18d3e9423353e678d441e6e4cc118eae14500d480df6f021cf1fd7`,
  `6e5eb52c7c9600f80f91df80cbb134bedf46435967682cadc8ba6bf90d44075a`,
  and `c241708cbee039e4dafd7aeebeb67a5da2593c355a3818092c824383199c1f47`.
- Candidate 062 and FullLap remain unauthorized. Next only: implement N199
  as exact N198 plus one Gate-3 low-confidence recovery latch, prove its
  separation offline, then require a new exact reset proof and passive shadow.


## N199 Gate-3 Low-Confidence Latched Recovery — Offline Freeze

- N199 evaluates exact source-pinned N198 once, then adds one Gate-3-only
  recovery. While official Gate 3 is visible, within `3.0 m`, closing at
  least `1.0 m/s`, and legal raw-confidence field 30 is `<= 0.025` for five
  consecutive inference samples, it latches until the official gate index
  changes. While latched it sets only normalized roll to `0.0` and commands
  the existing bearing-signed `0.25 rad` yaw scan. Pitch and thrust remain
  exact. Weak nonzero false detections no longer release the recovery.
- Source-hashed replay covers every Candidate 016--061 trace. Only four failed
  stale/weak-confidence Gate-3 paths activate, exactly once each: Candidate
  034 for `305` archived samples beginning `11.375 s`, Candidate 053 for
  `308` beginning `11.031 s`, Candidate 057 for `309` beginning `11.047 s`,
  and Candidate 061 for `299` beginning `11.344 s`. Every other trace remains
  inactive, including protected clean paths 016/018/020/021/022/055/058/060.
  Candidate 061 recovery now begins after its `10.453 s` camera-side crossing
  but before the old fragmented exact-zero recovery at `15.266 s`.
- All active samples change only roll/yaw, preserve pitch/thrust exactly, and
  remain finite/in range. Candidate 061's command-free post-stop proof is
  embedded. Focused dependency/policy/runner tests pass `35/35` on Linux and
  `35/35` under the actual Windows Python; the PowerShell deployment parser
  passes. Policy/verifier/offline-report/runner SHA-256 values are
  `9deaecd3957d02b11dabdc7b1fb36ce649f1e5575fd11fc1e190332508742d30`,
  `3bb48ce664d8591747479a95a77ef08245ac5643ffccb49b5173ff1b475ac5fa`,
  `420074d73a2879b587c63895007906c6daf1065423f632890c36c1c992397495`,
  and `b9b3486802a45fafc9ed59767e53d9acdf2bbbf62919444bbe51b25aa1a5010e`.
- No simulator command occurred during N199 development. Candidate 062 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick`; send normal disarm, wait
  `100 ms`, then exactly one learned-target MAVLink `COMMAND_LONG 31000`,
  confirmation `0`, parameters 1--7 zero. Require boot rollback greater than
  `1000 ms`, fresh nonnegative race start, official index `0`, finish `-1`,
  and visible Gate 1. Then run passive zero-command
  `n199_gate3_low_confidence_latched_recovery_n198_shadow_060` with
  `-Gate3LowConfidenceLatchedRecoveryN198`. Only a fully passing hash/parity/
  cadence/stream/fault report may authorize one separately preregistered
  Candidate 062 `BoundedGate3`.


## N199 Reset-31000 Proof and Passive Shadow 060 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 061's inactive state was recovered through the sole VQ1/R1 UI
  with `-NoRelaunch -SkipLoginClick` on the same PID. Command-free readiness
  showed base/status `193/4`, index `0`, finish `-1`, visible Gate 1, and
  old boot/race epoch `568832/560784 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `568832 -> 3127 ms`; fresh race start
  was `3321 ms`. A later command-free proof showed boot `25169 ms`,
  base/status `193/4`, index `0`, finish `-1`, `236` Gate-1 detections,
  and no collision.
- Passive `n199_gate3_low_confidence_latched_recovery_n198_shadow_060` passed
  while sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op
  cadence was `613/10.000 = 61.200 Hz`, zero violations. All `327/327` visible
  samples replayed at `32.700 Hz` with maximum action error
  `2.3379898070330363e-7`; every deployment hash matched. It observed `2448`
  telemetry messages, `1260` IMU samples, and `125/125` frames/detections,
  with zero collision, dropout, drain hit, malformed message, no-quad result,
  or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `c20fc71f536469c5f3383353005abfadc0ef6c4c9a92bb2ebdab5da815a0d2c1`,
  `c6845f9f28a4b00a94798489af21665587bc832378f93dc8ffd19ac7ade5e002`,
  `983ff37316b293f4a03b4fb6693894f9eb6bcdd2506a25a2b631cb854663c2cc`,
  `12646f641bfdbb01e8c9965213002d17beac492e7708833af076f5e9ddcec834`,
  `27cbdb3a9bdc9745a18c92835230b76933d11b421b2bc5b01a5e85b3fa3b1b44`,
  and `c3b21347334e02987d061fffb9b2aab3f70884c8ef24a5840b030ec81da265c9`.
- Authorize exactly one Candidate 062 `BoundedGate3` under tag
  `n199_gate3_low_confidence_latched_recovery_n198_bounded_062`: frozen N199
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 063 and FullLap remain unauthorized.


## N199 Candidate 062 — Rejected at Official Gate 3

- Candidate 062 ran exactly once as the authorized `BoundedGate3` attempt on
  reused responsive shipping PID `24476`; no process launch. Its disarm-first
  learned-target MAVLink `COMMAND_LONG 31000` reset was detected in `0.516 s`:
  boot rolled `128409 -> 3408 ms`, fresh race start was `3297 ms`, index `0`,
  finish `-1`, and calibration completed `60/60` samples.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.426924228 s`. The camera-side counter recorded its third crossing at
  controller elapsed `10.359 s`, but official index remained `2`. Collision
  `1001`/threat `2`, impact `9.807114601`, ended the run near `10.562 s`;
  finish remained `-1`. Candidate 062 is rejected without unchanged retry.
- N199 did not activate before the collision: weak-confidence persistence was
  absent and the last raw confidence was `0.042188`. N198's `2.0 m` terminal
  admission also remained inactive; counterfactual replay of the prior
  `3.7 m` safe-projection controller activates for two recorded samples at
  `10.422/10.531 s`, forward `2.452/2.420 m`, replacing recorded roll
  `-0.467/-0.867` with level roll. Candidate 061 would similarly activate
  before its stale-track divergence at `10.297 s`.
- Transport was healthy: `625/10.469 = 59.605 Hz`, zero rate violations,
  `3305` telemetry messages, `1669` IMU samples, and `134` frames, with zero
  dropout, drain hit, malformed message, or decode failure. Exit disarm count
  was one.
- Command-free post-stop sent no reset/control command and retained
  base/status `65/3`, official index `2`, last-gate time `7.426924228 s`,
  finish `-1`, and no continuing collision event. Reuse-check/probe/smoke/
  attempt/summary/CSV/post-stop SHA-256 values are
  `7aa62a0651ca899f37dda365797b5f0b29faf72fa813343c8359fae871028223`,
  `9e281b9bd784596360c29d227c05b83e41f64a4d3823e6ee73074a9db3c37868`,
  `4695658dcbfc2cf78ae551f82bcad8aa0ffd920349f97352ad0b6bf6278821e4`,
  `0fc67b546d16500832575809143cb5a5cf7d803fa58c12cc560c70bb3ef6edbc`,
  `3c98c6ad008f3d7c9d88e6d799366c2b7b89c04e802469d1c45d728b383f5ad1`,
  `8a673f2a71192a16c3ba5a57a360d5a4eba6fd1c72b903f1eccd0ee87275e835`,
  and `e52289244072dd2c0e9514c24436fa7cc9fb4043926f7bb3615ace23cc7883f4`.
- Candidate 063 and FullLap remain unauthorized. Next only: implement N200
  as exact N199 plus restoration of the source-pinned `3.7 m` safe-projection
  terminal roll leveler, document the corrected Candidate-059 causality, prove
  separation offline, then require a fresh exact reset and passive shadow.


## N200 Restored 3.7 m Gate-3 Terminal Level — Offline Freeze

- N200 evaluates exact source-pinned N199 once, then reapplies the exact
  source-pinned N196 Gate-3 safe-projection terminal leveler. The only
  behavioral difference from N199 is restoration of first admission from
  `2.0 m` to `3.7 m`; projection bounds, minimum closing speed, level-roll
  value, latch behavior, N199 recovery, all other controllers, checkpoints,
  and course contract remain unchanged.
- This restoration corrects the prior Candidate-059 causal interpretation.
  Candidate 059 collided at `9.776438713 s`; the archived controller's first
  terminal activation was later at `10.250 s`, so it could not have caused
  that collision. Candidate 062 supplies the opposite evidence: restored
  leveling changes its two late hard-bank samples at `10.422/10.531 s` from
  roll `-0.467/-0.867` to level. Candidate 061 restoration starts at
  `10.297 s`, before its `10.453 s` camera crossing and stale-track divergence.
- Source-hashed replay covers every Candidate 016--062 trace with no blockers.
  Every protected clean recorded action remains exact on Candidates
  016/018/020/021/022/055/058/060. Only roll can change, every output is
  finite/in range, and Candidate 062's command-free post-stop proof is
  embedded.
- Focused dependency/policy/runner tests pass `39/39` on Linux and `39/39`
  under the actual Windows Python; the PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `acf17a600e25e2c4c43e928052789c03dfa934e865e7bbd7e7d39ddd9a056885`,
  `0565d47bf18dc29863d64473ddfdda739ba24cfac8eca6c3b2ebaf54f0813d5d`,
  `538fb805f92704446b205f00f74dd1ccf561e3d204f0033c6118533ac7fab2ff`,
  and `27ffde607c8772246d25c49b85f57f744a31200ecd4936db272fe94410f60c5d`.
- No simulator command occurred during N200 development. Candidate 063 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick`; send normal disarm, wait
  `100 ms`, then exactly one learned-target MAVLink `COMMAND_LONG 31000`,
  confirmation `0`, parameters 1--7 zero. Require boot rollback greater than
  `1000 ms`, fresh nonnegative race start, index `0`, finish `-1`, visible
  Gate 1, then passive zero-command
  `n200_gate3_restored_terminal_level_n199_shadow_061` with
  `-Gate3RestoredTerminalLevelN199`. Only a passing shadow may authorize one
  separately preregistered Candidate 063 `BoundedGate3`.


## N200 Reset-31000 Proof and Passive Shadow 061 — Passed

- The sole VQ1/R1 simulator remained the existing `FlightSim.exe` PID `24476`;
  no relaunch occurred. Command-free readiness showed boot/race
  `541086/524495 ms`, base-mode/system-status `193/4`, official index `0`,
  finish `-1`, and `217` visible detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. The resulting boot rollback was `541086 -> 3095 ms` and the
  new race epoch was `3318 ms`; official index was `0`, finish was `-1`, no
  collision was present, and Gate 1 remained visible. The later command-free
  readiness snapshot retained race epoch `3318 ms`, index `0`, finish `-1`,
  base-mode/system-status `193/4`, and `218` detections.
- Passive tag `n200_gate3_restored_terminal_level_n199_shadow_061` passed with
  the frozen N200 callable: `278/278` visible samples replayed, maximum action
  error `3.9267263413078624e-07`, and `27.8 Hz` inference. Its cadence probe
  counted `587` evaluations over `9.969 s` (`58.782 Hz`) with zero rate
  violations and zero MAVLink setpoints. It received `2435` telemetry messages,
  including `1254` HIGHRES_IMU messages; all `117/117` detector frames were
  detections, with zero collisions, dropouts, drain-limit hits, malformed
  messages, arm commands, disarm commands, reset commands, or control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `b6cb0aa780a7908f03201a8bf1e4ebdc6ad016aad4fb8738023852a691c95bf8`,
  `8607b74be429046aa476be081c3506676089613d525f9d07a9e1259f999ed93d`,
  `9f1c8a09c595d381602a98b82341cd35e3e5215f968a474fbf86ab28b4047567`,
  `366b3a1731f58e8077ef7cd8c660bfe010f7d51ee6e035f812ad802eb5000209`,
  `3a973821a930549c6064c7878b7ae3e27d554ee396919eef24ac698bbfdebbee`,
  and `9a9895b0767d2961dfdc7fbb6c730131851023a98da4353c19e82adda4e8c898`.
- Exactly one Candidate 063 is now authorized: tag
  `n200_gate3_restored_terminal_level_n199_bounded_063`, mode `BoundedGate3`,
  frozen `-Gate3RestoredTerminalLevelN199`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same PID. It must
  receive another exact reset-31000 proof immediately before arming. Candidate
  064 and `FullLap` remain unauthorized; pass or fail, disarm and capture a
  command-free post-stop snapshot without resetting.


## Candidate 063 N200 BoundedGate3 — Failed at Gate 1

- Candidate 063 used the sole existing PID `24476`, tag
  `n200_gate3_restored_terminal_level_n199_bounded_063`, exact N200 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and no
  relaunch. Its embedded reset proof sent the required pre-reset disarm and one
  MAVLink `COMMAND_LONG 31000`; boot rolled back `375215 -> 3415 ms`, the new
  race epoch was `3292 ms`, reset detection took `0.610 s`, and calibration
  used `60` samples.
- The run failed acceptance and stopped after `4.594 s`: camera counter pass 1
  occurred at `4.391 s` and `2.318841 m`, but the in-run official snapshot was
  still index `0`. Collision `1001`, threat `2`, impact `1.9213672876358032`
  then made the run invalid. It sent `258` setpoints over `4.469 s`
  (`57.507 Hz`) with zero rate violations and received `2020` telemetry
  messages, including `1022` HIGHRES_IMU messages, with zero dropouts,
  drain-limit hits, or malformed messages.
- The command-free, no-reset post-stop snapshot later exposed the authoritative
  delayed transition: official index `1`, last-gate time `4.675325393 s`, race
  epoch still `3292 ms`, and finish `-1`. It was disarmed at base-mode/status
  `65/3`; there was no continuing collision. Thus Gate 1 was officially passed,
  but the vehicle struck the gate immediately afterward. Its closest accepted
  pose had the gate `0.518116 m` down at `2.318841 m` range; this is a Gate-1
  vertical-terminal miss, before any Gate-3-only N200/N199/N198 controller can
  activate. An unchanged N200 retry is forbidden.
- Reuse-check, probe, smoke, attempt, summary, CSV, and post-stop SHA-256 values
  are respectively
  `916f30383e11fc24f763ab0d0a85847d9ec0594021efd8139679536a87e0fb03`,
  `4635f809070cf47bf3002eb7c893d654eb911a5d48451074bc8470ada559f162`,
  `1112c264fe995c94df74184276adcdd061935fcdf0fda072eba63966554df778`,
  `12ddbfe259dd4eaf46a7d23b46648f81e30fb65059398fb33ac654c5ae63e7ad`,
  `8871f2643ab7f3c5f2155565957b7561b07205b08aa5033c712cbbaf8f295d4d`,
  `32167184fb78a5e2d852633469018d00448cdf64047903a89b6c492fdf29ddbd`,
  and `ab1a66d16b2e0fa70c40eec20a0ca0e4da2e4bace84c87337a73066640e57b8f`.
- Candidate 064 and `FullLap` remain unauthorized. Next only: use the archived
  trace corpus to design the smallest source-pinned N201 Gate-1 terminal
  vertical correction, prove that it activates on Candidate 063 while leaving
  protected clean Candidates 055/058/060 exact, then require a fresh reset-31000
  proof and passive shadow before any new live command.


## N201 Gate-1 High-Down Thrust Ceiling — Offline Freeze

- N201 evaluates exact source-pinned N200 once, then applies one stateless
  Gate-1-only rule: while the official gate index is `0`, the gate is visible,
  range is in `(0, 6.0] m`, closing speed is at least `1.0 m/s`, and the gate
  is at least `0.35 m` down, normalized thrust is capped at `-0.70`. Pitch,
  roll, yaw, N200, every Gate-2/Gate-3 controller, checkpoints, and the six-gate
  course contract remain unchanged.
- Source-hashed replay covers every Candidate 016--063 with no blockers. The
  rule activates only on Candidate 026 (`6` stored samples), Candidate 036
  (`1`), and causal Candidate 063 (`5`). On Candidate 063 it begins at
  `4.000 s`, before the `4.391 s` camera crossing and `4.675325393 s` official
  transition, and changes only thrust on all five samples. Protected clean
  Candidates 055/058/060 remain bit-exact; all outputs are finite and bounded.
  Candidate 063's command-free, no-reset official-index-1 post-stop proof is
  embedded in the report.
- Focused dependency/policy/runner tests pass `47/47` on Linux and `47/47`
  under the actual Windows Python; the PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `1c555692497ebb2e0ec35c309158ee55813d5caad299709483439bf636c4c1e9`,
  `cdd0fb38570c24f36a23beb485a82969c80ece28f619a0bc84d72700dcae9155`,
  `5ac86c7d905539e0978c3ff9094a2574f5901846318bfa6d0f3a41a833accf1a`,
  and `3edbfb3df8d15c35f5c8bc3871aa9a2345dde9f0d310b3a4f1f87f1e10e3d143`.
- No simulator command occurred during N201 development. Candidate 064 and
  `FullLap` remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI without relaunch if base-mode/status remains `65/3`; perform
  normal disarm, wait `100 ms`, then exactly one MAVLink `COMMAND_LONG 31000`
  with confirmation `0` and parameters 1--7 zero. Require boot rollback greater
  than `1000 ms`, a fresh nonnegative race epoch, index `0`, finish `-1`, and
  visible Gate 1, then run passive zero-command tag
  `n201_gate1_high_down_thrust_ceiling_n200_shadow_063` with
  `-Gate1HighDownThrustCeilingN200`. Only a passing shadow may authorize one
  separately preregistered Candidate 064 `BoundedGate3`.


## N201 Reset-31000 Proof and Passive Shadow 063 — Passed

- The sole VQ1/R1 simulator remained the existing `FlightSim.exe` PID `24476`;
  the inactive `65/3` post-crash state was recovered through the UI with
  `-NoRelaunch -SkipLoginClick`, and no process restart occurred. Command-free
  readiness showed boot/race `1029754/1010353 ms`, base-mode/status `193/4`,
  official index `0`, finish `-1`, no collision, and `182` detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. Boot rolled back `1029754 -> 3236 ms`, the fresh race epoch
  was `3265 ms`, official index was `0`, finish was `-1`, no collision was
  present, and Gate 1 remained visible. The later command-free ready snapshot
  retained epoch `3265 ms`, index `0`, finish `-1`, base-mode/status `193/4`,
  no collision, and `211` detections.
- Passive tag `n201_gate1_high_down_thrust_ceiling_n200_shadow_063` passed with
  the frozen N201 callable: `304/304` visible samples replayed, maximum action
  error `2.0600662231640143e-07`, and `30.35446829755367 Hz` inference. Its
  cadence probe counted `619` evaluations over `10.015 s` (`61.707 Hz`) with
  zero rate violations and zero MAVLink setpoints. It received `2427` telemetry
  messages, including `1250` HIGHRES_IMU messages; all `126/126` detector
  frames were detections, with zero collisions, dropouts, drain-limit hits,
  malformed messages, arm commands, disarm commands, reset commands, or
  control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `e8f617b993480b3eb3ab4a8cdbc5b932bc8d0ce5eaff0a39bce9b437242f3261`,
  `fb167e7d522673a0d5447f6ae03ae281c06668acf788dbf3229af92958cffe2f`,
  `ffcf0dd508541ce8429277453fc0892219bb379cf3674bf8d191be5a974613ac`,
  `d0540228e5f169511b7ec86c609376203487c35dc82053bd8548a47a63ca5733`,
  `3a64fe1dbec9c82929445926969d7a0022e4a615a935ef8b164fdc8c2f77a535`,
  and `e54e57cdf2a56b7451507d7e04005c9ba2306cdcdf04c35e72de4f5d84843592`.
- Exactly one Candidate 064 is now authorized: tag
  `n201_gate1_high_down_thrust_ceiling_n200_bounded_064`, mode `BoundedGate3`,
  frozen `-Gate1HighDownThrustCeilingN200`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same PID. It must
  receive another exact reset-31000 proof immediately before arming. Candidate
  065 and `FullLap` remain unauthorized; pass or fail, disarm and capture a
  command-free post-stop snapshot without resetting.


## Candidate 064 N201 BoundedGate3 — Official Gate 3, Collided and Invalid

- Candidate 064 used the sole existing PID `24476`, tag
  `n201_gate1_high_down_thrust_ceiling_n200_bounded_064`, exact N201 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and no
  relaunch. Its embedded reset proof sent the required pre-reset disarm and one
  MAVLink `COMMAND_LONG 31000`; boot rolled back `262331 -> 3381 ms`, the fresh
  race epoch was `3300 ms`, reset detection took `0.531 s`, and calibration
  used `60` samples.
- Gates 1--3 advanced officially; Gate 3 recorded at `10.561188697 s` and the
  stop fired at index `3` after `10.532 s` control. This is not a valid pass:
  collision `1001`, threat `2`, impact `4.9736151695251465` was present at the
  stop, two collision messages were received, and smoke acceptance explicitly
  failed on `crash_detected` and `invalid_run`. The camera counter was ordered
  through three, but its explicit events contained only Gates 1 and 2. N201's
  Gate-1 ceiling was inactive, so the new Gate-1 correction did not perturb
  this trajectory.
- The run sent `607` setpoints over `10.407 s` (`58.230 Hz`) with zero rate
  violations and received `3270` telemetry messages, including `1650`
  HIGHRES_IMU messages, with zero dropouts, drain-limit hits, or malformed
  messages. Near Gate 3, N200's restored terminal leveler zeroed roll while the
  gate remained `0.584034 m` left at `2.689076 m`; the official crossing and
  collision followed. This is the next causal boundary to analyze offline.
- The command-free, no-reset post-stop snapshot showed inactive base-mode/status
  `65/3`, cleared race epoch/index `-1/0`, finish `-1`, and continuing collision
  `1001` (threat `1`, impact `0.021709894761443138`; `90` collision messages).
  No reset was sent.
- The validation wrapper exposed a fail-open aggregation bug: its child exit
  code was `1`, status was `smoke_failed_acceptance`, and smoke acceptance was
  false, yet official index `3` alone set `gate_progress_passed=true`, producing
  aggregate promotion success and outer exit `0`. This result is manually and
  authoritatively rejected. Future aggregation must require both clean smoke
  acceptance and official progress.
- Reuse-check, probe, smoke, attempt, summary, CSV, and post-stop SHA-256 values
  are respectively
  `c772678bb8c53d1b914452ee9145f69d0c6805813c4b87306aca7623126934c6`,
  `31b473bed37d73c667026967c4c28122379acd347febac80b004f67e3c833d37`,
  `b9fbb0d3e5dfcbc669be9be1c020b3c474310cf8f1fede3562f06915e5b9ca82`,
  `98685bd9690cc77dfb19ab454bdff4cf3ee4c36a5ec6bd04e3482dc345a66f27`,
  `1be5fcd7aa083c8a86643c0cbdbd36996d962cee1a239bf623197f4efe68689d`,
  `92db9c670fd84c438b1d0c8d11e56de23df8958fd07a410493ef11fe34bd2a97`,
  and `816fb5a86f1198f3674f041fd52a1ad9c43062273947eee8c965005b6c38cdd9`.
- Candidate 065 and `FullLap` remain unauthorized. Next only: make official
  validation fail closed on child exit/smoke acceptance/crash validity, then
  use source-hashed replay to replace N200's `0.5 m` Gate-3 terminal projection
  admission with the smallest separator that preserves C060 and the intended
  C061/C062 corrections but does not level Candidate 064's unsafe lateral path.


## N202 Fail-Closed Validation and Tight Gate-3 Terminal Corridor — Offline Freeze

- Official policy validation now fails closed. Official progress counts only when
  the child exits `0`, smoke `acceptance_passed` is explicitly true,
  `crash_detected` and `invalid_run` are not true, and the official index
  reaches the requested minimum. Candidate 064's archived child-exit-`1`,
  rejected, collided fixture therefore evaluates false and cannot promote from
  its official index `3` alone.
- N202 evaluates exact source-pinned N199 once, applies the restored Gate-3
  terminal roll leveler with only the projected-lateral admission tightened
  from `0.50 m` to `0.30 m`, then applies the exact N201 Gate-1 thrust
  ceiling. The `3.7 m` forward limit, `2.0 m` projection plane, `1.0 m/s`
  minimum closing speed, `1.0 m` projected vertical bound, zero-roll output,
  latch, all non-roll actions, checkpoints, and six-gate contract are unchanged.
- Source-hashed replay covers every Candidate 016--064 trace with no blockers.
  Candidate 061 remains admitted at projected-right `+0.257 m` at
  `10.297 s` and latches for `308` samples; Candidate 062 remains admitted at
  `+0.093 m` at `10.422 s` for both samples. Candidate 064's prior N200
  admissions at `-0.340628 m`/`-0.373181 m` at `10.313/10.454 s` are both
  rejected. Protected clean Candidates 055/058/060 remain action-exact, no
  non-roll output changes, and every output is finite and bounded. The report
  embeds Candidate 064's rejected attempt and command-free collision post-stop
  evidence.
- Focused dependency/policy/validator/runner tests pass `59/59` on Linux and
  `59/59` under the actual Windows Python; the PowerShell deployment parser
  passes. Validator/policy/verifier/offline-report/runner SHA-256 values are
  `2be1b44399a5b19dce2204567efbd2cad38099026ff765eabc01207fb9cef311`,
  `892f38e34da75f29d7b20a42638ff26a19773518ecab751e4f408842733538dd`,
  `d57b0a8b5e1a3842f756e8bae7bca6e40048f023a5423c05b04242a46c3463ee`,
  `bc46ba1bdfeb0424b041af6c3517f3a04caccaed356000cb34fa024bb493f82b`,
  and `4c37ce1bc21dadf2e475471cb591e0035e2e45a74100a95c433df84938c72233`.
- No simulator command occurred during N202 development. Candidate 065 and
  `FullLap` remain unauthorized. Next only: preserve sole `FlightSim.exe` PID
  `24476`; recover the VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` if needed;
  send normal disarm, wait `100 ms`, then exactly one learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 all zero. Require a
  boot rollback greater than `1000 ms`, fresh nonnegative race epoch, official
  index `0`, finish `-1`, and visible Gate 1, then run passive zero-command tag
  `n202_gate3_tight_terminal_level_n201_shadow_064` with
  `-Gate3TightTerminalLevelN201`. Only a passing shadow may authorize one
  separately preregistered Candidate 065 `BoundedGate3`.


## N202 Reset-31000 Proof and Passive Shadow 064 — Passed

- The sole shipping simulator remained responsive as PID `24476`; Windows also
  exposed launcher PID `25552`, but no process was relaunched. The inactive
  `65/3` collision state was recovered through the UI with
  `-NoRelaunch -SkipLoginClick`. Command-free readiness then showed
  boot/race `1193933/1182994 ms`, base-mode/status `193/4`, official index
  `0`, finish `-1`, no collision, and `196` detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. Boot rolled back `1193933 -> 3221 ms`, the fresh race epoch
  was `3283 ms`, official index was `0`, finish was `-1`, no collision was
  present, and Gate 1 remained visible. The later command-free ready snapshot
  retained epoch `3283 ms`, index `0`, finish `-1`, base-mode/status `193/4`,
  no collision, `16250 ms` boot time, and `234` detections.
- Passive tag `n202_gate3_tight_terminal_level_n201_shadow_064` passed with the
  frozen N202 callable: `311/311` visible samples replayed, maximum action error
  `2.493780136107737e-7`, and `30.908 Hz` inference. Its cadence probe counted
  `605` evaluations over `10.0 s` (`60.4 Hz`) with zero rate violations and
  zero MAVLink setpoints. It received `2423` telemetry messages, including
  `1248` HIGHRES_IMU messages; all `126/126` detector frames were detections,
  with zero collisions, dropouts, drain-limit hits, malformed messages, arm
  commands, disarm commands, reset commands, or control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `e75df46c68a522330c5052dea6d13b703f759988129210803c3947687ca57458`,
  `c6871330efafeadd994ea9fa3b52303902c10bd9930962800ad9e8ad9032de93`,
  `7999da78a30970549a0905bcf9f5d8a0bc91b16a270f5d6b3e9b0872d2cc59bc`,
  `7f651542f8aa3d6b05913c734c2e4c564838926d28fbcf4b18bd1b8c2c1a992d`,
  `504af65c772ce64afb648d3397556d71f6d9985bec40501989080fc9b29bf635`,
  and `6a1542ca2792375edbbe54c42811a3cbea9cb74dc84074b890c402b6896b752f`.
- Exactly one Candidate 065 is now authorized: tag
  `n202_gate3_tight_terminal_level_n201_bounded_065`, mode `BoundedGate3`,
  frozen `-Gate3TightTerminalLevelN201`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same shipping PID.
  It must receive another exact reset-31000 proof immediately before arming.
  Candidate 066 and `FullLap` remain unauthorized; pass or fail, disarm and
  capture a command-free post-stop snapshot without resetting.


## Candidate 065 N202 BoundedGate3 — Gates 1–2 Official, Collided and Rejected

- Candidate 065 used the sole shipping PID `24476`, tag
  `n202_gate3_tight_terminal_level_n201_bounded_065`, exact N202 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and
  no relaunch. Its embedded reset proof sent the required pre-reset disarm and
  one learned-target MAVLink `COMMAND_LONG 31000`; boot rolled back
  `177116 -> 3381 ms`, the fresh race epoch was `3297 ms`, reset detection
  took `0.515 s`, and calibration used `60` samples.
- The attempt stopped invalid after `7.703 s`. Its camera counter reached two,
  with the explicit Gate-2 event at `7.546 s` and `2.165414 m`, while the
  in-run official snapshot still showed index `1` and last-gate time
  `5.066904544 s`. Collision `1001`, threat `2`, impact
  `6.18809700012207` then triggered immediate rejection. The hardened wrapper
  correctly retained child exit `1`, `gate_progress_passed=false`, zero valid
  passes, `promotion_passed=false`, and outer exit `1`.
- The command-free, no-reset post-stop snapshot exposed the delayed official
  Gate-2 transition: index `2`, last-gate time `7.863622665 s`, unchanged
  race epoch `3297 ms`, and finish `-1`. It was disarmed at base-mode/status
  `65/3`, the collision event had cleared, and no reset was sent. Thus Gates 1
  and 2 passed officially, but the vehicle struck immediately after Gate 2.
  N202's changed Gate-3 branch never received official index `2` during live
  control and did not activate; this run does not validate the N202 separator.
- Transport remained healthy: `472/7.547 = 62.409 Hz`, zero rate violations,
  `2633` telemetry messages, `1330` HIGHRES_IMU messages, `97` detector
  frames/detections, and zero telemetry dropouts, drain-limit hits, or malformed
  messages. Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values
  are respectively
  `16899e9aa45176ea3f9682640d167d45d563732ccabccb72ac3c84e88c6452c8`,
  `ab498dd94d29dc96e65ae742943deeca6ffc88a85568e0e1fefc784de96a01cc`,
  `d3054c5eb60a1e5ab78972ba62de3fb07761d0e37f5b12d3b04b5c62dd7460d5`,
  `e3b525c5beacea1a1bfb4353a26657480b2b833410775dc3509e867de8397feb`,
  `86b09516a71b8b6b643fb1b155a580c0095656665b0f563b94018c8bdb64dacc`,
  `7888ffe734dede2c4a754241c484402abfab7c476af2e4d3aa571feea6257063`,
  and `7540e8973f147d5e11d96ecc04edd5968f40ba47f478e195763fa0367f8e11fb`.
- Candidate 066, unchanged retries, and `FullLap` are unauthorized. The
  single-trace patch/one-attempt loop is closed: Candidate 065 failed before the
  changed branch could run, demonstrating prefix outcome variation. Next only:
  define and implement a repeated-run six-gate optimizer whose lexicographic
  objective is valid finish, clean official gate count, then lap time; automate
  the same MAVLink-reset/same-process recovery contract; and promote only from
  aggregate batch evidence rather than one crash trace.


## Six-Gate Optimizer Stages 1–3 and N205 Gate-4 Hold — 2026-07-18

- The active objective is a competitive VQ1/R1 lap through all six official
  gates. No candidate is promoted without official gate index 6, a nonnegative
  finish time, no collision, and repeated evidence. The only simulator is
  `C:\\Users\\anon\\Desktop\\AI-GP Simulator v1.0.3385\\AIGP_3385\\FlightSim.exe`;
  reuse the responsive shipping process (PID `24476` in these stages) and do
  not relaunch it. Every attempt uses normal disarm, a `100 ms` wait, and one
  learned-target MAVLink `COMMAND_LONG 31000` with confirmation `0` and
  parameters 1–7 all zero. Reset acceptance still requires a boot rollback
  greater than `1000 ms`, a fresh nonnegative race epoch, official index `0`,
  and finish `-1`. Collision-inactive `65/3` is recovered through the UI in
  the same process with `-NoRelaunch -SkipLoginClick`.
- The new resumable, dry-by-default tournament runner is
  `scripts/run_windows_six_gate_batch_optimizer.ps1`. It gives each attempt
  exactly one reset/arm/flight/disarm lifecycle, captures a command-free
  five-second post-stop proof, and stops on the first valid finish. The scorer
  `scripts/score_official_six_gate_batch.py` ranks by valid finishes, minimum
  official gate count, median official gate count, collision-free count,
  collision count, then time. Scorer SHA-256 is
  `538161ed73bf27fff14c2cd8eb6b59dbfc75981477c10e060073112d3724a48c`.
- Stage 1, `competitive_six_gate_batch_001`, ran N195, N197, and N198 once
  each. All three ended at official index `2`; N195 collided after `8.344 s`,
  while N197 and N198 remained collision-free for the full `45 s` but
  deadlocked on Gate 3. There was no finish, so no candidate was promoted.
  Evidence is in the batch manifest, score, and Gate-3 diagnosis JSON under
  `logs/sitl/competitive_six_gate_batch_001_*`.
- Stage 2 introduced N203, an exact N202 parent plus a Gate-3-only counter-bank
  hysteresis with fixed `0.8 m` release gap and frozen entry variants H08,
  H12, and H16. All three advanced from the N202 baseline's official index
  `2` to official index `3`, proving the changed branch in the simulator.
  H08/H12/H16 then collided at Gate 4 after `15.219/15.141/17.937 s`;
  baseline N202 collided at index `2` after `16.984 s`. H12 ranked first and
  is retained only as the experimental parent, not as a valid lap. N203 source
  SHA-256 is `b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe`;
  its focused tests passed `8/8`. Aggregate evidence is
  `logs/sitl/competitive_gate3_hysteresis_batch_002_*`.
- Stage 3 tested the isolated Gate-4 yaw-sign hypothesis as N204/YP. The exact
  repeat batch `competitive_gate4_yaw_sign_batch_004` produced official gate
  counts `3,3,2`: two collision-free `45 s` Gate-4 stalls and one Gate-3
  collision at `11.062 s`, with no finish. On the two Gate-4-evaluable runs,
  the positive sign made bearing error grow to about `0.77 rad` and the target
  left the field of view; safety came from the subsequent hover, not correct
  steering. N204/YP is rejected and must not be promoted or used as the next
  parent. Its source SHA-256 is
  `a06c41623d7dd154d63450ff028f3de274d107e59a8c2d7407d88b39c2628db9`.
- Association replay isolated the next contradiction. In the original H12
  trace, Gate 4 acquired a coherent anchor, but later visible poses were
  rejected as discontinuous; the parent association correctly rejected them
  while the yaw acquisition path still overwrote its yaw target from those
  same rejected detections. N205 fixes only that inconsistency: after an anchor
  exists, a visible rejected pose retains the last coherent associated yaw
  target with the existing `0.35 rad` bound. Pre-anchor acquisition,
  associated/re-anchored poses, dropouts, yaw sign, and pitch/roll/thrust are
  exact passthrough. Source is
  `scripts/policy_callable_gate4_coherent_target_hold_n203.py`, SHA-256
  `23100ecad829add77d9de88f4778d8ec9d920d3a2d03ce93c460b47ff484d511`.
  The focused Linux suite passed `32/32`, the exact Windows runtime subset
  passed `21/21`, and both PowerShell entry points parse. The deployment
  runner SHA-256 is
  `c22f36ca037efa76abc67a188a1eb69c5f0143d7996773eb0ff18a51f4d8b904`;
  the batch runner SHA-256 is
  `43269e175d3f16d0fc42d4c0daf8d8e60dc1bb2c298860f1bcac4373e091fbb3`.
  The preregistered official batch is
  `competitive_gate4_coherent_hold_batch_005`, variant `CH`, three exact
  `45 s` full-course attempts at `80 Hz`. It is evidence gathering only;
  promotion still requires an actual valid six-gate finish.


## Stage 4 N205/CH Official Batch — Rejected; Learned Gate-4 Tail Next

- `competitive_gate4_coherent_hold_batch_005` completed all three exact
  attempts on the reused responsive shipping PID `24476`; every attempt used
  the required disarm/`100 ms`/single MAVLink `COMMAND_LONG 31000` reset
  proof and a command-free post-stop capture. Official gate counts were
  `3,3,1`, all three attempts collided, no finish timestamp existed, and no
  candidate was promoted. Run 1 reached post-stop index `3` but hit ID `1001`
  after `10.656 s` at impact `1.7049003`; run 2 reached index `3` then hit
  ID `1002` after `15.343 s` at impact `5.5965281`; run 3 hit ID `1001`
  at index `1` after `7.282 s` at impact `8.6681356`. Aggregate minimum/
  median/maximum official count is `1/3/3`, with `0/3` collision-free and
  `0/3` valid finishes. Manifest/score SHA-256 values are
  `d4245e5b6f2fbba474cb97d0613ac526b7d9641533a5ce4c721e410272b4b39b`
  and `3a2f8b9cd7c73167409c2204eab53534a2bb42c46b0f38fd63ecdca5e519d310`.
- N205's coherent rejected-visible yaw hold did activate in the sole Gate-4-
  evaluable run, but it did not solve the collision and is rejected. The trace
  isolates a stronger blocker: Gate 4 became active at `11.062 s`; continuous
  `0.22` thrust began at `12.109 s` while the accepted family was about
  `27.43/3.94/10.76 m` forward/right/down, and persisted until ID `1002`
  contact. This repeats the ledger's Candidate-139/159 ground-contact result:
  far `21–25 m` descent is unsafe, while current-associated close/bounded
  descent was the retained safe form. The present N158-era handoff's one-second
  continuous vertical branch conflicts with that later evidence.
- Do not make another hand-controller scalar child. The historical manual Gate-4
  bracket was already closed at Candidate 207 after no index-4 pass. The next
  smallest distinct mechanism is the already-trained N142 phase-reset tail:
  retain exact H12/N203 for Gates 1–3, but at official Gate 4 bypass the manual
  safety handoff and execute the frozen learned Gate-4/N112 checkpoint directly,
  with a fresh recurrent reset at phase entry. That checkpoint passed its native
  fixed Gate-4 segment `128/128`; it has never received official Gate-4 evidence
  because its earlier N142 live candidate failed in the prefix. Test it as a
  bounded, source-pinned variant and reject it on any collision or lack of
  aggregate official improvement. It is not promoted by native evidence alone.


## N206 Learned Gate-4 Tail Direct — Implemented and Running Stage 5

- `scripts/policy_callable_gate4_learned_tail_n203.py` implements the distinct
  learned-tail experiment. It source-pins H12/N203 for official Gates 1–3.
  Only while official Gate 4 is active, it clears the retired manual Gate-4
  state and calls the frozen N142 composite exactly once using the standard
  coordinate-free 32-value observation with reserved slots `30/31` cleared.
  The N142 composite owns the phase-entry recurrent reset and selects frozen
  N112 step `294912` for Gate 4. Gates 5–6 remain the same learned phase-reset
  composite. No gate coordinate, future-gate state, GPS, or privileged simulator
  state is used. Source SHA-256 is
  `4a976df88dc95af46d2bd022fd8497608ff9cbe3a47c2265c4215e8243e775af`.
- The focused Linux suite passed `86/86`; the exact Windows Python 3.12 runtime
  subset passed `30/30`; both updated PowerShell entry points parse. Deployment
  runner SHA-256 is
  `e85aca0f87438ff531dd8d48801b29bbadc7c94bdaa8436931a941e34dde5fcc`;
  batch runner SHA-256 is
  `21fecd9fba9914e1e8bfb964c600da5d0b91b35e4f8b3dd0e0a3b3fd5172e90a`.
- Stage 5 is the fixed tag `competitive_gate4_learned_tail_batch_006`, variant
  `LT`, three exact full-course attempts, `45 s`, `80 Hz`, on the reused
  responsive v1.0.3385 VQ1/R1 simulator at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`.
  Each attempt retains the normal-disarm/`100 ms`/single-command MAVLink-31000
  reset proof and command-free post-stop capture. Stop the batch on any valid
  official six-gate finish. Native success is rationale for testing only; it is
  not promotion evidence.


## Stage 5 N206/LT Official Batch — Rejected; Terminal Crossing Isolated

- `competitive_gate4_learned_tail_batch_006` completed all three exact
  full-course attempts on the reused responsive shipping PID `24476` with the
  required normal-disarm/`100 ms`/single MAVLink `COMMAND_LONG 31000` reset
  lifecycle and command-free post-stop captures. Official gate counts were
  `3,1,3`; every run collided, no finish timestamp existed, and N206 is not
  promoted. Run 1 reached index `3`, then hit ID `1001` at impact
  `3.0251808` after `20.531 s`; run 2 was a prefix failure at index `1`,
  ID `1001`, impact `8.5861702`, after `7.531 s`; run 3 reached index
  `3`, then hit ID `1002` with threat `1`, impact `0.56966996`, after
  `14.469 s`. Aggregate minimum/median/maximum official count is `1/3/3`,
  with `0/3` collision-free and `0/3` valid finishes. Manifest/score
  SHA-256 values are
  `68d5650bef4b52b84391dc078469ace1adc0887f02abe2e1a1944d585c5554d9`
  and `ac117148de934c8b1a8c6813203f48db151f02cf8276e66dea593f1a828524ca`.
- Run 1 supplies the first useful official Gate-4 terminal evidence from the
  learned tail. It drove the visible gate to about `1.471891 m` forward,
  `-0.121891 m` right, `0 m` down, and `-0.082624 rad` yaw error while
  outputting decoded pitch `-0.052567 rad`, roll `0.011956 rad`, thrust
  `0.266508`, and yaw rate `0.001383 rad/s`. The last recorded policy
  sample was likewise centered at `1.576642/-0.226642/0 m` with yaw error
  `-0.142772 rad`, but the official counter stayed at `3` before gate
  contact. The learned tail therefore solved the far approach in this run and
  then commanded a terminal brake instead of a forward crossing.
- Do not reopen the retired manual Gate-4 controller ladder. The next and only
  admitted child is a narrow observable-pose residual on the N206 learned tail:
  once per Gate-4 phase, when a current visible pose is within `3.0 m`,
  `|right| <= 0.5 m`, `|down| <= 0.5 m`, and `|yaw error| <= 0.15 rad`,
  retain the learned roll/thrust/yaw and floor pitch at the already-established
  `+0.06 rad` forward command (normalized `+0.12`) for at most `0.5 s`.
  The bounded pulse may bridge detector dropout but may not retrigger in the
  same Gate-4 phase. These bounds admit the exact run-1 near pass and reject
  run 3's last coherent approach at about `11.85/3.50/-0.33 m`. Test this
  single mechanism as Stage 6; judge it only by official gate count, collision,
  and finish evidence across repeated full-course trials.


## N207 Terminal-Crossing Residual — Implemented and Running Stage 6

- `scripts/policy_callable_gate4_terminal_crossing_n206.py` source-pins N206
  and changes only learned Gate-4 pitch inside the preregistered terminal
  envelope. The one-shot pulse triggers from current visible observation values
  at `0 < forward <= 3.0 m`, `|right| <= 0.5 m`, `|down| <= 0.5 m`, and
  `|yaw error| <= 0.15 rad`; for at most `0.5 s`, normalized pitch is floored
  at `+0.12` (`+0.06 rad`) while learned roll, thrust, and yaw pass through.
  It can bridge a detector dropout only while that pulse is already active,
  cannot retrigger in the same Gate-4 phase, resets at a phase transition, and
  uses no course coordinate or privileged state. Source SHA-256 is
  `b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63`.
- The focused Linux deployment suite passed `90/90`; the actual Windows Python
  3.12 subset passed `37/37`; both PowerShell entry points parse. Unit evidence
  includes the exact Stage-5 run-1 terminal pose, every threshold boundary,
  dropout expiry, no same-phase retrigger, action-channel preservation, and
  phase/reset behavior. Deployment runner SHA-256 is
  `25fe052bed03504f2b0b8ea749db045aa55a2ee73fa006ed3fee4580601245e8`;
  batch runner SHA-256 is
  `0343814d2cca3ab8f9fa79ceb675b435a31667544bce9e0e5303d63a1cb083b2`.
- Stage 6 is fixed as tag `competitive_gate4_terminal_crossing_batch_007`,
  variant `TP`, three `45 s`, `80 Hz` full-course attempts on the already
  running v1.0.3385 VQ1/R1 simulator. Preserve the same shipping PID whenever
  responsive, use the documented MAVLink-31000 reset acceptance proof once per
  attempt, and stop immediately on a valid official six-gate finish. This is
  an experimental child, not a promoted policy.


## Stage 6 N207/TP Official Batch — No Trigger; Raw Tail Contract Rejected

- `competitive_gate4_terminal_crossing_batch_007` completed all three exact
  attempts on responsive shipping PID `24476` with the documented reset and
  post-stop lifecycle. Official counts were `2,3,3`; all three runs collided,
  no finish existed, and N207/TP is not promoted. Run 1 was a prefix collision
  at ID `1001`, threat `2`, impact `3.8700616`, after `7.266 s`. Run 2
  reached Gate 4 and hit ID `1002`, threat `2`, impact `5.7731671`, after
  `15.407 s`. Run 3 reached Gate 4 and hit ID `1002`, threat `1`, impact
  `0.02998717`, after `15.140 s`. Aggregate minimum/median/maximum official
  count is `2/3/3`, with `0/3` collision-free and `0/3` valid finishes.
  Manifest/score SHA-256 values are
  `1500673f6d3642fdb73faf972470a41d45b55ca02abb05f4889a7894585f6c24`
  and `77ffc79ae1ecf6e0dcb4f0f75851b4876133c84910f7d249b366acbea051acbf`.
- The terminal pulse had zero eligible Gate-4 samples in all runs. Run 2's
  raw Gate-4 trace began at about `38.40/4.44/4.62 m` forward/right/down and
  ended at `5.52/2.90/-1.76 m`, yaw error `0.483 rad`. Run 3 began near
  `36.92/6.40/6.63 m`; its closest raw forward sample was
  `3.91/3.74/-0.85 m`, yaw error `0.764 rad`, followed by far-target jumps.
  Neither entered the fixed centered envelope. Do not widen its bounds from
  this evidence; Stage 6 did not causally exercise the changed branch.
- The causal contract error is upstream of the pulse. The hybrid deployment
  globally forced raw Gate-4 observations because the retired manual handoff
  owned its own target association. N206 bypassed that handoff but accidentally
  inherited the raw setting. Consequently, during learned Gate-4 control the
  standard `ObservableGateMotionFilter` was reset and then never initialized;
  the N112 checkpoint received unassociated detector jumps even though its
  learned observation contract expects associated pose and observable rates.
  Stage-6 snapshots show the Gate-4 motion filter `initialized=false`, with
  `23/26` rejected-sample counters inherited from earlier phases but no tracked
  Gate-4 identity. This is a deployment-contract mismatch, not a learned-policy
  hyperparameter.
- The next distinct experiment is filtered-tail N208/FT: exact N207 policy and
  all fixed terminal-pulse bounds, but disable raw observation only for this
  variant so Gate 4 uses the existing observable association/rate filter.
  Preserve the raw setting for historical manual variants. The standard filter
  rejects first aliases beyond `35 m`, rejects discontinuous target jumps,
  holds the last officially associated gate through brief dropout, and resets
  only on official identity change. Validate this configuration offline and in
  the Windows runtime, then run a fixed repeated full-course batch. Do not tune
  filter or pulse thresholds within the batch.


## N208 Filtered Learned-Tail Contract — Implemented and Running Stage 7

- N208/FT uses the exact N207 policy source and fixed terminal-pulse parameters;
  the only change is deployment observation routing. New runner switch
  `Gate4FilteredTerminalCrossingN206` removes the raw Gate-4 environment
  interval, restoring the existing observable motion filter for Gate 4. Raw
  routing remains unchanged for every historical TP/manual variant. Variant
  `FT` is separately named in the batch runner, so raw and filtered evidence
  cannot be conflated.
- The focused Linux suite passed `92/92`; the actual Windows Python 3.12 subset
  passed `41/41`; both PowerShell entry points parse. Policy source remains
  `b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63`.
  Updated deployment runner SHA-256 is
  `8820e500ef3cd9ca48bc06b15af7abca7821174c07dae55daa909acdbee46367`;
  updated batch runner SHA-256 is
  `f2759cacc7f85ae4f06f37eadc5deb8c3c001cec4bc71c2fb2f4e21423fe3c9c`.
- Stage 7 is fixed as `competitive_gate4_filtered_tail_batch_008`, variant
  `FT`, three exact `45 s`, `80 Hz` full-course attempts on responsive
  shipping PID `24476`, with the same reset acceptance and post-stop proof.
  There is no within-batch change. Promotion still requires a valid official
  six-gate finish; otherwise retain or reject only from aggregate official
  counts, collisions, filter initialization, and trace evidence.


## Stage 7 N208/FT Official Batch — Rejected; Train the Six-Gate Policy

- `competitive_gate4_filtered_tail_batch_008` completed all three fixed
  attempts on shipping PID `24476` with the documented reset and post-stop
  proof. Official gate counts were `2,3,2`; every run collided and no finish
  existed. Run 1 collided at ID `1001`, threat `2`, impact
  `9.6425323`, after `10.453 s`. Run 2 reached Gate 4, then collided at
  ID `1002`, threat `1`, impact `0.42715925`, after `20.991 s`.
  Run 3 collided in the prefix at ID `1001`, threat `1`, impact
  `0.06559401`, after `7.297 s`. Aggregate minimum/median/maximum is
  `2/2/3`, with `0/3` collision-free and `0/3` finishes. Manifest and
  score SHA-256 values are
  `c452c97172f4e5d55eb94c3f2226f0e7ff56a5ae24ed9fd042701cf554554af1`
  and `32d37e2fa5173771dccfcf55b8b13894b28c1ed7cb9350428f509d58397c1224`.
- The filtered contract was causally exercised in run 2: Gate-4 identity was
  associated and discontinuous target jumps were rejected. It still failed
  because a long detector dropout caused the last accepted world pose and
  rates to be held indefinitely. The policy then saturated near
  `[0.855,1.0,-1.0,0.000676]` normalized action until contact. This closes
  both the raw-target tail and the indefinitely held filtered tail. N208/FT is
  rejected; do not add a timeout wrapper or continue scalar Gate-4 tuning.
- The next mechanism is actual distributional six-gate training. Extend the
  native reset contract to sample a bounded distribution around measured
  official Gate-4 entry states, mix those starts with ordinary full-course
  starts, and optimize one recurrent policy whose observation includes all six
  gate phases. Screen it first on both full-course and entry-distribution
  native suites, then use the same official simulator lifecycle. This directly
  trains the entry-state variation exposed by Stages 5-7 instead of adding
  another deployment controller.


## N209 Measured-Entry Six-Gate Training — Native Gain, Official Rejected

- The native environment now supports opt-in symmetric reset distributions
  over segment elapsed time, position, velocity, Euler attitude, and body
  rate. Zero amplitudes consume no RNG and preserve every historical fixed
  reset exactly. The native regression and focused script tests pass. N209
  reconstructs its Gate-4 entry envelope from five usable official Stage 5-7
  traces: center position about `64.3,-0.5,-6.9 m`, velocity
  `8.5,0.4,-5.0 m/s`, measured attitude/rate center and bounded jitter. It
  mixes these starts 50/50 with ordinary starts and trains the entire accepted
  N096 recurrent policy for all six gates; there is no per-gate weight freeze,
  controller wrapper, privileged coordinate input, or official simulator use
  during training.
- One `1,048,576`-step FP32 run produced 32 admissible checkpoints. Every
  checkpoint was deterministically screened for exactly 128 fixed full-course
  episodes and 128 measured-entry episodes. Step `589824`, SHA-256
  `efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb`,
  is the balanced winner: full-course `109/128` success, mean `5.375` gates,
  and measured-entry `20/128` success, mean `3.921875` gates, with zero crash
  and timeout in both. On the identical paired cohorts, unmodified N096 scored
  `103/128`, mean `5.203125`, and `17/128`, mean `3.8359375`, also with zero
  crash/timeout. N209 therefore strictly improves both paired finish counts
  (`+6` full, `+3` entry). Full/entry aggregate SHA-256 values are
  `a0dd3671b7e2cda6f8a2d77e3a1d2648a533a6acb08708a15c447b6918f18c37`
  and `af377c5d0146106413a2d5bf391124a9dfa8915ac5453db24fb6a07a384205a0`.
- The single permitted official attempt used N209 alone, FP32 input 32 with
  progress/6 at observation 23 and reserved-zero columns 24..31, 60 Hz, one
  policy-ready disarm -> MAVLink-31000 reset, and healthy reused shipping PID
  `24476`. Gate 1 passed at `5.018333435 s`; the policy then collided at
  official index `1` with object `1001`, threat `1`, impact
  `0.3188585043`, and stopped after `8.0 s`. It sent 421 commands at
  `53.225 Hz`, with zero command violations, telemetry dropouts, or drain
  limit hits. Official count is `1`, no finish exists, and N209 is rejected
  as a start-to-finish official candidate. Smoke/aggregate SHA-256 values are
  `b1ecf4b19b8e26a66cd66e5b8fdeb77bb783cfb20a2b72feb3dc19156f7cb566`
  and `b0079813186289b4cce03b34980b69c868b93694ae9a4208fb93bfd8bddc1bac`.
  A passive post-stop proof reports base/status `65/3`, responsive PID
  `24476`, official index `1`, finish `-1`; SHA-256
  `259bcf891ed7a5294bf4e5e648f7338e44ccd85e8ffe42beda87f812bb44b62e`.
- Causal conclusion: measured-entry training improves native six-gate
  completion but does not repair the known native-to-official Gate-2 prefix
  transfer gap. Do not repeat N209 start-to-finish unchanged. The simplest
  next candidate is the already official-proven H12 Gates-1-through-3 prefix
  followed by one reset N209 recurrent tail for Gates 4-6. This uses the new
  policy exactly on the measured entry distribution it improved, replaces the
  rejected per-gate/manual tail, and adds no action residual.


## N210–N217 Unified Tail, Teacher Intervention, and Distillation — 2026-07-18

- The active objective remains one competitive, collision-free official VQ1/R1
  lap through all six ordered gates. There is still no valid six-gate finish.
  Continue to reuse responsive shipping PID `24476`; the exact simulator,
  Windows Python, six-gate semantics, and normal-disarm/`100 ms`/single
  MAVLink `COMMAND_LONG 31000` reset proof at the top of this file remain
  mandatory. No unchanged N209–N217 retry is authorized.
- N210 combined the exact official-proven H12/N203 Gates-1-through-3 prefix with
  one zero-state N209 recurrent controller for the complete Gates-4-through-6
  tail. The topology is
  `scripts/policy_callable_n209_unified_tail.py`, SHA-256
  `5a00e97df818477e18424cbccb9c21fa53c435a05a8602572390419c420f7769`.
  Passive shadow `n210_h12_n209_unified_tail_shadow_002` passed with 269
  inference ticks, maximum replay error `1.6844e-7`, all manifests exact,
  zero setpoints, and `60.403 Hz` cadence; shadow/parity hashes are
  `8ef3d1fc...`/`2ca0b6a8...`.
- N210 had two real official attempts and one void startup. Attempts 001 and 003
  both passed official Gates 1–3 and then failed Gate 4. Attempt 001 stayed
  collision-free until the 45 s timeout at index `3`; attempt 003 hit object
  `1002`, threat `2`, impact `9.3623`, at index `3`. Transport was
  healthy at `60.502/60.011 Hz` with zero rate violations, dropouts, or drain
  limits. Attempt 002 sent no controller action and is void: command 31000 was
  not acknowledged because the startup reused a stale inactive race. UI
  recovery returned the same PID to `base_mode=193/system_status=4` and a
  fresh race epoch. Real-attempt smoke hashes are `1195de4f...` and
  `93058a9c...`; the void summary hash is `a8fb0fc1...`.
- N211 reward-only spline-teacher shaping did not improve the N209 tail. Its
  best exact common-256 checkpoint was `40/256=15.625%`, mean
  `3.9140625` gates, with `2/256` crashes; evaluation SHA-256 is
  `4e7caee2...`. It was rejected without live deployment.
- N212 introduced a default-off teacher-intervention mechanism in the native
  plant. Before each attitude-mode step, selected pitch/roll/thrust channels
  can be convexly blended with the teacher action; the policy's raw sampled
  action remains the imitation target, while the observation's last-action
  fields record what the plant actually executed. Blend zero preserves the
  historical dynamics exactly. Native regression and focused tests pass.
  Source hashes are `drone_race.c=e0a0b815...`,
  `drone_race.h=28b7112a...`, `binding.c=1f680ec9...`, and the config
  `2303645c...`.
- A 100% roll/thrust intervention probe completed `56/128=43.75%` measured
  six-gate tails with zero crash; probe hash is `de8df645...`. N212 training
  at blend 1.0 produced only `18/128=14.06%` when evaluated unassisted. N213
  continued at blend 0.5 and produced `21/128=16.41%`, zero crash. N214 then
  consolidated at blend zero; screening every admissible checkpoint selected
  early step `65536`, SHA-256
  `999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75`.
  It scored `23/128=17.97%`, zero crash on the selection cohort, then
  `53/256=20.70%`, mean `4.0390625` gates, `2/256` crashes on confirmation.
  Confirmation JSON SHA-256 is `1848720a...`. Later N214 checkpoints degraded;
  promote only step 65536 as a native experimental artifact, not as a valid
  official policy.
- N215 packaged that exact N214 checkpoint behind hash pins while preserving
  the N210 topology. Callable
  `scripts/policy_callable_n214_unified_tail.py` has SHA-256
  `b13b36d7c22af83eb6f2c0e1faf8f8df340fc02e49df8d555064b8aa1751d4ba`.
  The first passive shadow correctly failed in stale collision state
  `65/3`; same-PID UI recovery restored R1. Shadow 002 then passed: 192
  replayed ticks at `19.141 Hz`, maximum error `2.2291e-7`, all manifest
  artifacts exact, zero setpoints, and `54.411 Hz` cadence. Shadow/parity
  hashes are `85515e46...`/`eacd769b...`.
- The single N215 official attempt used a verified MAVLink-31000 reset and was
  valid. It passed Gates 1–3, stayed collision-free, but timed out after 45 s at
  official index `3`; finish remained `-1`. It sent 2723 commands at
  `60.616 Hz`, with zero rate violations, telemetry dropouts, or drain-limit
  hits. Smoke/attempt/summary SHA-256 values are `192fbaba...`,
  `8a9bf0fe...`, and `3641c0dc...`. At Gate-4 activation (`10.641 s`),
  the filter correctly zeroed the first three observations while rejecting
  `64–37 m` aliases. The first accepted family appeared near
  `31 m forward / 7 m right / 6 m down`; roll saturated negative, later
  counter-banked too late, overshot laterally, and the held dropout state drove
  indefinite saturation. This is the same live transfer failure as N210, not a
  transport/reset failure.
- N216 showed that forcing the teacher's fixed pitch is not a solution:
  full-channel intervention scored only `10/128=7.8125%`. N217 then captured
  a six-gate roll/thrust-intervention cohort (`59/128=46.09%`, zero crash) and
  built 128 recurrent episodes/128,727 labels solely from the next observation's
  executed last-action fields. Trace/dataset hashes are `7b06d396...` and
  `deca47cb...`; converter source is
  `scripts/build_teacher_intervention_dataset.py`, SHA-256 `021444ec...`.
  Five decoder-ridge children and one full recurrent child were evaluated on
  disjoint episode offset 128. The unchanged N214 parent scored
  `22/128=17.19%`; the best ridge child scored `4/128`, and the recurrent
  child `3/128`, all with new crash/timeout regressions. N217 is rejected:
  low teacher imitation error does not survive the partially observable
  closed-loop dropout dynamics.
- Current causal boundary: native entry training and privileged teacher
  intervention can improve synthetic completion, but the deployed observation
  stream does not identify the teacher's correction during long Gate-4
  association dropouts. Do not run another reward-only teacher, blend schedule,
  decoder fit, recurrent distillation, N209/N214 unified-tail retry, scalar
  Gate-4 wrapper, or live attempt from this lineage. The next admitted work must
  change observability or state estimation and prove that change on held-out
  closed-loop native evidence before FlightSim.
- N218 made that state-estimation change without changing N214's checkpoint or
  policy topology. A new default-off gate-motion option advances the last
  associated world-frame gate pose by its observable filtered motion during
  detector dropout. The matching live contract is enabled only by
  `PUFFER_POLICY_PREDICT_GATE_DROPOUT=1`; default-off behavior remains
  historical. Python/native regression passed, the CUDA FP32 extension was
  rebuilt, and 61 focused tests plus the native regression passed after runner
  integration.
- On two disjoint measured Gate-4 native cohorts, the unchanged N214 weights
  improved from `23/128` to `43/128` and from `22/128` to `30/128`, for an
  aggregate `45/256=17.58%` to `73/256=28.52%`, with no added crashes.
  Evaluation JSON SHA-256 values are `d30d2871...` and `9137331d...`.
- N218 shadow 001 correctly failed against stale collision state `65/3` and
  emitted zero inference/setpoints. Same-PID recovery restored R1 to
  `base_mode=193/system_status=4`. Shadow 002 passed with 251 replay ticks,
  `25.022 Hz` inference, maximum replay error `1.8254e-7`, exact manifests,
  zero setpoints, and `59.3 Hz` cadence. Shadow/parity SHA-256 values are
  `c50e2397...`/`178af23b...`.
- The sole N218 official attempt used a verified MAVLink command-31000 reset,
  passed Gates 1-3, stayed collision-free, and timed out after 45 seconds at
  official index `3`; finish remained `-1`. It sent 2664 commands at
  `59.343 Hz` with no transport fault. Smoke/attempt/summary SHA-256 values are
  `51752dac...`, `5ce49643...`, and `7f594266...`.
- N218 was causally active but is rejected for an unchanged retry. At Gate 4 it
  first acquired about `31 m forward / 6 m right / 5 m down`, then oscillated
  across the aperture under alternating saturated banks and had already
  overshot before the long dropout. After detections ceased, indefinite motion
  extrapolation ran to an impossible world target near
  `[-137.0, 96.0, 42.4] m` and saturated the observation/action. Prediction
  improves held-out native completion, but unbounded prediction does not solve
  the observed live Gate-4 control/association failure. Do not launch another
  unchanged N218 attempt.
- N219 trained the N214 recurrent tail under the N218 predictor with teacher
  reward retained but intervention explicitly zero. The predictor plus full
  roll/thrust teacher intervention supplied a controllability ceiling of
  `58/128=45.31%`, zero crash. Seventeen checkpoints were screened; exact step
  `32768`, SHA-256
  `20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec`,
  ranked first. It completed `47/128` offset-0 and `33/128` true offset-128
  tails, aggregate `80/256=31.25%`, versus N218's `73/256=28.52%`; both had
  three crashes in the second cohort. The initially duplicated offset-128
  screen was detected and discarded before promotion; the retained JSON hashes
  are `be1120d4...` and `8e50412f...`.
- N219 packaging uses callable
  `scripts/policy_callable_n219_unified_tail.py`, SHA-256 `28cce611...`, and
  requires the predictor. Seventy-five focused tests passed. Passive shadow 001
  was void because a prior moving race yielded only one stationary calibration
  sample and emitted no setpoints. Same-PID UI recovery restored R1. Shadow 002
  passed 192 replay ticks at `19.110 Hz`, maximum action error `1.8508e-7`,
  exact manifests, zero setpoints, and `53.946 Hz` cadence; shadow/parity hashes
  are `2a2967d6...`/`9741dee8...`.
- N219 official attempt 001 used a valid reset but the frozen H12 prefix struck
  Gate-2 object `1001` at official index `1`, impact `4.9878`; N219 had zero
  Gate-4 samples. It is retained only as prefix-failure evidence. The one
  replacement attempt recovered the same PID without relaunch, verified a new
  command-31000 epoch, passed Gates 1-3, and then timed out collision-free at
  Gate 4. It sent 2729 commands at `60.791 Hz` with zero transport fault;
  finish remained `-1`. Smoke/attempt/summary hashes are `3fd4e184...`,
  `30b0d2d2...`, and `19fa3213...`.
- N219's live Gate-4 trace began with the first accepted target near
  `32 m forward / 9 m right / 5 m down`. The policy saturated negative roll,
  crossed to about `4 m forward / 3 m left`, counter-saturated positive roll,
  then reacquired about `16 m forward / 3 m right` and saturated negative roll
  again. Detections ended near `9 m forward / 8 m right`; the unbounded
  predictor then diverged while the policy saturated roll/thrust until timeout.
  Predictor-specific PPO improved native cohorts but did not change the live
  oscillation. N219 is rejected; no unchanged retry is authorized.

## N220-N221 authoritative execution snapshot - 2026-07-18

- The active objective is still a competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N220 or N221 run finished; the
  goal is not complete. The exact simulator remains
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`
  with Windows Python
  `C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe`.
  Work reused responsive shipping PID `24476` without relaunch. Normal attempt
  reset remains disarm, 100 ms wait, then learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1-7 all zero; count it
  only after boot rollback greater than 1000 ms plus a fresh race epoch at
  index `0` and finish `-1`.
- N220 added a default-off roll-aware dropout predictor. The live switch uses
  `4.295 m/s^2 per tan(roll)`; native observation sampling is 15 Hz
  (four 60 Hz steps). Nine PPO checkpoints trained under that harder contract
  all scored `0/128` unassisted, as did their parent under the same condition.
  A full teacher remained controllable at `43/128`. Expert-only ridge
  distillation, seven DAgger ridge children, and a recurrent last-layer child
  all scored `0/128`; these learned branches are rejected.
- N220's observation-only tail controller was the first material native
  improvement in this branch. From measured Gate-4 entry distributions it
  completed `90/128=70.31%` development episodes and
  `82/128=64.06%` frozen holdout episodes. It owns roll/thrust only at official
  Gates 4-6, uses N219 for pitch/yaw, and reads only the 32-value deployed
  observation plus a fixed development-course prior. Its callable SHA-256 is
  `dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746`.
  Passive shadow passed with no setpoints; shadow/parity hashes are
  `3f12f038...` and `e3ef4e19...`.
- N220 official attempt 001 reset correctly, passed Gate 1, then collided while
  approaching Gate 2: official index `1`, collision ID `1002`, impact
  `6.111586`, finish `-1`. Smoke SHA-256 is `432df241...`. A causally
  changed replacement restricted the roll-aware predictor to index `>=3`;
  it again passed only Gate 1, then timed out collision-free at index `1`
  after `45.008 s`, `2716` commands, and `60.418 Hz`. Smoke SHA-256 is
  `409225a0...`. The trace showed a Gate-2 association frozen near
  `25.31 m / 2.41 m / -6.64 m` with zero rate and 115 rejected samples.
  N220 is rejected; do not retry it unchanged.
- N221 removes the brittle learned-prefix/controller-tail handoff. One
  observation-only fixed-course controller owns pitch, roll, and thrust at all
  six gates; N219 supplies yaw only. The deployable pitch loop uses observable
  gate-rate speed, not simulator velocity, with a `4.00 m/s` target. Gates 5-6
  coordinates in this controller remain synthetic development proxies, not
  measured official coordinates. Callable SHA-256 is
  `468eb14d2234d640d644a06c9877759374553905c3116bc47d6e69e7211285c7`.
- An initial `128/128` screen was invalidated before promotion because pitch
  used privileged native velocity. The corrected exact-deployment screen uses
  the same observable speed estimate as N221. It completed
  `126/128=98.4375%` perturbed full-course episodes with zero crashes and
  timeouts; two episodes missed zero-based gate 4 (official Gate 5). The
  fixed-origin run completed in `2463/60=41.05 s`. Dataset path
  `logs/drone_race_full_policy_six_gate_bootstrap/n220_control_aware_predictor/observable_controller/n221_exact_jitter128_speed_4p0.bin`,
  SHA-256 `01c0772a...`. Eight-episode, 19,356-record Python/native controller
  replay has pitch/roll/thrust RMS
  `1.032e-6 / 0.002113 / 2.790e-6`; report SHA-256 `1344e609...`.
  Ninety-five focused tests passed.
- N221 passive live shadow passed at official index `0`: `214` replay ticks,
  `21.300 Hz` inference, maximum action error `6.736e-5`, all manifest
  matches true, and no MAVLink setpoints. The no-op publisher measured
  `55.469 Hz` with zero violations. Shadow/parity SHA-256 values are
  `961d57c3...` and `d6e0bb86...`.
- N221 official attempt 001 used a fresh command-31000 epoch and healthy
  streams, but made zero official passes. It approached Gate 1 to
  `10.0 m forward / 3.266 m right / 1.422 m up`, then the association
  diverged while roll/thrust saturated. The run timed out collision-free at
  index `0`, finish `-1`, after exactly `45.0 s`, `2750` commands, and
  `61.259 Hz`. Smoke/attempt/summary SHA-256 values are `d434eeb2...`,
  `c7014011...`, and `6a10723a...`. N221 is rejected for live deployment;
  do not retry it unchanged.
- The current blocker is specifically live target association, not another
  unconstrained policy-training round. Native dynamics and controller porting
  are sufficient to complete the proxy course, while live camera association
  can accept an alias, reject subsequent correct samples, and propagate a
  false target indefinitely. The next candidate must first demonstrate a
  course-consistent, bounded association/reacquisition mechanism on the N220
  Gate-2 and N221 Gate-1 traces. It must fail closed when consistency is
  unavailable and pass passive shadow/parity before another official command.
## N222-N224 association-clock execution snapshot — 2026-07-18

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N222-N224 run finished, so the
  goal is not complete. The exact simulator, Windows Python, responsive reused
  PID `24476`, and disarm/100 ms/one MAVLink `COMMAND_LONG 31000` reset
  contract remain unchanged.
- The N220 Gate-2 trace isolated a real clock-conflation defect in
  `ObservableGateMotionFilter`: every 60 Hz prediction advanced `_time_s`,
  so the innovation horizon collapsed to one inference tick and rejected the
  next physically valid camera pose. A separate last-accepted-camera clock now
  governs association. The failed trace counterfactual changes Gate 2 from a
  frozen `[25.3118,2.4141,-6.6441]` m track to accepting the next valid
  22.3 m observation at 5.055 s and following the gate inward. Focused tests
  cover the last-good horizon, nearer-only optional reacquisition, and farther
  alias rejection.
- N222 enabled the bounded reacquisition mechanism with N221's all-course
  observable controller. It was passive shadow only; no official flight was
  run. Shadow/parity passed with zero setpoints, but live actions were more
  saturated than N221 (roll saturation 63.32%, pitch 98.25%), so N222 was
  rejected before flight. Shadow/parity SHA-256 values are
  `4e2e7818...`/`3ddca8fb...`.
- N223 paired only the corrected association horizon with the proven H12
  Gates-1-through-3 prefix and N220 Gates-4-through-6 tail. Passive shadow
  passed 224 ticks at 22.4 Hz, maximum replay error `1.5786e-7`, exact
  manifests, zero setpoints, and 57.592 Hz no-op cadence. Its one valid official
  attempt passed Gate 1 and continuously tracked Gate 2 instead of freezing,
  then hit Gate-2 object `1001` at impact `3.319516`; it sent 445 commands
  at 60.847 Hz with no transport fault. Shadow/parity/smoke/attempt/summary
  hashes are `2f88cdff...`, `5474f8d4...`, `69227537...`,
  `19c2d7af...`, and `0ccc5e8d...`. No unchanged N223 retry is authorized.
- N223 also exposed that the association horizon had inadvertently been fed
  into the pose/rate filter timestep, changing the already-proven H12 control
  signal. N224 keeps the long last-good horizon only for gate membership and
  restores the historical time-since-propagation timestep for state filtering.
  The N223 counterfactual raises terminal Gate-2 closing-rate estimation from
  about 4.8 to 6.7 m/s while preserving the corrected association. The current
  contract/runner SHA-256 values are `473773f0...`/`487195d7...`; all 66
  focused tests pass.
- N224 reset recovery proved boot rollback and fresh index 0/finish -1 with zero
  collision. Passive shadow passed 215 ticks at 21.466 Hz, maximum replay error
  `1.9576e-7`, exact manifests, zero setpoints, and 58.6 Hz no-op cadence.
  Its one official attempt made the first advancement in this branch: official
  Gates 1 and 2 passed, the run stayed collision-free, and transport delivered
  2735 commands at 60.904 Hz with no rate/dropout/drain fault. It then timed out
  at Gate 3/index 2 after the close track projected roughly 2.1 m outside the
  vertical aperture and detector loss let the predictor diverge to an
  impossible -297 m forward target. Finish remained -1. Reset/shadow/parity/
  smoke/attempt/summary hashes are `ca45bcae...`, `6dc130bf...`,
  `75affd8a...`, `45c7ee33...`, `71a873e2...`, and `1fbe3bbf...`.
- The Gate-2 association bug is closed; do not restart association research,
  retrain another reward-only policy, or repeat N223/N224 unchanged. The next
  admitted change must preserve the now-proven clean Gates-1/2 prefix and
  address only Gate 3's projected vertical overshoot and unbounded
  post-plane prediction. It must show replay/native evidence and passive
  parity before another official attempt.

## N225-N228 phase-gating and terminal-control snapshot — 2026-07-18

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. Best progress in this block is
  three official gates; no finish exists, so the goal is not complete. All
  work reused responsive shipping PID `24476` and the exact documented
  disarm/100 ms/one learned-target MAVLink `COMMAND_LONG 31000` reset contract.
- N225 added a Gate-3-only vertical thrust guard. It changes strong sub-hover
  thrust only when the visible trajectory projects above the vehicle at the
  terminal plane. Offline it changed four samples on N224's miss and was exact
  on five prior Gate-3 passes. Callable SHA-256 is `c2f104c1...`; passive
  shadow/parity passed with hashes `0a486625...`/`5a5bd48d...`. Official
  attempt 001 hit Gate-1 object `1001` at impact `6.404631`; the one replacement
  timed out collision-free at index 0 after its unbounded prefix predictor ran
  the stale target to about -287 m. The guard never activated live. Smoke/
  attempt/summary hashes are `928bba5d...`/`d15ac519...`/`265c7a33...` and
  `6fe12fad...`/`bdf1050b...`/`7787f9dc...`. N225 is stopped.
- N226 introduced an explicit phase switch for dropout propagation. It restores
  the historically live-proven non-predictive H12 contract at official indices
  0-2, then enables constant-velocity and roll-aware prediction at index 3.
  The valid shadow passed 299 visible replay samples at 29.9 Hz with maximum
  error `2.3537e-7`, zero setpoints, and 59.886 Hz passive cadence; reset/
  shadow/parity hashes are `c161b10d...`/`aa9ec640...`/`6e4019e7...`.
  Its official attempt used reset `68356 -> 3384 ms`, passed Gates 1-3
  collision-free, and sent 2697 commands at 60.078 Hz. At Gate 4 the predictor
  became active as configured, but the last association propagated to about
  `[-238.9,-0.5,130.5] m`; the run timed out at index 3. Smoke/attempt/summary
  hashes are `38fd8473...`/`3332a9a7...`/`41f6d202...`. This is the first
  three-gate advance in the N220-N228 branch and proves phase gating, not a lap.
- N227 kept N226's proven prefix and replaced only Gate-4 roll/thrust with a
  coordinate-free terminal-plane PD law. Callable SHA-256 is `f0fd8ed0...`;
  77 focused tests and passive shadow/parity passed, with shadow hashes
  `4631d112...`/`f37d3bf9...`. The official run again passed Gates 1-3, but
  missed Gate 4 near an associated `2.0 m right / 4.1 m down` plane estimate
  and struck object `1002` at impact `2.400161`. Smoke/attempt/summary hashes
  are `8c33b715...`/`ec73affa...`/`dc1b33a2...`. N227 is rejected.
- N228 retained N227 lateral control and changed only Gate-4 vertical PD plus a
  bounded post-plane response. Callable SHA-256 is `3531c3be...`; 83 focused
  tests and passive shadow/parity passed, hashes `ff44aacc...`/`e4dd0682...`.
  Both authorized official attempts failed in the unchanged prefix before any
  Gate-4 sample: Gate-3 object `1001` impacts `1.929871` and `0.319780`, each
  at official index 2. Their smoke/attempt/summary hashes are
  `1488266f...`/`472340f5...`/`447707e1...` and
  `c2c2b58f...`/`6ab53499...`/`0f9d1d47...`. N228 is stopped and its target
  mechanism has no live evidence.
- Current source hashes are motion contract `473773f0...`, smoke runner
  `acaeadbb...`, and N209-tail deployment runner `7682ac1d...`. Do not repeat
  N225-N228, add another scalar PD child, or claim that three gates is a lap.
  The next admitted mechanism must be structurally distinct: reuse the existing
  learned phase-reset Gate-4 tail/terminal pulse under the corrected association
  clock and N226 phase-gated observation contract, with passive parity before
  aggregate official evidence.

## N229 learned-tail batch snapshot — 2026-07-18

- N229 implemented that structurally distinct candidate. Variant `PP` keeps the
  H12/N203 learned prefix non-predictive through official index 2, then enables
  constant-velocity dropout propagation at index 3 and dispatches the existing
  learned Gate-4/5/6 checkpoints plus the N206 learned Gate-4 terminal crossing
  pulse. The hybrid runner SHA-256 is `49f39699...`; the batch runner SHA-256 is
  `21e26040...`. Eighty focused tests and the Windows PowerShell parser passed.
- Preflight reset was clean at official index 0 with no collision. Passive
  shadow/parity passed: 274/274 visible samples, maximum action error
  `2.2516e-7`, 27.315 Hz inference, 60.341 Hz passive cadence, zero MAVLink
  setpoints, and exact deployment manifests. Reset/shadow/parity SHA-256 values
  are `d2fd82ca...`/`35bc7889...`/`cd782ccb...`.
- The three-run official batch produced post-stop official gate counts `3, 1,
  2`, median 2 and maximum 3, with zero finishes and zero collision-free runs.
  Run 001 hit Gate-3 object `1001` at impact `1.640297` after 627 commands at
  60.065 Hz. Run 002 hit Gate-2 object `1001` at impact `0.753877` and also fell
  below the 50 Hz transport floor at 48.389 Hz. Run 003 hit Gate-3 object `1001`
  at impact `10.596446` after 656 commands at 61.289 Hz. All failures occurred
  before official index 3, so the learned tail never activated and has no live
  target-mechanism evidence. Batch manifest/score SHA-256 values are
  `150f6d6d...`/`f81bc686...`.
- N229 is rejected as a six-gate deployment and must not be repeated unchanged.
  Best verified progress remains three of six official gates; no valid lap has
  been produced. The live blocker is the high-variance Gates-1-through-3 prefix.
  Before any further Gate-4 work or live run, the next candidate must improve
  prefix collision-free reliability against the accumulated passing and failing
  traces and preserve the corrected association and phase-gating contracts.

## N230-N231 fixed-clock and MAVLink discovery snapshot — 2026-07-19

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N230/N231 run finished; best
  verified progress remains three of six. The exact v3385 simulator and reused
  responsive shipping PID `24476` remain authoritative.
- N230 added an opt-in 60 Hz local-monotonic recurrent-state clock independent
  of camera/inference arrival and the fixed-rate MAVLink publisher. The latest
  observation and previous-action slots are reused for catch-up ticks; no
  future observation is invented. Core smoke/batch-runner SHA-256 values are
  `e6f2d717...`/`70e78ee4...`. Focused Linux and actual-Windows tests passed,
  and passive shadow/parity proved exactly 600 recurrent ticks in 10 s,
  600/600 replay, maximum error `2.210226e-7`, 63.495 Hz no-op cadence, zero
  setpoints, and exact manifests. Shadow/parity hashes are
  `219431c9...`/`351abe78...`.
- The preregistered N230 `PF` batch ran three 45 s attempts at command request
  80 Hz / recurrent state 60 Hz. Official gate counts were `2,3,3` (median
  three), with two collision-free Gate-4 timeouts and zero finishes. Run 1 hit
  Gate-3 object `1001`, impact `10.403496`; runs 2/3 delivered 2700 recurrent
  ticks at exactly 60 Hz and 63.643/63.555 Hz wire cadence with no transport
  fault. The Gate-4 predictor then extrapolated last valid targets near
  `[18.226,9.436,-7.015]` and `[11.826,6.040,-5.104]` m to impossible final
  targets near `[-224,19,95]` and `[-264,0,106]` m, saturating the policy.
  Manifest/score hashes are `580a2681...`/`4e3affbc...`. N230 proves the
  clock mechanism and rejects unbounded Gate-4 prediction.
- N231 retained the fixed 60 Hz clock and removed prediction only for the
  learned Gate-4 terminal tail, using current raw/partial-frame pose. Linux
  passed 110 focused tests; actual Windows passed 70 plus launcher parses.
  Passive shadow/parity again proved exactly 600 ticks at 60 Hz, maximum error
  `2.3588e-7`, 63.401 Hz no-op cadence, zero setpoints, and exact manifests;
  hashes are `5e93180b...`/`9111ca12...`.
- The preregistered N231 `RF` batch produced official counts `1,3,3`, median
  three, one collision-free run, and zero finishes. Run 1 timed out at Gate 2.
  Runs 2/3 reached Gate 4 but the learned tail immediately commanded aggressive
  forward pitch, negative bank, and minimum thrust from entries near
  `[32,3.8,4.2]` and `[32,4.1,4.7]` m; each struck Gate-4 object `1002`
  before entering the terminal-pulse envelope. Impacts were `0.015273` and
  `6.071695`. Manifest/score hashes are
  `2cb79c15...`/`11686157...`. N231 rejects the raw/current-frame tail.
- Together N230/N231 close all three simple Gate-4 observation substitutions:
  indefinite filtered hold, unbounded constant-velocity prediction, and raw
  current-frame control. Fixed recurrence improves timing determinism but does
  not solve Gate 4; another scalar PD wrapper or unchanged tail is not admitted.
- A complete MAVLink site rescan and compliant active-state probe are recorded
  in `docs/mavlink_site_context_2026-07-18.md`. Command `31000` is the standard
  user-defined enum slot `MAV_CMD_WAYPOINT_USER_1`; its reset meaning remains
  simulator-specific. The active VQ1/R1 probe sent zero reset/arm/disarm/
  setpoint/persistent-rate/position requests and received no ACK, interval,
  capability, one-shot `ATTITUDE`, TIMESYNC, or PING response. Probe SHA-256 is
  `f9f3bc4b...`. Standard discovery cannot provide the missing Gate-4 state.
- The next admitted mechanism is a bounded, uncertainty-aware receding-horizon
  visual optimizer, not another reactive tail: fit compact relative-pose and
  attitude dynamics from accumulated official traces, optimize short body-rate/
  thrust sequences in camera-relative coordinates, execute only the first
  action, and replan on every accepted frame. Prediction must terminate at a
  plane/time/uncertainty bound and fail closed. Before flight it must turn the
  recorded N230/N231 Gate-4 entries into centered terminal states in replay,
  preserve the source-hashed Gates-1-through-3 prefix, and pass passive Windows
  shadow/parity.
## N232 bounded visual MPC admission and preregistration — 2026-07-19

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. N232 is admitted for a bounded
  three-attempt batch; it has not yet flown and no finish is claimed.
- N232 fits only current raw aperture poses and observable attitude/rates from
  `118` official reports (`276` segments, `3,672` raw samples). The constrained
  five-fold model has nonnegative thrust/drag terms and no fitted forward bias.
  Model SHA-256 is
  `9aaefaae111734314353757adbd49da913f13c9bb558627e382ba3c2eb8cc275`;
  fitter/controller/evaluator hashes are `69bf3004...`/`3f0fe8f1...`/
  `d3d37212...`.
- The Gate-4 controller carries only observable Gate-3 relative velocity,
  rejects incoherent/far aliases, anchors an inertial frame on the first
  coherent raw Gate-4 pose, optimizes bounded pitch/roll/thrust knots against
  the leave-one-fold-out model ensemble, executes the first action, and stops
  prediction at the gate plane, `2.0 s`, `0.30 s` measurement age, or `0.80 m`
  uncertainty. It changes only official index `3`; unit evidence preserves
  parent outputs exactly at indices `0-2` and `4-5`.
- Deployment uses `32` candidates, two refinements, a deterministic `10 Hz`
  logical clock, one recurrent update per completed live update
  (`PolicyStateHz=0`), and fresh-pose replanning no faster than `0.18 s` with a
  `0.25 s` hard maximum. This supersedes N230/N231's infeasible wall-clock
  catch-up: the exact Windows parent benchmark sustains only about `18.5 Hz`;
  optimized/cached N232 calls measured about `172/52 ms`.
- The authoritative `0.20 s` held-action leave-one-fold-out counterfactual
  passes all `20/20` N230/N231 Gate-4 entry/member rollouts. Worst absolute
  terminal offsets are `0.478516 m` right and `0.296263 m` down; maximum
  crossing time is `12.2 s`. Evidence SHA-256 is
  `ee352fd072a5cb49f778ba3908375d805e1c513db2ba9cbbdffc17564543e6a0`.
- A latest-frame transport correction now lets the bounded UDP socket discard
  obsolete retransmitted camera chunks during the `15 Hz` cooldown instead of
  starving the control loop. Camera receiver SHA-256 is
  `bac1c5d3fc5c0b8b509ffe81f6f49af19a6c1ecb3fe9401172c3f420b169dfeb`.
  Linux and actual Windows each pass `70/70` focused tests; both PowerShell
  launchers parse.
- Passive `n232_visual_mpc_shadow_004` passed for `10.047 s`: `129` live and
  replay ticks at `12.8397 Hz`, maximum action error `2.09942e-7`, `86` camera
  frames/detections, no telemetry dropout/drain-limit hit, exact manifests, and
  `59.017 Hz` no-op cadence. It sent zero reset, arm, disarm, or MAVLink
  setpoints and left official index `0`, finish `-1`, on responsive PID
  `24476`. Shadow/parity hashes are `ed32bc52...`/`a0bd5afc...`.
- Preregister exactly variant `MO`, tag
  `competitive_n232_visual_mpc_batch_001`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `0`, target six gates, and stop on the
  first valid finish. Every attempt must reuse the responsive simulator and
  prove the normal disarm/`100 ms`/one learned-target MAVLink
  `COMMAND_LONG 31000` reset rollback, fresh index `0`, and finish `-1`.
  Dry manifest SHA-256 is
  `3a991bb73ba057b49b87d3e3720cbbd1ff0c4df4a443ff59883cb7d537d89b09`.
## N232 official batch result and N233 admission — 2026-07-19

- The preregistered N232 `MO` batch is complete and rejected. Official gate
  counts were `1,2,2` (median `2`, maximum `2`), with zero collision-free
  attempts and zero finishes. All failures were in the unchanged prefix before
  official index `3`, so the Gate-4 optimizer never activated and still has no
  live target-mechanism evidence.
- Run 1 hit Gate-2 object `1001` at `7.375 s`, impact `9.523264`, after
  `463` commands at `63.724 Hz`. Runs 2/3 passed Gates 1-2 then hit Gate-3
  object `1001` at `10.375/10.625 s`, impacts `10.404356/8.742578`, after
  `653/672` commands at `63.709/63.905 Hz`. All three had zero command-rate
  violation, telemetry dropout, or drain-limit hit.
- Final manifest/score SHA-256 values are
  `0fdee072f729c0252b1548e56d5adbae2ac2ecc66c4ec3f285ee3c21f3658d26`/
  `88c18407401074d851607ddafcccbf99a80330a35392c3d90408be130cd10e6e`.
  Do not repeat N232 unchanged.
- The causal regression is the `PolicyStateHz=0` prefix compromise, not the
  measured recurrent network cost. A direct exact-Windows benchmark runs the
  deployed FP32 recurrent checkpoint at about `6,320 Hz`; N232 instead paid
  about `50 ms` per call because it resolved, read, and SHA-256 hashed the
  dynamics JSON on every inference, including Gates 1-3.
- N233 is admitted as the smallest corrective mechanism: launcher/hash-verify
  the same model once, cache that immutable verified model in the callable,
  restore the N230/N231 `60 Hz` recurrent prefix clock, and retain N232's same
  raw-pose tracker, bounded `32 x 2` optimizer, uncertainty/plane/time stops,
  and approximately `0.20 s` held-action cadence only while Gate 4 is active.
  It must pass source-hash tests, exact prefix parity, an actual-Windows timing
  benchmark, and passive live shadow before any bounded official batch.

## N233 fixed-rate visual MPC preregistration — 2026-07-19

- N233 passed every admission gate. Its callable SHA-256 is
  `4a3f0c7a0c8b73053f6aa7da8995b4fffe1e2c8ed2f430c79135476ca8f1c4ba`;
  the pinned hybrid/batch launcher hashes are `6fc426e0...`/`2af447bb...`.
  Linux and actual Windows each pass `70/70` focused tests, both launchers
  parse, and `git diff --check` passes.
- On actual Windows, 600 Gate-1 prefix decisions took `0.24733 s`
  (`2,425.9 Hz`). The first Gate-4 optimized decision took `0.13724 s`, the
  cached following decision took `0.00994 s`, and model validation remained
  exactly once. This is safely inside the fixed `60 Hz` prefix and roughly
  `5 Hz` bounded Gate-4 replan contract.
- The existing responsive simulator PID `24476` was reused without relaunch.
  Same-process UI recovery cleared the prior terminal `base_mode=65`,
  `system_status=3` state and restored VQ1/R1 at official index `0`, finish
  `-1`, with `base_mode=193`, `system_status=4`.
- Passive tag `n233_visual_mpc_fixed_shadow_005` passed: exactly `600` live
  and replayed decisions at `60 Hz`, maximum action error `2.340714e-7`, `142`
  completed camera frames (`126` detector frames), zero telemetry dropout or
  drain-limit hit, zero MAVLink setpoints, and zero reset/arm/disarm commands.
  Shadow/parity hashes are `d781b6f4...`/`1c3d3812...`.
- Preregister exactly variant `MC`, tag
  `competitive_n233_visual_mpc_fixed_batch_002`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Each attempt must reuse the responsive
  simulator and preserve reset rollback, fresh index `0`/finish `-1`, health,
  collision, command-rate, and command-free post-stop evidence. Dry manifest
  SHA-256 is
  `d52519417daf7bcd0a2ebf6faf7ed053d4d4a426353ede27855e1f3a2779c641`.
+

## N233 official batch result and N234 admission — 2026-07-19

- The preregistered N233 `MC` batch is complete and rejected unchanged.
  Official gate counts were `2,3,3` (median/maximum `3`), with zero finishes
  and zero collision-free attempts. This improves N232's `1,2,2`, and runs
  2/3 activated Gate 4, so the target mechanism now has live evidence.
- Run 1 collided at official index `2` after `10.719 s` with object `1001`,
  impact `9.212901`, and `662` commands at `61.938 Hz`. Runs 2/3 reached
  official index `3`, then collided with object `1002` after
  `16.015/36.750 s`, impacts `2.621719/6.141685`, and `1013/2333`
  commands at `63.436/63.563 Hz`. Every run had zero command-rate violation,
  telemetry dropout, or drain-limit hit and proved the normal single-reset
  boot rollback/fresh-start sequence.
- Final manifest/score SHA-256 values are
  `3f577a381e1a5006d3c6f2c198349c95e5f582180913939ac5dd7e4cdc95bfa2`/
  `7c20ccb5297ad39cc5314a991c39ad47a2dcdf0a7c7f15182d72e9cc0c6ca114`.
  Do not repeat N233 unchanged.
- The new causal evidence is a Gate-4 track-identity/staleness failure. Run 2's
  observable target jumped from `[68.57,1.07,1.29] m` at `10.4 s` to
  `[33.10,6.00,2.12] m` at `10.9 s`, later reached
  `[3.39,-1.07,-0.36] m` at `15.2 s`, then jumped to
  `[19.59,-4.04,-8.23] m` one sample later without an official advance.
  Run 3 lost all target poses after `16.7 s` but held the stale saturated
  action `[1.0,-0.9,0.6523,-0.0322]` for about `20 s` until collision.
- Admit only an N234 stale-track/reacquisition correction next: do not alter the
  proven 60 Hz prefix or bounded optimizer. Expire aggressive cached actions
  after the 0.30 s observable age limit, reset the Gate-4 tracker after a
  bounded rejection/staleness interval, require coherent fresh samples before
  reacquisition, and use a neutral bounded brake/search action while no track
  is admitted. First capture the callable controller snapshot and replay the
  N233 Gate-4 traces offline; require that no stale saturated action can persist
  and that Gates 1-3 remain bit-exact before any new official run.
+

## N234 stale-track reacquisition preregistration — 2026-07-19

- Fixed-clock replay of the two live N233 Gate-4 traces confirmed the cause.
  N233 produced only `2` accepted optimizer plans in each trace but
  `56/121` failsafe plans; final reasons were `measurement_stale`, with
  saturated pitch/roll commands. Replay artifacts for runs 2/3 have hashes
  `ad0ed381...`/`b97fa758...`.
- N234 preserves N233's exact parent, 60 Hz clock, verified dynamics ensemble,
  optimizer, and Gate-3 velocity carry. It changes only observable association
  failure handling: accept initial targets out to `75 m`, require three fresh
  coherent poses, reset after `0.30 s` staleness, and command bounded
  `[0.25,0,0,current_yaw]` brake/hover whenever acquisition or planning is
  rejected. Callable SHA-256 is
  `84c08dac68be66039567173a7319831232601c22790792588a044401d7671530`.
- Counterfactual replay on N233 runs 2/3 performed `3/6` track resets,
  admitted `5/4` optimized plans, and ended with neutral cached actions
  instead of stale saturation. The run-3 maximum recorded plan-event magnitude
  fell below `0.934`; replay hashes are `60427632...`/`a7b3fbd3...`.
  The relevant suite passes `88/88` on Linux and actual Windows, both
  PowerShell launchers parse, and `git diff --check` passes.
- PID `24476` was again reused without relaunch and VQ1/R1 recovered to index
  `0`, finish `-1`, `base_mode=193`, `system_status=4`. Passive tag
  `n234_reacquiring_mpc_shadow_006` passed with exactly `600` live/replayed
  decisions at `60 Hz`, maximum action error `1.711920e-7`, `116`
  detector frames, zero telemetry dropout/drain-limit hit, zero setpoints, and
  zero reset/arm/disarm commands. Shadow/parity hashes are
  `94c006da...`/`91ebc9d8...`.
- Preregister exactly variant `MR`, tag
  `competitive_n234_reacquiring_mpc_batch_003`, three attempts, `45 s`
  each, command request `80 Hz`, policy state `60 Hz`, target six official
  gates, and stop on the first valid finish. Preserve the same reset, health,
  collision, cadence, and command-free post-stop proofs. Dry manifest SHA-256
  is `31d3fe386b257764def577503e05ac1f1186845cef6de50a3fc1c3d5772eefdd`.
+

## N234 official result and N235 admission — 2026-07-19

- The preregistered N234 `MR` batch is complete and rejected unchanged.
  Official counts were `1,3,2` (median `2`, maximum `3`), with zero
  finishes and zero collision-free attempts. Runs 1/3 failed in the unchanged
  prefix at `7.313/10.578 s`; run 2 alone reached Gate 4 and failed after
  `42.281 s`.
- Runs 1/3 hit objects `1001/1001` with impacts `2.375203/10.908347`.
  Run 2 hit object `1002` at official index `3`, impact `10.075733`.
  Effective command rates were `62.069/63.479/63.147 Hz`; every run had zero
  command-rate violation, telemetry dropout, or drain-limit hit and a proven
  normal reset rollback/fresh start. Final manifest/score SHA-256 values are
  `18b3b49fcc5ae3f86d1643f4a5aa026e3874c7b25e494461c775e3ea5c939e35`/
  `5d22e174e69a65945092ea0eed9a47d19f8d0121d8386cfd832adb6157c81c02`.
- N234 fixed N233's stale-saturation defect live, but exposed the next cause.
  In run 2 it entered Gate 4 at `10.3 s`; its only non-neutral interval was
  approximately `13.3-13.6 s`. The visible target then moved to
  `[16.0,13.1,9.8] m` with yaw error `+0.686 rad` by `17.2 s`, while
  neutral brake/hover did not steer toward it. All poses disappeared after
  `17.2 s`, and the controller correctly stayed at bounded
  `[0.25,0,0,current_yaw]` rather than saturating until the late collision.
- Fixed-clock replay of that live trace reports only `1` accepted optimizer
  plan, `18` rejected plans (`14` state-uncertain, `4`
  prediction-uncertain), `4` track resets, and `1704` acquisition-wait
  calls. Replay SHA-256 is `8eb06a63...`.
- Admit only N235 observable acquisition/search fallback next. Preserve N234's
  prefix, optimizer, coherent admission, stale reset, and neutral safety
  bounds, but when a raw current Gate-4 pose exists, use a bounded low-gain
  pitch/roll/thrust/yaw acquisition servo to keep the target centered; when it
  leaves frame, carry/search in the last observed yaw direction for a bounded
  interval. No map coordinates or privileged simulator state. Prove bounded
  offline replay, exact non-Gate-4 invariance, Windows/Linux tests, and passive
  shadow before any flight.
+

## N235 observable acquisition/search preregistration — 2026-07-19

- N235 keeps N234's exact verified parent/model, 60 Hz clock, coherent tracker,
  stale reset, and accepted optimizer path. Its only new behavior is a bounded
  no-plan acquisition action: far/medium/near pitch `-0.12/0/+0.25`, raw
  lateral gain `0.03/m` capped at `0.55`, raw vertical gain `0.04/m`
  capped at `0.50`, yaw step capped at `0.25 rad`, and a `0.10 rad`
  last-direction yaw search for at most `2.0 s` before brake/hover. Callable
  SHA-256 is
  `dd8cd8d89f070c9ca51820c18a9565bb61eb478b1dcd9cce464d4fd4674c569a`.
- Counterfactual replay of N234's live Gate-4 trace produced `221` visible
  acquisition calls, `104` bounded blind-search calls, then `1397` blind
  brake calls; maximum recorded plan-event magnitude remained `0.9`.
  Replay SHA-256 is `2c3392a7...`. The relevant suite passes `92/92` on
  Linux and actual Windows, both launchers parse, and `git diff --check`
  passes.
- PID `24476` was recovered and reused without relaunch at VQ1/R1 index `0`.
  Passive tag `n235_search_mpc_shadow_007` passed with exactly `600`
  live/replayed decisions at `60 Hz`, maximum action error `1.992399e-7`,
  `127` detector frames, zero telemetry dropout/drain-limit hit, zero
  setpoints, and zero reset/arm/disarm commands. Shadow/parity hashes are
  `dffebee3...`/`b26ffe9e...`.
- Preregister exactly variant `MS`, tag
  `competitive_n235_search_mpc_batch_004`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Preserve all reset, health, collision,
  cadence, and command-free post-stop proofs. Dry manifest SHA-256 is
  `f61c389f7a0950ecfa1bd3a35b97ed52d34dc5d7d0f3e10834a1e82179117e2b`.
## N235 official result and N236 admission — 2026-07-19

- The preregistered N235 `MS` batch is complete and rejected unchanged.
  Official counts were `2,0,3` (median `2`, maximum `3`), with zero
  finishes. Run 1 collided with object `1001` at `7.171 s`, impact
  `0.211197`; run 2 remained collision-free but passed no gate in `45 s`;
  run 3 reached Gate 4 then hit object `1002` at `14.906 s`, impact
  `6.979586`. All command rates were `63.5-63.9 Hz` with zero command-rate
  violation, telemetry dropout, or drain-limit hit and valid reset proofs.
- Final manifest/score SHA-256 values are
  `c1096ed17d849a103116a2d99784ead97cdc480f377ab9f004419eef4659b05d`/
  `0e079b44075beeeffda856b13bd24f9bf5426721a5aa8c76a978a17bd53b56ee`.
  Do not repeat N235 unchanged.
- N235 succeeded at its intended acquisition mechanism: the Gate-4 target
  remained visible in `38/44` retained samples. It failed because acquisition
  pitch was forward before centering: at `10.6 s` the target was
  `[40.0,4.75,6.38] m`; at `12.7 s` it was still
  `[23.41,4.79,7.32] m`, yet pitch remained `-0.12`, and collision followed
  at `14.9 s`.
- Admit only N236 staged acquisition next: preserve all N235 lateral, vertical,
  yaw, search, tracker, and optimizer behavior, but command brake pitch while
  raw right/down/yaw errors exceed the existing center envelope. Permit gentle
  forward pitch only after centering; blind search also brakes. Validate this
  exact causal change offline and through the normal test/shadow gates.

## N236 staged Gate-4 preregistration — 2026-07-19

- N236 preserves N235's exact observable tracker, lateral/vertical/yaw search,
  optimizer, and fixed `60 Hz` policy clock. Its only causal change is staged
  pitch: brake at `+0.25` while the raw target lies outside the existing
  `1.5 m` right/down and `0.12 rad` yaw center envelope; allow N235's gentle
  forward pitch only after centering; and brake during blind yaw search.
  Callable SHA-256 is
  `fe7984c17b203ade4b3967ff0e6166be17c4d9a89a827231e2e34be6e28862f4`.
- Counterfactual replay over N235's actual Gate-4 collision trace produced
  `63` visible uncentered-brake decisions, `35` blind-search brake decisions,
  zero centered-forward decisions, and ended in brake rather than repeating
  the collision-producing forward command. Replay SHA-256 is
  `38760787cc32ca88f4f0f73996550a7fbd1debfa1be6d3756d6cfc3917186f52`.
- The relevant suite passes `96/96` on Linux and the actual Windows runtime;
  both PowerShell launchers parse. Responsive PID `24476` was recovered and
  reused without relaunch at VQ1/R1 index `0`. Passive tag
  `n236_staged_mpc_shadow_008` passed with exactly `600` live/replayed
  decisions at `60 Hz`, maximum action error `2.555741e-7`, `122` detector
  frames, zero telemetry dropout/drain-limit hit, zero setpoints, and zero
  reset/arm/disarm commands. Shadow/parity SHA-256 values are
  `310bf2b3e8c68a6e8eb6aa35b027c9a9c70a4fc1730c4c5616cfb4a296bdb73f`/
  `2a023bfecadb565c1d1cbd584f1d657950b2971905abaa4ca69204cdd7d60226`.
- Preregister exactly variant `MB`, tag
  `competitive_n236_staged_mpc_batch_005`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Preserve all reset, health, collision,
  cadence, and command-free post-stop proofs. Dry manifest SHA-256 is
  `3c818c036e240e60bf2f61d09cea24fbd297339ac951b59940828c34698a0105`.

## N236 official result — rejected — 2026-07-19

- The preregistered N236 `MB` batch completed unchanged. Every attempt passed
  official Gates 1-3, but none passed Gate 4 or finished: counts were `3,3,3`.
  Run 1 was collision-free for `45 s`; run 2 hit object `1002` at
  `17.75 s`, threat `1`, impact `0.010981`; run 3 hit object `1002` at
  `17.187 s`, threat `2`, impact `10.552031`. Effective command rates were
  `61.930/63.549/63.302 Hz`, with zero command-rate violation, telemetry
  dropout, or drain-limit hit. Final manifest/score SHA-256 values are
  `ff595221184c7f710a1c24f0831023738946a9d639a94810d6f52e513bff49f4`/
  `50595bb65f3970d1ef5c32786f66d273b3236660231f1da22b16efbf811ab0fe`.
- N236 causally removed N235's immediate uncentered forward drive, but the
  target mechanism remains rejected. Gate-4 entries repeatedly appeared near
  `30-38 m` forward, `4-7 m` right, and `5-7 m` down. The staged controller
  braked forward, yet raw detections jumped between inconsistent range/bearing
  families; lateral/vertical errors grew or reversed, only `1-4` optimized
  plans were admitted per replay, and run 1 lost the target after `19.1 s`.
  Live replay SHA-256 values for runs 1-3 are `a25bd2b...`, `cfa621b7...`,
  and `1d944348...`. Do not repeat N236 unchanged.
- Standard MAVLink discovery is closed as a source of the missing state. The
  next mechanism must use legal camera/IMU/action-history observations and fix
  visual target identity/bearing consistency before optimizing control. Do not
  reopen persistent message requests, hidden-position assumptions, stale-track
  carry, or another renamed scalar Gate-4 PD wrapper.
- After the batch, responsive PID `24476` was recovered through the UI without
  relaunch. Active VQ1/R1 proof is base/status `193/4`, official index `0`,
  finish `-1`; recovery SHA-256 is
  `56d34547a470ed34157c030e958c1d4b90e3d52bdc094b10fef1d08ee3080965`.

## N237 action-observer Gate-4 admission and preregistration — 2026-07-19

- N236 failed because range-family jumps corrupted camera finite-difference
  velocity after the Gate-3 handoff. The first N237 draft retained that
  velocity estimate and was rejected offline at `0/15`; it was never flown.
  The admitted N237 instead carries the reliable Gate-3 entry velocity through
  the fitted ensemble-mean translational plant using the actual action history
  and observed attitude. Fresh camera observations correct position but never
  replace velocity with raw range differences. It requires three coherent
  fresh frames, caps initial range at `40 m`, limits observer steps to `0.10 s`
  and uncertainty to `2.0 m`, and uses projected-intercept actions only while
  the observations remain fresh and coherent. All stale, missing, incoherent,
  or unapproved cases retain N236's staged brake/search and hard `0.30 s`
  reset. No hidden simulator state is used.
- Admitted N237 source SHA-256 is
  `6fc71223f11246bc90d0c70df41cf67e5bc3e721ae83e6b66c05fcfe1720cb5d`.
  The actual Windows deployment suite passed `178/178` tests in `41.62 s`,
  both PowerShell launchers parsed, and `git diff --check` passed. All three
  N236 traces passed all five held-out causal folds (`15/15` total); worst
  projected absolute right/down error was `0.04378/0.02395 m` and maximum
  crossing time was `9.3 s`. Per-trace evaluator artifact SHA-256 values are
  `9a0b7785b6452389db2c6bb657d01f19e8c53c0b07f3ad04d51b0ae6a33874c7`,
  `d5e3b62ec90ca726c759a486ec52befbad165c554822da51d4b8d063c4f8f8b5`,
  and `340c1fa44d231eca0c8140854b92be88cfa99acd2714b787ddce61cf2397eaba`.
  Exact controller replays are
  `63b19c3eb3e0074e1412f30df3b42667fcb036aec3a4adfdd7f82b8df6e1984c`,
  `6d7817788e770608f7861140bced52d8264d0e1f27100d567e9c478438d5054e`,
  and `c22836f1a0230bf89c96eb15e9fb9b2ac73081bf8612da81f28350dce757bdd0`.
- Mandatory live passive shadow passed: `600/600` fixed `60 Hz` samples,
  maximum replay action error `2.2779426575580963e-07`, `127` detector
  frames, zero setpoints/reset/arm/disarm commands, zero collision/dropout/
  drain-limit hit, and no-op cadence `63.902 Hz`. The existing simulator
  remained active VQ1/R1 at base/status `193/4`, official index `0`, finish
  `-1`. Shadow/parity SHA-256 values are
  `9fcc4fdaf466d521b92bb37cea8ec3c381e2d436310fcf12177e0ffc73b881bb`/
  `baf9367ebd902358a2e887b52e28aeae4af8b756ca6a6ff001961f4d93920eec`.
- Preregister exactly variant `MI`, tag
  `competitive_n237_action_observer_batch_006`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Use the already-running responsive
  `FlightSim.exe` PID `24476`; never relaunch it. Preserve the documented
  normal disarm/`100 ms`/exactly-one command-`31000` reset and all health,
  collision, cadence, fresh-start, and command-free post-stop proofs. Dry
  manifest SHA-256 is
  `bf6dc909e1f6e0eb5824f2a609f74ce9c5111d98cf4b991908afd40dd7062ddf`.

## N237 official result — rejected — 2026-07-19

- The preregistered N237 `MI` batch completed unchanged. All three attempts
  passed official Gates 1-3, stayed collision-free for the full `45 s`, and
  failed Gate 4: counts `3,3,3`, finish `-1` throughout. Effective command
  rates were `63.621/63.354/63.622 Hz`, with zero rate violation, telemetry
  dropout, or drain-limit hit. Every attempt proved a fresh boot rollback and
  the required reset lifecycle. Final manifest/score SHA-256 values are
  `183cc8fb7c7f413e5ec6058e1a0f8e1cc57ad1694b02ed3df3d4124e187f2442`/
  `114a6be0681cb455344a5cec9bf072f409e7e90cd7163008c6aa80700a74693f`.
- N237 is rejected live despite the offline counterfactual. Raw closest
  Gate-4 poses were only `3.77/2.42/2.74 m`, `9.60/6.53/4.02 m`, and
  `10.05/7.94/4.30 m` forward/right/down before the apparent target moved
  away. Replay admitted `28/13/16` optimized plans but invoked the new
  rejected-plan projected-intercept branch `0/0/0` times and reset its track
  `3/4/4` times. Each hard reset admitted a discontinuous range/bearing family
  (for example run 1 jumped from about `12 m` back to `24 m`; runs 2/3
  repeatedly jumped between `10-35 m`). The inherited no-plan acquisition
  also continued steering from raw rejected poses. This is the exact target-
  identity defect N236 exposed; N237 corrected velocity propagation but did
  not correct association.
- The observer/model is also not admissible unchanged: its replayed forward
  velocity ended at `-0.56/-3.91/-6.29 m/s` while associated/raw detections
  remained `10-24 m` forward, producing large family-dependent lateral and
  vertical actions. Replay artifact SHA-256 values are
  `7e1faf798ae579a8d2615b625e3b4a2ab17a95a2b5d09c001172922d9dce7450`,
  `3ebfe4db9f7dfb9dabe7cc86c26500904a80d7c5b8dc023fdb68951e8340df13`,
  and `da3fb3f198b9833004673e8bb6a4b0f7865780f10c52838752021afa0585c386`.
  Do not repeat N237 or tune its fitted dynamics.
- The next admitted mechanism must own one persistent observable Gate-4
  target family across detector gaps, reject farther/discontinuous aliases,
  and ensure rejected raw poses cannot affect pitch, roll, thrust, or yaw.
  Association failure must produce bounded level/brake behavior; only a
  coherent associated target may drive an intercept. Preserve the exact
  official-proven Gates-1-3 prefix, learned Gates-5-6 tail, reset contract,
  and existing simulator process.

## N238 persistent identity-lock admission and preregistration — 2026-07-19

- N238 replaces N237's complete Gate-4 tracker/MPC/raw-fallback path while
  preserving the exact parent actions outside official Gate 4. It acquires one
  camera-relative family only after three coherent fresh poses no farther than
  `40 m`. A current pose is accepted only inside the bounded continuity and
  `2.5 m` forward-regression envelope. A discontinuous family can re-enter
  only after three coherent samples and at least `3.0 m` closer progress;
  persistent farther families never re-anchor. Rejected poses have zero
  pitch/roll/thrust/yaw authority. After `0.25 s` without association the
  output is bounded level/brake; a `1.0 s` forward commit is possible only
  from a fresh associated pose at at most `8 m` with projected right/down at
  most `1.0/1.0 m` and yaw error at most `0.15 rad`. Runtime inputs remain
  only the legal camera pose, attitude, official phase, and action history.
- Callable SHA-256 is
  `2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc`.
  All three N237 live traces pass the exact identity-lock evaluator. They
  exercise `40/49/45` rejected aliases and `38/47/44` identical-state
  rejected-alias versus missing-frame parity comparisons, with maximum action
  difference exactly `0.0`. Maximum accepted forward regression is only
  `1.4765/0.9767/1.5716 m`; each trace has exactly one initial family.
  Evaluator artifact SHA-256 values are
  `3e9df7d5864a8e0a28d4ecf4e00ceabc541747ed245425e1a2dfcad8da59e481`,
  `3d48f94cd4d3082e8605e7d8df574b17f9d85705422d76401be1c9cf3d85f537`,
  and `9f49477125f8281f79d6d1be4c5f4bb982d2cc7f92a24852b6ba0eb7a0068007`.
- The actual Windows deployment regression passed `210/210` in `46.03 s`;
  both launchers parse and `git diff --check` passes. Hybrid/batch launcher
  SHA-256 values are
  `660c1113a69a92d6106d873c965c0265072a6d079f775448286ae57dd6c82902`/
  `42aeefb2c4f50a26596f840bf7684625f91cfbf4073b043ab71b2bdbb609cba3`.
  PID `24476` was recovered through the UI without relaunch to active VQ1/R1,
  base/status `193/4`, index `0`, finish `-1`; recovery artifact SHA-256 is
  `95ef599eb8d935ba6a1cc17723ee8315beb82975c78227cbc8821c22276ac46c`.
- Mandatory command-free shadow passed with `599/599` fixed `60 Hz` decisions,
  maximum replay error `2.707422256387204e-07`, `127` detections, no-op
  cadence `63.702 Hz`, zero setpoints/reset/arm/disarm commands, and zero
  collision, telemetry dropout, or drain-limit hit. The race remained active
  at index `0`, finish `-1`, base/status `193/4`. Shadow/parity SHA-256 values
  are `d9a313414ea394497826f00ee879ea7c4b17dbbd24f176634e72576d18161658`/
  `cff1e2a42886a7407e4679c0506f68fb9a1bb7575845faa10f7cf74918bbbdb7`.
- Preregister exactly variant `IL`, tag
  `competitive_n238_identity_lock_batch_007`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Reuse PID `24476` without relaunch and
  preserve the documented reset/fresh-start, health, collision, cadence, and
  command-free post-stop proofs. Dry manifest SHA-256 is
  `1b34f95430f705f0139ce0d8e3b0215079291aa77746dc72e3454c6576546b62`.

## N238 official result — rejected — 2026-07-19

- The preregistered N238 `IL` batch completed unchanged with official counts
  `3,3,3`, zero finishes, and zero collision-free runs. All three attempts hit
  Gate-4 object `1002` at `16.781/17.078/17.250 s`; threat/impact values were
  `2/1.183825`, `1/0.051233`, and `2/1.017933`. Effective command rates were
  `63.162/63.881/63.884 Hz`, with zero rate violation, telemetry dropout, or
  drain-limit hit and valid fresh resets. Final manifest/score SHA-256 values
  are `158c1ea0179688bd0ed48c3765a551631f35557d1663ca1445f74b3015ed70ef`/
  `122bf6b103ef92fbf04ab5ec8b43002d92d806fd8e5b03de1e765b929b946f9b`.
- Identity rejection worked but the control law failed. Replays retained zero
  rejected-alias action error, `25/33/30` associated fresh samples, and only
  `1/1/2` persistently closer re-associations. The crossing commit triggered
  zero times. Closest raw poses were about `5.11/3.61/-1.53 m`,
  `4.80/2.46/-1.35 m`, and `3.79/1.88/-1.24 m` forward/right/down while
  the controller still commanded brake pitch and negative roll. Associated
  lateral rates were still `+5.3` to `+6.7 m/s` and yaw errors
  `0.46-0.61 rad`; the direct projected servo counter-banked too late and hit
  the aperture/frame. Live identity-evaluation SHA-256 values are
  `bc9b169b83c80c6e89c3f68cb727726863bec0eab1a17549b329f330c2c982a1`,
  `17643976350cd66b80283b1de69b14425f9ab7aedf6a8ced13c0bc313a112ad2`,
  and `0fe620a1902134a2980fd6e8cac1621207e3d153fb4e0b58f14417d71b079e01`.
- Do not repeat N238, relax its commit envelope, alter a scalar PD gain, or
  treat closer-family confirmation as solved identity. The traces now provide
  clean associated official input/output segments. The next admissible work is
  offline system identification and controller optimization on those segments,
  with held-out whole-run prediction/control evidence before another flight.

## N239 trained identity-body lateral MPC admission and preregistration — 2026-07-19

- N239 is the required offline system-identification/controller-optimization
  successor to rejected N238, not another scalar PD adjustment. The v2 fit
  reconstructs locally coherent N238 association segments from all six N237/
  N238 official traces, excludes every transition across a family re-anchor,
  and uses only recorded official actions. It provides `113` clean transitions
  and six leave-one-whole-run-out structured models. Selected ridge is `10`;
  mean held-out next-rate RMSE is `2.2992/1.3313/0.7128 m/s` and the aggregate
  score is `1.447774`. Only roll-to-right acceleration is deployable across
  every fold: its gain range is `3.1081-3.9474 m/s^2` per normalized roll.
  Pitch/forward and thrust/down action gains change sign across folds and are
  explicitly forbidden from optimization. Fit source/model SHA-256 values are
  `5952461d3756b2c53c5de10c6d60c1cc8ab9eab4f65ad10bb41f0bb6c11174bf`/
  `a40e9279bc2d34f69a2296350971ee27eb1393f64e5527b37f2bdc1a545677df`.
- The first counterfactual draft was rejected offline and never flown. It tried
  to center every identity-associated family and passed only `17/42` rollouts;
  all `24/24` far-family rollouts remained uncontrollable even at roll `-1`.
  N239 therefore quarantines far associations: they have zero action authority
  and command N238's bounded level/brake. The learned optimizer is unlocked
  only when a newly acquired or newly re-acquired coherent family first appears
  at at most `16 m` forward. It then runs deterministic five-knot, `384`-
  candidate, four-iteration robust lateral MPC over the fold ensemble and
  executes only the first roll. N238's association/rejection, pitch, thrust,
  yaw, dropout, and commit rules remain exact; rejected raw poses still have
  zero action authority. Runtime inputs remain legal camera-relative pose,
  attitude, official phase, and action history, with no simulator truth or map.
- The admitted counterfactual selects the three live near-family N238 entries
  at `14.328/-4.657`, `14.118/-5.449`, and `11.566/-6.452 m` forward/right.
  For each entry, every one of six dynamics members is used as the plant while
  being excluded from the planner. All `18/18` rollouts cross within the
  `1.0 m` lateral box; worst terminal absolute right error is `0.969814 m`.
  Core/evaluator/counterfactual SHA-256 values are
  `1059acff7135bbdd7bb20fc9630891d28cda7794630a3a51a590fc014d96338b`,
  `6a67110324625fa2c7bb6eb2945bb303e3ffc85f4078596a072b91d6e15e77e2`,
  and `4c631c0cbcfd688457433a8000cf7b9601a3ecf2992e62a732d9028268222ea3`.
  The source-pinned N239 callable SHA-256 is
  `716352e7b2ec603e0316e59f0d97cc6fe96ca5bdde4f5f7923b0be2fb6b75f4c`.
- A command-free deployment shadow exposed and corrected one runtime defect:
  the first callable re-hashed the UNC model on every `60 Hz` state step,
  causing a catch-up feedback loop. Shadow 011 passed only `205` decisions;
  shadow 012 reached `592` but failed with two drain-limit hits. Model hash
  validation is now once per controller construction and is regression-tested.
  The final actual-Windows deployment suite passed `561/561` in `119.20 s`;
  both launchers parse and `git diff --check` passes. Hybrid/batch launcher
  SHA-256 values are
  `6ca53a4e3945402b770175180a14818755ebce137e28ccc613b4ecaba8d5f85c`/
  `4a3aa2d2625ebfdaca54f3e9d613da9a7b62dc379f281e43594fbf54641ee5f9`.
- Responsive shipping PID `24476` was recovered through the VQ1/R1 UI without
  relaunch; recovery SHA-256 is
  `3b41554476ea69b2a651c66f62781d3871a6a5bbfcda9c74b1c8145c2574b995`.
  Final mandatory shadow 013 passed `600/600` fixed-clock decisions at
  `59.910 Hz`, `128` detections, maximum replay action error
  `2.5794487001906674e-07`, and no-op cadence `63.804 Hz`. It sent zero
  setpoints/reset/arm/disarm commands and recorded zero collision, telemetry
  dropout, or drain-limit hit. The race remained base/status `193/4`, official
  index `0`, finish `-1`. Shadow/parity SHA-256 values are
  `fc7b1d370c9b45a45c50ff5757cd73f06091a5e1443723bfb91c02936d5dcb96`/
  `7869523662a571023e5a65905c47102dfbd6c957e9f0d3e63361b77f005742c4`.
- Preregister exactly variant `LM`, tag
  `competitive_n239_identity_body_mpc_batch_008`, three attempts, `45 s`
  each, requested command `80 Hz`, policy state `60 Hz`, target six official
  gates, and stop on the first valid finish. Reuse responsive PID `24476` and
  never relaunch it. Each attempt must retain normal disarm, `100 ms` wait,
  exactly one learned-target `COMMAND_LONG 31000` with confirmation/parameters
  zero, boot rollback over `1000 ms`, fresh index `0`/finish `-1`, all health
  and cadence checks, and the command-free post-stop proof. Dry manifest
  SHA-256 is
  `301ddfa1cb0080db4f42eb222d0b7c283efd2c91b845892b6ce0f931c52da004`.

## N239 official result — rejected deployment, unexercised MPC — 2026-07-19

- The preregistered `competitive_n239_identity_body_mpc_batch_008` `LM`
  batch completed unchanged with official counts `3,3,3`, zero finishes, and
  two collision-free attempts. Run 001 hit Gate-4 object `1002` at `41.329 s`,
  threat `2`, impact `6.007645`; runs 002/003 remained collision-free for the
  full `45 s` but never advanced beyond official index `3`. Effective command
  rates were `63.151/63.688/63.710 Hz`; the fixed state schedule delivered
  `2479/2700/2700` steps at `59.982/60/60 Hz`, with zero command-rate
  violations, telemetry dropouts, or drain-limit hits. All three starts used a
  detected disarm-first, single-command-`31000` reset with boot rollbacks
  `2604520->3339`, `33834->3357`, and `20139->3439 ms` and fresh race starts.
  Final manifest/score SHA-256 values are
  `ad4742281a1078385319187fdcdf8a7310161b261a7ab3526ec495393bfad6d0`/
  `98b13a8dcadca042da34363a013ac89ea6b02ea13b53ccb91d26f4214007d246`.
- The learned lateral optimizer was never exercised. Every retained Gate-4
  action had exactly zero roll. A timestamp-preserving audit through N239's own
  association object found accepted-family minimum forward positions
  `15.781706`, `16.000003`, and `15.026607 m`, but zero trusted-family
  triggers, zero optimized plans, and `302/338/337` quarantined associated
  calls. The three-micrometre excess in run 002 is explicitly treated only as
  diagnostic encoder-boundary tolerance, not as a deployed comparison change.
  Audit script/artifact SHA-256 values are
  `88ed13e972a6db26fec3b23a08a623f539b85be0ab17026d84c6c3ab12eeda4c`/
  `ed9b2aaedf5d516e699a0dd2bbafe05182c96d3bd326d1962231d3f0a03ea6a3`.
- Root cause is the admission transition: N239 set `trusted_near_family` only
  if the `<=16 m` condition was already true on initial acquisition or a
  discrete re-acquisition. A continuously associated far family could enter
  the near region without ever enabling MPC. Reject N239 as deployed and do
  not repeat it unchanged. This result does not falsify the trained roll model.
  The next admissible candidate is the minimal semantics correction: allow a
  newly accepted, still-coherent associated sample to latch trust when it
  crosses the same `16 m` boundary, while preserving zero authority for
  rejected poses and preserving all model, pitch, thrust, yaw, dropout,
  commit, prefix, and safety contracts. It must first prove the transition and
  robust centering on all three N239 traces offline; no threshold relaxation or
  new PD gain is authorized.
- Responsive shipping PID `24476` was recovered through the VQ1/R1 UI without
  relaunch after the batch. It is again base/status `193/4`, official index
  `0`, finish `-1`. Recovery SHA-256 is
  `f2a57a14653c3cc0f9b9f735e62ecaf7d724f55fc26d745b5c910ea76b379cbb`.

## N240 offline rejection and N241 roll-identification probe — 2026-07-19

- N240 implemented the minimal N239 admission correction: an accepted,
  continuously associated family could latch trust at the unchanged nominal
  `16 m` boundary. A `1 mm` encoder-round-trip tolerance was explicit because
  one nominal boundary sample reconstructed as `16.000003 m`; rejected poses
  remained inert. N240 was rejected offline and never flown. It unlocked on all
  three N239 traces, but only `2/18` leave-one-model-out lateral rollouts
  centered; maximum terminal error was `15.802635 m`. A scan of every accepted
  near state found zero feasible plans in all three traces; per-run best worst-
  member errors were `7.516056/15.802635/2.368993 m`. Policy/evaluator/result
  SHA-256 values are `8f3b622b...`/`d3ce1637...`/`d1dba169...`; scan script/
  artifact values are `1325009a...`/`83c99ed1...`.
- Adding the three zero-roll N239 trajectories to all six N237/N238 traces
  produced `145` clean transitions and improved separation of ambient drift
  from action. The nine-whole-run fit correctly failed admission: its full
  roll gain was only `1.1210`, fold gains ranged `0.1387-2.2954`, and the
  required roll response was not identified in every fold. The artifact is
  `passed=false` and must not be loaded by a controller. Its SHA-256 is
  `6bc624b805185ca8ad5dfd091a5393378caa078e3554be00bdf250088d82378b`.
- N241 is therefore one bounded causal identification probe, not a lap
  candidate. It inherits N238's persistent accepted association and exact
  Gates-1-through-3 parent. Only while that association is fresh and
  `20-40 m` forward, it applies the balanced two-second roll sequence
  `[-.35,+.35,-.35,+.35,-.20,+.20,-.20,+.20]`, each for `0.25 s`.
  Pitch/thrust/yaw remain the level-brake values. A newly accepted sample below
  `20 m`, a stale/missing association, or completion ends excitation; rejected
  raw poses have zero action authority. Callable SHA-256 is
  `205ab6380a92c4770d42793c6b0450f16294d21ce9b8e5efabe6708fd3f8301a`.
- Actual-Windows focused tests passed `14/14` and the launcher parses. Passive
  shadow 015 passed exactly `600/600` decisions at `60 Hz`, maximum replay
  action error `2.300133e-7`, no-op cadence `63.696 Hz`, and zero setpoints,
  reset, arm, disarm, collision, dropout, or drain-limit hit. Shadow/parity
  SHA-256 values are `aa64b2473a384d2c5790b11d1f559c3541ff09e62e962b6db5016e362fc5a29d`/
  `1873c5e7cf93a39d500d81bf1751535bf7ce2153197a44e52a136836e3f6822c`.
- Preregister exactly one diagnostic tag `n241_gate4_roll_identification_probe_001`
  in `FullLap` lifecycle mode for `15 s`, requested command `80 Hz`, state
  `60 Hz`, and target six gates. Reuse PID `24476` without relaunch; require
  normal disarm, `100 ms`, exactly one learned-target command-`31000` reset,
  boot rollback over `1000 ms`, fresh index `0`/finish `-1`, health/cadence
  checks, and final disarm. The result is useful only if official index `3` is
  reached and multiple balanced probe phases are recorded before the near
  abort. Stop after this single probe and refit before any further flight.

## N241 official diagnostic result — uninformative prefix miss — 2026-07-19

- The single preregistered 15-second diagnostic completed exactly once. It was
  collision-free and transport-clean but passed only Gate 1: official index
  `1`, finish `-1`. The retained trace contains `48/102/0/0` samples at
  official indices `0/1/2/3`, so the Gate-4 controller never activated and
  recorded zero occurrences of every probe roll value. No dynamics row or
  controller conclusion may be inferred from this attempt, and it must not be
  repeated unchanged.
- The start proved normal disarm plus one reset-`31000`: boot rolled
  `1815309->3419 ms`, fresh start `3300 ms`. The run delivered exactly `900`
  state steps at `60 Hz` and commands at `59.312 Hz`, with zero collision,
  command-rate violation, telemetry dropout, or drain-limit hit. Smoke,
  validation, and summary SHA-256 values are
  `7731990e1119a6cd10dfd13681f339ff66454a1d4bb41ecfea5c1c32bdd48741`,
  `42b3b0d4b90396e37954da62bacee4d58cd0dbbc91eeaf6beb848cd1464201cb`,
  and `bf84041e3609b416dc136939f84d2d7700f5a23df66329e96b9cbe130b67f486`.
- PID `24476` was recovered through the UI without relaunch and is again
  base/status `193/4`, official index `0`, finish `-1`; recovery SHA-256 is
  `62bdc8296b522c3ffcd58dc31c998cc15ab0cddabd5dea3923ce50a6cecf7e0a`.
  Before another identification attempt, add a runner-owned bounded stop clock
  that begins only when official index `3` is first observed. This permits a
  longer prefix opportunity while still disarming shortly after the two-second
  probe. Do not extend blind flight duration without that stop contract.

## N242--N248 v3391 telemetry waypoint supersession — 2026-07-26

- Live v3391 evidence resolves the SDK contradiction. A 10-second passive run
  received `ATTITUDE` at about `102 Hz`, `LOCAL_POSITION_NED` at `83.5 Hz`,
  `ODOMETRY` at `65.3 Hz`, `HIGHRES_IMU` at `49.2 Hz`, actuator status at
  `83.5 Hz`, and camera at `30.2 Hz`, with non-null pose, velocity, attitude,
  and quaternion values.
- The command-31000 proof rolled `sim_boot_time_ms 334629 -> 374` in `0.906 s`
  and published six non-null `2.72 x 2.72 m` gates. Ordered NED centers are
  `(-23.298,-0.400,-0.032)`, `(-46.894,-2.500,5.068)`,
  `(-74.594,1.200,13.668)`, `(-111.494,-5.100,24.568)`,
  `(-135.494,-0.800,25.356)`, and `(-159.194,-4.400,25.968)` metres. Artifact
  SHA-256 is `187895d524454ea2809218843670493bfda6eb945c07c7ec19a4311e7e92bc8a`.
- `SET_POSITION_TARGET_LOCAL_NED` velocity control is functional in v3391.
  N247 proved the retained launch/handoff: `0.5 s` of zero body rates at thrust
  `0.27`, then local-NED segment velocity. At requested `1 m/s` it moved to
  `(-1.757,-0.015,0.060) m` in two seconds with maximum cross-track error
  `0.0651 m` and no post-launch collision. Six threat-1 object-1002 messages
  occurred only inside the first-second/first-metre spawn-support envelope and
  are retained separately rather than hidden. N247 SHA-256 is
  `55ecc521194afc26f2700776fa5361d8401b098731d3c4c310a2b21c25149cd6`.
- Active successor N248 is one reset-isolated six-gate waypoint run using the
  published ordered gate centers: `45 s` maximum, `5 m/s` segment speed,
  cross-track gain `1 s^-1`, correction cap `2 m/s`, `60 Hz` commands, `2 Hz`
  heartbeat, the N247 `0.5 s`/`0.27` launch, and immediate abort on any
  collision outside the explicit spawn-support envelope. Success requires
  official index `6`, nonnegative finish time, and no in-course collision.
  Runner SHA-256 is
  `2b778f795e3ebad2c76b20a361476a0ecc0030503cd196cacddb5064563b5166`.

### N248 result and N249 crossing-speed successor

- N248 was rejected after one attempt. It tracked Gate 1's telemetry center to
  `1.1 cm` cross-track error, but at `4.37 m/s` actual velocity and about
  `0.097 rad` pitch it hit gate object `1001`, threat `2`, impact `4.4685`, at
  position `(-22.927,-0.404,-0.030) m` before index zero advanced. Reset,
  cadence, and transport were healthy. Artifact SHA-256 is
  `85a48c9b0adc6b5c518d623ad17f57a09afe4ba97b8d7a025020031e4fe59c25`.
- N249 changes only gate-plane speed: retain `5 m/s` cruise, but smoothly reduce
  to `1.5 m/s` over the final `5 m` before every gate and hold that crossing
  speed until official index advances. All geometry, correction, launch,
  reset, support-contact, and abort contracts remain exact. Run once for at
  most `45 s`; source SHA-256 is
  `3486fe4fd21cc21c896cd0d48dbc2ff719d3fe82e320dedb5edbfccf2a7ed3de`.

- N249 was rejected once: the five-metre schedule lowered commanded speed to
  `1.60 m/s`, but actual speed remained `2.65 m/s` at Gate 1 and still produced
  object-1001 threat-2 impact `2.7369`; cross-track error remained only
  `0.0131 m`. The impact reduction confirms speed/pitch dynamics, while the
  short braking distance is insufficient. Artifact SHA-256 is
  `c34d9b223fcab7c90b35f9b1f281a9a2110aec071c709674aee7b34ddaba3c93`.
- N250 is one constant-`1 m/s`, `30 s` Gate-1 isolation run using the same
  telemetry line, launch, reset, and collision contracts. Do not infer a
  geometry offset unless N250 also hits the frame.

## N250--N252 six-gate PufferLib policy status — 2026-07-26

- N250 was rejected once. Even at constant `1 m/s`, the telemetry-centered
  path contacted Gate-1 object `1001` at threat `1`, impact `0.065064`, near
  NED `(-23.080,-0.397,-0.026) m`; official index remained `0`. This rules out
  crossing speed as the only live blocker. Result SHA-256 is
  `5f8f7b9bf27f1c73d213a5e8a04c7ef3cfeccbd8163bc2eee253ef0d9a15cbc9`.
- Preserve the historical v3385 policy
  `checkpoints/drone_race_full_policy_stage_d_gate3_bc/v6c_latest_obsmatch_r135_dropout_e10.bin`
  (SHA-256 `ea03965f...`) as the official clean three-gate baseline. It is not
  directly loadable into the v3391 successor: it consumes camera/body-state
  observations and emits attitude commands, whereas the successor consumes
  exact local telemetry and emits local-course velocity. Do not describe the
  new policy as inheriting those incompatible tensors.
- N251 added a full-course PufferLib environment using the exact six published
  v3391 gate centers, a local-velocity plant, and a 32-value deployable
  telemetry observation. A random-weight PPO run failed. A six-gate analytic
  teacher completed at `23.375 s`, but its PPO checkpoint scored `0/6` across
  an exact teacher-off `512`-episode evaluation; it is rejected as a learned
  policy and must never be reported as a policy success.
- N252 uses explicit behavior cloning on `512` analytic-teacher full-course
  episodes (`718080` float32 records). The resulting two-layer MinGRU policy,
  with teacher blend and reward both zero, passed exactly `512/512` native
  deterministic episodes: score/gates `6`, success/valid `1.0`, timeout and
  miss `0`, mean completion `23.065336 s`, and mean final-gate radial error
  `0.205588 m`. Promoted native checkpoint:
  `checkpoints/drone_race_vq1_v3391_telemetry/n252_bc_6gate.bin`, SHA-256
  `cec95192a04ebbe0c28f61226fcd76a6851404b6548be12ca6c44165397e9d76`.
  Evaluation evidence is
  `logs/drone_race_vq1_v3391_telemetry/n252_bc_teacher_off_eval.json`, SHA-256
  `9d00c204ebc983974992e92495e9cb233aed1a31de6c3ecac8940abd2afe297a`.
- Native promotion is not an official simulator finish. Next implement and run
  a zero-command v3391 live shadow that reconstructs the same observation from
  `LOCAL_POSITION_NED`, official progress, and the transferred gates. Reconcile
  the N250 Gate-1 collider/geometry evidence before authorizing a full live lap.

## N253--N255 live transfer audit — 2026-07-26

- N253 passed the required command-free live policy shadow on a fresh v3391
  race: `601` finite inference steps in `10.0 s` (`60.1 Hz`), zero MAVLink
  heartbeat/reset/arm/setpoint/disarm commands, no telemetry dropout, malformed
  message, or drain-limit hit. The source checkpoint hash matched N252 exactly;
  the maximum absolute normalized action was `0.697509`. Evidence SHA-256:
  `75dbce6730b821f15c44dd2859ab379be35b88b858a48ab4b1fefae4f399f112`.
- N254 passively paired camera frames with the same stationary origin telemetry.
  The Gate-1 aperture was detected at image center `(326.0,176.5) px`, size
  `20x21 px`, and visual range `23.4146 m`; the published center is `23.3014 m`
  from that vehicle pose. This rejects the proposed `1.36 m` base-to-aperture
  correction and supports the published pose as the aperture geometry. The
  probe sent zero MAVLink messages. Evidence SHA-256:
  `3e471fbf6981b122313a2428925105a3c8a8c65a947651194746b6d9b70c0d37`.
- N250's threat-1 object-1001 message arrived `0.218 m` before the gate center
  while the simulator remained armed/active (`193/4`). It therefore may be the
  bounded gate-volume warning/leading contact that precedes the scoring-plane
  transition, not proof of an aperture-frame strike. N255 is exactly one
  constant-`1 m/s` Gate-1 test. It may continue only past object `1001`, threat
  `<=1`, while within `0.5 m` of the active published gate center; it records
  every such message and stops on official index `1`. Threat `2`, off-center
  messages, other objects, invalid state, or timeout remain immediate failures.
  No automatic retry is authorized. Runner SHA-256:
  `083192e1540946e27a4c7f0c6bfe316cfdf92911b1436b5fadb6ae08c13e062e`.

### N255 result; N256 measured vertical-clearance probe

- N255 was rejected once. It recorded `88` threat-1 object-1001 messages, was
  physically stopped near NED x `-23.093 m` about `0.205 m` before the published
  center, and official index stayed `0`. Collision response then drove NED z
  from about `-0.034 m` to `+0.408 m`; the vehicle eventually moved beyond x
  `-26.6 m` without scoring. This disproves the event-order hypothesis and
  shows the contact was a real gate collision. Evidence SHA-256:
  `0579c0792c54c25b590af5e6887d506d4551486200a982b97557d559d6508a1b`.
- N256 is one bounded clearance probe based on that measured response: retain
  constant `1 m/s`, target Gate 1 at the published x/y and published NED z plus
  `+0.4 m` (down), abort on every collision message with no exception, and
  stop/disarm on official index `1`. The offset is explicit and defaults to
  zero everywhere else. A clean official advance is required; no retry.
  Runner SHA-256:
  `4643df1590ece72075e2e87b5bddf9799c6c64cb4546723e7b1c95bca14fd56e`.

### N256 result; N257 base-origin confirmation

- N256 was rejected once for scoring but passed cleanly below the structure:
  target z was transmitted z plus `+0.4 m` NED-down, Gate-1 collision count was
  zero beyond the retained launch-support contacts, the vehicle crossed to x
  `-28.897 m`, and official index stayed `0`. Evidence SHA-256:
  `e56f5ea238936a8f7eb5b652f957acf5ca8dbb0e64851330b4e8eaf8d9ef213e`.
- N255 contact at transmitted z pushed the vehicle down; N256 passed still
  farther down without touching or scoring. Together these supersede N254's
  initial center interpretation: transmitted z is consistent with the gate
  bottom/base. For height `2.72 m`, aperture center is transmitted z minus
  `1.36 m` in NED.
- N257 confirms this once at constant `1 m/s`, published x/y, target z minus
  `1.36 m`, strict abort on any non-launch collision, and stop/disarm on
  official index `1`. No retry. Runner SHA-256:
  `f37c8431e9924fd8855e48091d55a440693f9d0bc488d4d1edff49810396e20a`.

### N257 passed; N258 policy correction

- N257 passed official Gate 1 cleanly at target z equal to transmitted z minus
  `1.36 m`. Official index advanced `0 -> 1` at `24.469 s`; collision is null,
  final position was `(-23.464,-0.403,-1.397) m` NED, and the run stopped and
  disarmed immediately. Evidence SHA-256:
  `6676991f27ade521cc437d3a8957971f24b7022346bed753b3706a173d5d4c29`.
- The PufferLib profile now uses all six aperture centers by applying the same
  `-1.36 m` correction to transferred NED z (equivalently `+1.36 m` native z).
  Config SHA-256:
  `9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08`.
- N252 remains preserved but is no longer live-eligible because it learned base
  coordinates. N258 is not from scratch: initialize from N252, regenerate `512`
  corrected six-gate teacher trajectories, full-network behavior-clone, then
  require exact deterministic `512/512` teacher-off six-gate success before a
  new command-free live shadow. Do not actuate N252.

### N258 rejection; N259 full-phase correction

- N258's corrected dataset passed `512/512` teacher rollouts (`718033` records,
  mean `23.373470 s`, maximum radial `0.050898 m`). Dataset/report SHA-256:
  `7911972c1db1f8a7c95a2fe64bdd23051112f65b753dd77dec4493c9e990e915` /
  `4cd163477d44867d26705a294d1aeec82e24a9c98b593d971c52a3fa68427be7`.
- Reject N258 checkpoint SHA-256
  `526bf671aabe15d4c98cf8ce15a08e51770428ab5c0f47c592bdbe394a0746f0`:
  exact teacher-off evaluation passed only Gate 1 and timed out. Evaluation
  SHA-256: `d55726c2dd330204e9233b24ea51ac2d44ab81e5bae7d896244d5e2abdfdc336`.
- Root cause is exact: BC omitted `--gate-progress-observation`, so the legacy
  three-phase decoder mapped Gate-2 progress above `1.0` and excluded all
  `105472` Gate-2 records (`612561/718033` samples used). The trainer now fails
  closed when six-gate one-hot columns are present without that flag; focused
  suite passes `21/21`.
- Direct N252 evaluation on corrected geometry also failed Gate 1 at radial
  `0.777577 m` versus radius `0.6`; evidence SHA-256:
  `39d62cb1517cf015f0a99931436208adcb3886b8d093e8a874e7b3d8ebf7bfe6`.
- N259 reuses N252 initialization and the accepted N258 dataset, but explicitly
  enables `--gate-progress-observation`; require all `718033` samples and exact
  `512/512` teacher-off success. This is still transfer, not random restart.

### N259--N261 learned-policy promotion path

- N259 trained from N252 with all `718033` corrected records and the explicit
  six-gate progress contract. Exact deterministic teacher-off evaluation passed
  `512/512`: gates/score `6`, success/valid `1.0`, zero misses/timeouts, mean
  completion `23.088762 s`, final radial `0.206347 m`. Checkpoint/eval SHA-256:
  `f13d4b8d571d70be16731068c88dd105e81fe7bd00a86eae6f13f81031f00fc0` /
  `fc9ece3956c37f95793b4c4beab7d7ecd53a2de079cff60210878febdd2fafc8`.
- N260 passed a fresh command-free Windows v3391 shadow: `601` finite steps in
  `10 s` (`60.1 Hz`), maximum action `0.708850`, no telemetry fault, and zero
  heartbeat/reset/arm/setpoint/disarm messages. SHA-256:
  `2440e182a198b859e4cd29a08d3d0ca98ed02e542cf252c62801d923a31b824f`.
- N261 is one reset-isolated learned-policy Gate-1 run. Use N259, verified base
  to aperture-center gate transform, `60 Hz` local-NED velocity, `2 Hz`
  heartbeat, and the proven `0.5 s` body-rate launch at thrust `0.27`. Retain
  only low-threat object-1002 launch-support contacts inside one second/one
  metre; abort on every other collision/fault. Stop/disarm on official index
  `1`; no retry. Runner SHA-256:
  `5e140d835719ff36d99d8b3fa85a2bd9c930568a894cfa5c2f17c202341b0481`.

### N261 passed; N262 full-course attempt

- N261 passed the learned-policy Gate-1 gate once: official index `0 -> 1` at
  `4.922 s`, null collision/fault, final NED
  `(-23.636,-0.318,-1.398) m`, `296` commands at `60.138 Hz`, and exact N259
  checkpoint identity. The six transferred bases matched configured aperture
  centers plus half-height within `6.13e-8 m`. Evidence SHA-256:
  `e0215049227126f4ec60f907504ff18c19db1a8ef9d0d37747d7637df7dca3d8`.
- N262 is exactly one full-course run with the frozen N259 checkpoint and N261
  runner/lifecycle: `45 s` bound, `60 Hz`, `2 Hz` heartbeat, `0.5 s`/`0.27`
  launch, strict non-launch collision/fault abort. Success requires official
  index `6` and nonnegative finish time; then stop/disarm immediately. No retry.

### N262 two-gate result; N263 governed Gate-3 successor

- N262 is rejected as a lap but retained as clean learned-policy progress. N259
  advanced official indices `0 -> 1` at `4.891 s` and `1 -> 2` at `9.157 s`,
  with null collision and invalid reason, then timed out at index `2`. Artifact
  SHA-256: `4100f48bdd658407788fee11173cc7e7bf13be68f4a93ea9b541d00fadbf3169`.
- Trace reconstruction identifies the failure exactly: the first directed Gate
  3 plane crossing occurred at `14.236 s`, `1.8731 m` from its aperture center;
  the unchanged official index then caused the recurrent policy to reverse and
  oscillate. This is an off-center miss/recovery defect, not a collision,
  transport fault, reset failure, or lack of six-gate observations.
- N263 retains N259 inference and its learned along-track speed, but governs the
  cross-track velocity from public `TRACK_INFO` aperture centers and public
  `LOCAL_POSITION_NED`. It slows learned along speed smoothly to `4 m/s` over
  the final `10 m`, uses gain/cap `1.5 s^-1`/`4 m/s`, and aborts if the active
  gate remains unchanged `1.5 m` past its plane. First execute once with stop
  index `3`; only a clean official Gate-3 pass may promote an identical full-lap
  attempt. Runner SHA-256:
  `27e202604e4e5834b0d726c851cad8cebb19dff826be7ffac50e8a2bee7f4b1e`.

### N263 passed; N264 full-lap promotion

- N263 cleanly advanced official indices through `3` at `5.156`, `9.406`, and
  `14.421 s`. Collision/invalid reason are null, the Gate-3 approach trace ended
  at `0.06233 m` cross-track error, checkpoint/config transforms remained exact,
  and the runner stopped/disarmed at the bounded target. Artifact SHA-256:
  `a0bfdc2a84a697bc36904792bb01ea702aa7506fb51a300a0411bde63b3bc114`.
- N264 is exactly one full-lap run with the identical N263 checkpoint, source,
  reset, launch, governor parameters, rates, and strict abort contract. Change
  only stop index `3 -> 6` and allow at most `40 s`. Acceptance requires official
  index `6`, nonnegative finish time, null collision/fault, then immediate stop
  and disarm. No tuning or automatic retry is authorized.

### N264 accepted official six-gate baseline

- N264 completed all six official v3391 VQ1 gates. Official finish time is
  `29.237468719 s` (`race_finish_time_ns=29237468719`); wall stop observation was
  `29.438 s`. Gate transition observations occurred at
  `5.141/9.391/14.406/20.922/25.172/29.438 s`. Collision and invalid reason are
  null; telemetry dropout, drain-limit hits, and malformed messages are all
  zero; `1767` commands produced `60.024 Hz`; final stop and disarm were sent.
- Retain the complete official artifact at
  `logs/sitl/n264_v3391_n259_governed_full_lap_001.json`, SHA-256
  `3752249b0560a165f870fc5d222d111694f10b8290b2816de644c92ae3ce6d90`.
  This supersedes the three-gate v3385 result as the best current official gate
  count, while that historical checkpoint remains preserved.
- The accepted controller is N259 plus a deployable public-telemetry governor,
  not a random restart and not a hidden-state script. Preserve N264 unchanged as
  the valid baseline. The next objective is lap-time optimization toward N259's
  `23.088762 s` native mean using offline speed/schedule sweeps before any
  separately numbered live successor; never overwrite or weaken N264 evidence.

### N265 offline governor sweep; N266 bounded successor

- N265 reconstructs the immutable N264 trace before changing live control.
  Exact official segment times were `4.976024/4.279658/4.999875/6.502513/`
  `4.284304/4.195095 s`; the `38.989 m` Gate-4 segment is the largest absolute
  time block. Mean scheduled along speeds were `6.81--7.11 m/s`, while mean
  actual along speeds were only `5.05--6.00 m/s`. The fitted live x-axis
  response has steady gain `0.8341` and time constant `0.8369 s`; the public
  cross-track correction cap never activated in N264.
- The deterministic `840`-candidate sweep covers crossing speed, slowdown
  distance, maximum along speed, cross-track gain, and correction cap. Its raw
  baseline replay finishes `0.3572 s` early; that complete residual is recorded
  and gate-wise residual correction makes the baseline exactly N264 rather than
  hiding model error. A per-gate coordinate sweep found no advantage over the
  fastest globally shared settings.
- Preserve N264 and N259. The smallest sub-`27 s` offline rung changes only
  crossing speed `4.0 -> 7.5 m/s`; maximum along speed `8.0 m/s`, slowdown
  `10 m`, cross-track gain `1.5 s^-1`, and correction cap `4.0 m/s` stay exact.
  It predicts `26.792198 s`, `2.445271 s` faster than N264, with maximum modeled
  crossing error `0.30604 m` versus the `0.45 m` offline limit. N265 evidence/
  analyzer SHA-256 values are `8642d60f...`/`aa9fc959...`; the combined focused
  analyzer and existing governor suite passes `7/7`.
- Queue exactly one N266 bounded Gate-3 run using tag
  `n266_v3391_n259_cross7p5_gate3_bounded_001`. Retain the N259 checkpoint,
  current runner SHA-256 `27e202604e...`, reset proof, six-gate transfer check,
  `60 Hz` commands, `2 Hz` heartbeat, `0.5 s`/`0.27` launch, strict collision/
  telemetry/arming aborts, unchanged `1.5 m` plane-miss guard, final zero
  setpoint, and disarm. Run at most `20 s`, stop at official index `3`, and do
  not retry unchanged. Only a clean bounded pass authorizes a separately
  numbered full lap with identical parameters.

### N266 passed; N267 full lap queued

- N266 passed the one authorized bounded run. Official gate times were
  `4.707435/8.494735/13.016106 s`, so the Gate-3 prefix is `1.239449 s` faster
  than N264's official `14.255555 s` prefix. The final Gate-1/2/3 sampled
  cross-track errors were `0.04829/0.17932/0.23168 m`; official index reached
  `3`, finish stayed `-1`, and collision/invalid reason are null.
- Reset rollback, fresh index `0`, finish `-1`, six transferred gates, and the
  aperture transform all passed. `793` commands delivered `60.062 Hz`; telemetry
  dropout, malformed messages, and drain-limit hits are zero; final zero
  setpoint and disarm were sent. Artifact SHA-256 is `f3dc3213...`.
- Queue exactly one N267 official full lap under tag
  `n267_v3391_n259_cross7p5_full_lap_001`. Change only stop index `3 -> 6` and
  duration `20 -> 40 s`; checkpoint, source, parameters, reset, launch, rates,
  plane-miss/collision/health guards, and final stop/disarm remain identical.
  Accept only index `6`, nonnegative finish time, null collision/invalid reason,
  healthy telemetry, and `[50,100) Hz`. Do not retry N267 unchanged. If it is
  faster than N264, retain it as the first optimization candidate but require
  two separately numbered identical valid confirmations before repeatable
  promotion.

### N267 faster valid lap; N268 first confirmation queued

- N267 completed all six gates in official `26.604381561 s`, improving N264 by
  `2.633087158 s` (`9.006%`). Official gate times were
  `4.736413/8.509527/13.041967/19.093603/22.865639/26.604382 s`; collision and
  invalid reason are null, telemetry faults are zero, `1601` commands delivered
  `59.990 Hz`, and final stop/disarm were sent. Artifact SHA-256 is
  `212c5af1...`.
- N267 is a valid optimization candidate but not yet repeatable. Queue N268 once
  under tag `n268_v3391_n259_cross7p5_full_lap_confirm_002`, byte- and parameter-
  identical to N267 except evidence path. Require every N267 acceptance item.
  Do not retry N268 unchanged; a pass authorizes one final identical N269
  confirmation, while a failure reopens reliability analysis.

### N268 passed; N269 final confirmation queued

- N268 passed the identical controller in official `26.591306686 s`, with gate
  times `4.727928/8.495562/13.030109/19.087511/22.870689/26.591307 s`. Official
  index is `6`, collision/invalid reason are null, health counters are zero,
  `1604` commands delivered `60.032 Hz`, and final stop/disarm passed. Artifact
  SHA-256 is `51d36c11...`.
- N267--N268 are `2/2` valid with `0.013075 s` spread. Queue final exact N269
  confirmation once under tag `n269_v3391_n259_cross7p5_full_lap_confirm_003`.
  No source/scalar/lifecycle/acceptance change and no unchanged retry. A pass
  completes the three-lap repeatability gate and triggers final aggregate/hash
  reporting; a failure reopens reliability diagnosis.

### N269 passed; N270 repeatable optimization promoted

- N269 passed all six gates in official `27.412017822 s`. It is slower than
  N267/N268 but still `1.825450897 s` faster than N264. Collision/invalid reason
  are null, health counters are zero, `1680` commands delivered `59.833 Hz`, and
  final stop/disarm passed. Artifact SHA-256 is `9be363aa...`.
- The exact crossing-`7.5` controller is `3/3` valid. Best/median/worst times are
  `26.591306686/26.604381561/27.412017822 s`; spread is `0.820711136 s` and mean
  is `26.869235356 s`. Every lap is faster than N264; best improvement is
  `2.646162033 s` (`9.051%`). Promote N259 plus the one-scalar governor change
  as the repeatable current competitive controller.
- N270 aggregate evidence is
  `logs/sitl/n270_v3391_cross7p5_repeatability_promotion.json`, SHA-256
  `ff466a254fc829dd764484783b20bcf1fdca0c42f96fbba6045f937da2e0a247`.
  Its audit re-hashes N264/N265/N266/N267/N268/N269, the checkpoint, runner, and
  config, then recomputes all acceptance and timing statistics. The first
  sub-`27 s` target is achieved; the optional near-`24 s` stretch remains open.

### N271 three-trace robust sweep; N272 bounded successor — 2026-07-26

- N271 preserves and source-locks the N270 aggregate (`ff466a25...`),
  N267/N268/N269 (`212c5af1...`/`51d36c11...`/`9be363aa...`), N259
  (`f13d4b8d...`), the runner (`27e20260...`), and config (`9eec139d...`).
- Each official trace receives an independent first-order response fit and its
  complete official-minus-raw gate-time residual. The deterministic
  `30`-candidate grid sweeps crossing speed, maximum along speed, and
  cross-track gain while retaining N270's slowdown distance and correction cap.
  Admission requires all models at or below `24.25 s` and within `0.45 m`.
- The minimum-change qualifying candidate is crossing/max/gain `9.0/9.0/2.0`,
  with slowdown/cap `10.0/4.0`. Predicted N267/N268/N269 finishes are
  `23.273484/23.243388/24.080792 s`; mean is `23.532555 s`, and worst crossing
  error is `0.432355 m`. The `9 m/s` command extrapolates beyond N270's observed
  `8 m/s` ceiling, so bounded live proof is mandatory. Analyzer/test/evidence
  SHA-256 values are `687e8230...`/`b8ac71a5...`/`a5d3e096...`; focused tests
  pass `10/10`.
- Preregister exactly one N272 tag
  `n272_v3391_n259_cross9_xgain2_gate3_bounded_001`: unchanged checkpoint,
  runner, config, `20 s`, `60 Hz`, `2 Hz`, official stop index `3`, governor
  crossing/max/slowdown/gain/cap `9/9/10/2/4`, and plane-miss abort `1.5 m`.
  Require one detected command-`31000` reset, six transferred gates, index `3`
  with finish `-1`, null collision/invalid reason, zero telemetry faults, rate
  `[50,100) Hz`, one final zero setpoint, and disarm. Any failure rejects N272
  without unchanged retry; a pass authorizes only a separately numbered lap.

### N272 passed bounded Gate 3; N273 full lap queued — 2026-07-26

- N272 passed official Gates 1--3 at `4.435974121`, `7.801361083`, and
  `11.806352615 s`. The prefix is `1.209753036 s` faster than N266's accepted
  crossing-`7.5` prefix. Final sampled cross-track errors were
  `0.048645/0.195365/0.240791 m`; maximum correction was `1.421769 m/s`, well
  below the retained `4 m/s` cap.
- Official index is `3`, finish is `-1`, and collision/invalid reason are null.
  The one reset rolled boot time `6227191 -> 377 ms`, race start was `3299 ms`,
  all six gates transferred within `6.13e-8 m`, `714` commands delivered
  `59.964727 Hz`, all telemetry fault counters are zero, and final stop/disarm
  were sent. Artifact SHA-256 is
  `8563b545191a03b5354e5d334c3ecc647627a834a2b5c3fe70ff39f84f7637a5`.
- Queue N273 exactly once under `n273_v3391_n259_cross9_xgain2_full_lap_001`.
  Preserve N272's checkpoint, runner, config, crossing/max/slowdown/gain/cap
  `9/9/10/2/4`, `60/2 Hz`, launch, reset, gate-transfer, plane-miss, collision,
  health, stop, and disarm contracts. Change only maximum duration to `40 s`,
  authoritative stop index to `6`, and the unused evidence path. Require index
  `6` plus nonnegative finish and a completely clean contract. Any failure
  rejects N273 without unchanged retry; a valid faster lap requires separate
  repeatability confirmation before promotion.

### N273 rejected at Gate 4; N274 acknowledgment correction — 2026-07-26

- N273 passed Gates 1--3 at `4.460481643/7.811909675/11.828386306 s`, then
  exited at Gate 4/index `3` with `gate_governor_plane_miss`. Collision is null,
  telemetry faults are zero, `1046` commands delivered `59.983943 Hz`, and
  final stop/disarm passed. Artifact SHA-256 is
  `6852268b9804d0744658e13640e5fc80ca7d38d8c7e884173158d3d5caa579e1`.
- N274 reconstructs Gate 4's configured NED-x aperture plane crossing at
  `17.242473 s`, position `[-111.493744,-5.220544,23.446622] m`, only
  `0.267310 m` from center and at `7.314596 m/s` along-track. The official
  status stream measured `5.103796 Hz`/`195.933 ms`; the existing `1.5 m` guard
  spanned only `205.069 ms`. Latest status boot `20453 ms` was `21.473 ms`
  before the estimated crossing boot `20474.473 ms`, so N273 observed no
  post-crossing authoritative sample before aborting.
- N274 retains the exact `1.5 m` miss distance. When explicitly enabled, the
  new guard defers only until a post-crossing official sample or a hard
  `350 ms` deadline; a same-gate acknowledgment or deadline expiry aborts.
  Default grace zero preserves legacy behavior. Corrected root/Windows runner
  SHA-256 is `643cdec893c511d59a8a5a9b54bde2bbab1d543a46b7d977ad92867a01e09505`.
  Analyzer/test/evidence hashes are `bfc1055e...`/`4924ef4c...`/`08a16334...`;
  focused tests pass `16/16`, Windows compile/help checks pass, and the two
  deployment runner files are byte-identical.
- Preregister N275 exactly once under
  `n275_v3391_n259_cross9_xgain2_ack350_gate4_bounded_001`: N259, corrected
  runner, unchanged config, `25 s`, `60/2 Hz`, crossing/max/slowdown/gain/cap
  `9/9/10/2/4`, status-ack grace `0.35 s`, and authoritative stop index `4`.
  Preserve every reset/gate-transfer/collision/health/rate/final-stop/disarm
  requirement. Any failure rejects N275 without unchanged retry; a pass only
  authorizes a separately numbered full lap.

### N275 passed bounded Gate 4; N276 full lap queued — 2026-07-26

- N275 ran once under
  `n275_v3391_n259_cross9_xgain2_ack350_gate4_bounded_001` and passed official
  index `4`. Gate times are
  `4.444323539/7.826340675/11.826963424/17.182216644 s`; finish remains `-1`,
  collision and invalid reason are null, and the effective wire rate is
  `60.049230 Hz` across `1049` control commands.
- Reset rolled boot time `184119 -> 408 ms`, race start is `3311 ms`, all six
  gates transferred within `6.13e-8 m`, and telemetry has zero dropout,
  malformed-message, and drain-limit faults. Final zero setpoint and disarm
  passed. The acknowledgment path was exercised with `4` deferrals and maximum
  observed race-status lag `245 ms`; no plane-miss abort diagnostic occurred.
  Artifact SHA-256 is
  `89fab65cee876300ea0a86614786bf4a540f11c5970d492050799592f44edde1`.
- Queue exactly one N276 full-lap attempt under
  `n276_v3391_n259_cross9_xgain2_ack350_full_lap_001`. Preserve N259, corrected
  runner SHA-256 `643cdec8...`, config, reset, transferred gates, and governor
  values `9/9/10/2/4`, miss distance `1.5 m`, and ack grace `0.35 s`; use
  `40 s`, command/heartbeat `60/2 Hz`, and stop index `6`. Require an official
  index `6` plus nonnegative finish, null collision/invalid reason, zero health
  faults, compliant command rate, final stop, and disarm. Any failure rejects
  N276 without unchanged retry; a valid result is not promoted until separate
  confirmation laps pass the same contract.

### N276 valid 23.848-second lap; N277 confirmation queued — 2026-07-26

- N276 ran exactly once under
  `n276_v3391_n259_cross9_xgain2_ack350_full_lap_001` and passed all six
  official gates in `23.848192214 s`. Gate times are
  `4.422417163/7.812987327/11.818965911/17.180858612/20.544542312/
  23.848192214 s`; the result is `2.743114472 s` faster than N270's best N268.
- Official index is `6` with finish `23848192214 ns`; collision/invalid reason
  are null. The run sent `1437` commands at `59.992485 Hz`, had zero telemetry
  dropout/malformed/drain-limit faults, reset once, transferred all six gates
  within `6.13e-8 m`, and sent the final zero setpoint and disarm. The ack path
  recorded one deferral and maximum status lag `248 ms`. Artifact SHA-256 is
  `6618847fcd8cbe959be045f89a43b623314cff22f9564984ffc614e626deef8a`.
- Queue exactly one confirmation under
  `n277_v3391_n259_cross9_xgain2_ack350_full_lap_confirm_002`. Preserve the
  N276 checkpoint, corrected runner/config, governor values `9/9/10/2/4`, miss
  distance `1.5 m`, grace `0.35 s`, `40 s`, `60/2 Hz`, and stop index `6`
  exactly. Require official index `6`, finish `<=24.5 s`, null collision/invalid
  reason, zero telemetry faults, compliant wire rate, exact artifacts, final
  stop, and disarm. Failure rejects N277 without unchanged retry; success
  authorizes only one final separately numbered confirmation.

### N277 rejected; N278 strict-timestamp correction; N279 queued — 2026-07-26

- N277 ran once under
  `n277_v3391_n259_cross9_xgain2_ack350_full_lap_confirm_002`, passed Gates
  1--3 at `4.432579994/7.811755180/11.823871612 s`, then stopped at Gate 4 with
  `gate_governor_plane_miss`. It is rejected without unchanged retry. Collision
  is null, `1045` commands delivered `59.981632 Hz`, telemetry faults are zero,
  and final stop/disarm passed. Artifact SHA-256 is
  `9e5e35f5142036c3aef1b58f208c36084650f75d11538e30e8e8d5857d52e690`.
- The abort diagnostic proves a runner-resolution defect: crossing and latest
  race-status boot times were both `20457 ms`, but `>=` treated equality as a
  post-crossing sample. At abort, current boot was `20668 ms`, only `211 ms`
  after crossing and below the `350 ms` deadline. Interpolation places the
  vehicle center inside the transferred Gate-4 aperture at
  `[-111.493744,-5.220194,23.446662] m`.
- N278 changes only the acknowledgment comparison from `>=` to `>`; the
  `1.5 m` distance and `350 ms` timeout remain exact. Root/Windows runner
  SHA-256 is `de4995a19c5f3c82f0de540b4b07ce810ffae49ad9fd4308befbcea3cfcb34ea`.
  Analyzer/test/report hashes are `c9b17c64...`/`92f7ced9...`/`232c9d19...`;
  focused Linux tests pass `19/19`, Windows compile/direct probes pass, and
  runners are byte-identical.
- Queue exactly one bounded N279 under
  `n279_v3391_n259_cross9_xgain2_ack350_strict_gate4_bounded_001`: preserve
  N259/config and governor `9/9/10/2/4`, miss distance `1.5 m`, grace `0.35 s`,
  use corrected runner, `25 s`, `60/2 Hz`, and stop index `4`. Require official
  index `4`, finish `-1`, null collision/invalid reason, one reset, six gates,
  zero telemetry faults, compliant rate, exact hashes, final stop, and disarm.
  Failure rejects N279 without unchanged retry; success authorizes only a new
  full-lap number.

### N279 passed bounded Gate 4; N280 full lap queued — 2026-07-26

- N279 ran exactly once under
  `n279_v3391_n259_cross9_xgain2_ack350_strict_gate4_bounded_001` and passed
  official index `4`. Gate times are
  `4.423978328/7.786879062/11.807009696/17.171604156 s`; finish is `-1`,
  collision/invalid reason are null, and `1045` commands delivered
  `60.036769 Hz`.
- Its one reset (`322972 -> 357 ms`, race start `3300 ms`), six-gate transfer
  (`6.13e-8 m` maximum error), zero telemetry faults, final zero setpoint, and
  disarm pass. The corrected guard had zero deferrals and no abort diagnostic.
  Artifact SHA-256 is
  `e068a68501ee5fc9a73f9a86b43a222b8c3f3b14e30ae2b910976bb62256ea47`.
- Queue exactly one N280 full lap under
  `n280_v3391_n259_cross9_xgain2_ack350_strict_full_lap_001`: keep N279's
  checkpoint, runner/config, governor values `9/9/10/2/4`, miss distance
  `1.5 m`, grace `0.35 s`, and all lifecycle/health guards exact; use `40 s`,
  `60/2 Hz`, and stop index `6`. Require official index `6`, finish `<=24.5 s`,
  null collision/invalid reason, zero telemetry faults, compliant rate, exact
  hashes, final stop, and disarm. Failure rejects N280 without unchanged retry;
  success authorizes separately tagged confirmation laps.

### N280 valid 23.832-second lap; N281 confirmation queued — 2026-07-26

- N280 ran once under
  `n280_v3391_n259_cross9_xgain2_ack350_strict_full_lap_001` and passed all six
  gates in `23.832376480 s`. Gate times are
  `4.415704250/7.803259372/11.801978111/17.162479400/20.518789291/
  23.832376480 s`; this is `2.758930206 s` faster than N270's best N268.
- Official index/finish are `6/23832376480 ns`; collision/invalid reason are
  null. Its `1441` commands delivered `60.041667 Hz`; telemetry fault counters
  are zero; reset, six-gate transfer, final zero setpoint, and disarm pass. The
  guard recorded one deferral and `244 ms` maximum status lag. Artifact SHA:
  `5b926383063ffe8cf2f881a634925853a73c5fbb81cce4c93418e6c95161abec`.
- Queue exactly one N281 confirmation under
  `n281_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_002`. Preserve
  N280's controller and complete execution contract exactly. Require index `6`,
  finish `<=24.5 s`, null collision/invalid reason, zero telemetry faults,
  compliant rate, exact hashes, final stop, and disarm. Failure rejects N281
  without unchanged retry; a pass authorizes one last separately numbered
  confirmation, targeting corrected-runner repeatability `3/3`.

### N281 passed first confirmation; N282 final confirmation queued — 2026-07-26

- N281 ran once under
  `n281_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_002` and passed
  all six gates in `23.879062652 s`. Gate times are
  `4.445917606/7.809613704/11.827727317/17.197458267/20.556997299/
  23.879062652 s`.
- Official validity, null collision/invalid reason, `1438` commands at
  `60.034234 Hz`, zero telemetry faults, reset, six-gate transfer, final zero
  setpoint, and disarm pass. Ack diagnostics show two deferrals and `244 ms`
  maximum lag. Artifact SHA-256 is
  `0fd5a7d4b78ebdcc11562443c11a70b6593769bd15ecc5d9d9b4bc9a29225250`.
- Queue exactly one final confirmation under
  `n282_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_003`. Preserve
  N280/N281 exactly; require official index `6`, finish `<=24.5 s`, null
  collision/invalid reason, zero telemetry faults, compliant rate, exact hashes,
  final stop, and disarm. Failure rejects N282 without unchanged retry; a pass
  completes the corrected-runner `3/3` promotion gate.

### N282 passed; N283 near-24 promotion — 2026-07-26

- N282 ran once under
  `n282_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_003` and passed
  all six gates in `23.867099761 s`. Gate times are
  `4.435152053/7.801836967/11.832441329/17.202220916/20.570261001/
  23.867099761 s`.
- Official index/finish are `6/23867099761 ns`; collision/invalid reason are
  null. `1438` commands delivered `59.994159 Hz`; telemetry faults are zero;
  reset, six-gate transfer, final zero setpoint, and disarm pass. Ack diagnostics
  show two deferrals and `245 ms` maximum lag. Artifact SHA-256 is
  `27fbf3ba00c62553468f95d515a1144b0aeb0a31555cc5177fcce6c7e9e3fb36`.
- N283 source-locks N280--N282 and promotes them `3/3`: best/mean/worst
  `23.832376480/23.859512964/23.879062652 s`, range `0.046686172 s`; every lap
  is faster than N270's `26.591306686 s` best. Best saving is `2.758930206 s`;
  mean saving versus N270 is `3.009722392 s`.
- Frozen checkpoint/runner/config SHA-256 values are `f13d4b8d...`/
  `de4995a1...`/`9eec139d...`. N283 analyzer/test/promotion hashes are
  `a776ccf9...`/`cdcd7117...`/`898c6671...`; the artifact regenerates exactly,
  focused tests pass `20/20`, `git diff --check` passes, and operational runner
  byte parity passes. N283 supersedes N270. No further live run is authorized
  without new offline evidence and a separately preregistered bounded path.


### VQ2 N402--N477 calibrated Puffer Gate-2 promotion — 2026-07-27

- The deployment contract remains Puffer-only: N294 emits the complete
  four-action vector at official index 0 and a second recurrent Puffer
  checkpoint emits the complete vector at index 1. Deterministic processing may
  transform public camera pose/rate, public attitude, previous action, and time
  into legal state features; it never emits, blends, clips selectively, or
  overrides a checkpoint action. Privileged native state and the classical
  teacher remain offline-only.
- N402 corrects the Gate-2 native geometry to
  `[14.673019876, 8.466995812, 2.404214418] m`. N403--N469 reject the old
  N380 policy, recurrent BC/DAgger/PPO, time-switched Puffer schedules, and
  partial forward-rate prediction. The key observability defect is that the
  public camera rate becomes stale after the gate leaves view; the legal ABI
  still supplies attitude and previous thrust.
- N470 implements a calibrated stationary-gate kinematic predictor that updates
  only the existing six public gate pose/rate observation slots. N460 plus this
  predictor misses at `3.176659 m`, so it is not retained. N471's offline
  teacher reference passes at `0.550163 m` and supplies labels only; no teacher
  action enters an admission rollout or deployment.
- N472 decoder ridge `1e-3` is the first reliable whole-Puffer result:
  `128/128` exact and `101/128` at perturbation scale `0.05`, zero crashes.
  N473 then collects `319/320` successful disjoint teacher trajectories
  (`213,024` records, SHA-256 `b3275022...`).
- N474 fits those legal predicted features into the Puffer decoder. Retain
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n474_full_kinematic_perturbed_decoder/ridge_1e-1.bin`,
  SHA-256
  `c39d3c653e33a2dbeda3eb90869fd65ae35a6ca1825380a876a8111b0b8341e6`.
  It passes `128/128` exact and two disjoint `128/128` scale-`0.05`
  screens, with nominal radial error `0.2126 m`; it also passes `123/128`
  at scale `0.10`. Every screen has teacher blend zero and zero crashes.
- Deployment preprocessing source-locks the legal N399 transition: live camera
  world-vector scales are `1.73424389/1.72005255/2.2471045`, initial
  stationary-gate rate is
  `[-4.195446885,-0.030547706,-0.352165921] m/s`, predictor step is `1/60 s`,
  and the live `14 s` elapsed fraction is normalized to the training
  `20 s` denominator. These values reconstruct
  `[14.673021052,8.466994748,2.404214235] m` from the public N399 camera
  sample. They are perception/state calibration only; the checkpoint still
  owns all four actions.
- N475 composite replay passes on Linux and actual Windows Python. All `195`
  N295 prefix actions reproduce within `4.805625e-7`; the first recorded N399
  Gate-2 observation selects the calibrated N474 action exactly. Linux/Windows
  replay SHA-256 values are `131ba00e...`/`19cede84...`; callable SHA-256 is
  `b9705f20354106de809f524acb568ada952fa090c4b497a2d46a437c18dabd68`.
- The UI was recovered without relaunch, the third
  `AI-GP VIRTUAL QUALIFIER R2 - TRAINING` row was visibly selected, and a
  receive-only check proved index `0`, finish `-1`, base mode `193`, and
  system status `4`. VQ2 Submission was not selected.
- Zero-command N476
  `vq2_n476_gate2_composite_shadow_003` passes: `523/523` inferences at
  `52.3 Hz`, maximum action error `3.539654e-7`, all nine manifest hashes
  exact, `1,805` telemetry messages, and `114/114` frames/detections.
  No-op cadence is `64.0 Hz` with zero rate violations and zero MAVLink
  setpoints. Reset, arm, control-setpoint, and disarm counts are all zero.
  Shadow/parity SHA-256 values are `30e5765e...`/`54b92afb...`.
- Preregister exactly one bounded Training attempt under
  `vq2_n477_gate2_bounded_002`. Use wrapper
  `scripts/run_windows_vq2_gate2_policy_n477_bounded.ps1`, SHA-256
  `9c9ca216fe65607984859fd40b483d17325571985993bfb1fde0b64d382f53f6`.
  Reuse the responsive Training process, issue one detected MAVLink-`31000`
  reset, request `80 Hz`, cap at `14 s`, and require target/min/stop official
  index `2`, finish `-1`, no collision/invalid/stream/rate failure, one exit
  disarm, and passive disarm proof. Any failure rejects N477 without unchanged
 retry. VQ2 Submission remains forbidden.


### VQ2 N477 rejection; N480--N483 fixed-state successor — 2026-07-27

- N477 ran exactly once and is rejected without unchanged retry. It passed
  official Gate 1 at `3.211249113 s`, then collided with ID `1002`, impact
  `6.237362`, at index `1` around `5.36 s`; finish stayed `-1`.
  Transport was clean: `348` commands at `63.999 Hz`, zero rate violation,
  detected reset, and one exit disarm. Attempt SHA-256 is `a04c6947...`.
  Command-free passive proof records base mode `65`, system status `3`, and
  zero outgoing packets/commands; SHA-256 is `77aa27d0...`.
- Exact Puffer replay is within `6.256e-7`, so checkpoint execution and wire
  delivery are not the failure. At official index-1 activation the detector
  retained the old Gate-1 pose for `0.984 s`; the first new-target jump was
  `7.257239 m` after `54` inference ticks. N477 therefore initialized the
  legal predictor at `[24.339361,14.275489,5.237102] m`, not the previously
  reconstructed Gate-2 vector. Its first Gate-2 action saturated roll and
  increased thrust to `[0.163092,-1.0,0.782781,0]`.
- N478 proves whole-Puffer N294 delays of `0.5/0.75/1.0/1.25 s` all miss.
  N479 delays the training-only teacher until `1.0 s`, but even the teacher
  cannot recover that N294 trajectory; close the delayed handoff branch.
- N480 changes only predictor initialization: use the legally reconstructed
  relative Gate-2 world vector
  `[14.673019876,8.466995812,2.404214418] m` instead of the first potentially
  stale camera association. N294 still owns all index-0 actions and unchanged
  N474 still owns all index-1 actions. This is public-evidence state
  initialization, not an action controller or privileged simulator coordinate.
  N480 passes `128/128` exact, `128/128` disjoint scale-`0.05`, and
  `122/128` scale-`0.10`, with zero crashes. On N477's exact first index-1
  observation it emits `[0.137554,-0.932296,0.676103,0]`.
- N481 replay passes identically on Linux and Windows. Prefix error remains
  `4.805625e-7`, Gate-2 selection error is exactly zero, and the transition is
  N477's source-hashed stale observation. Linux/Windows report SHA-256 values
  are `9dc4997b...`/`22904419...`; current callable SHA-256 is
  `1df2a68feac5e4b3f0c3ed3a25992114d6e6c95e28f4acb33a0cb453257b4ba2`.
- The responsive process was recovered without relaunch and only the visible
  third VQ2 Training row was selected. Zero-command N482
  `vq2_n482_fixed_state_shadow_004` passes: `531/531` Puffer inferences at
  `53.1 Hz`, maximum replay error `3.853779e-7`, all nine manifest hashes
  exact, `1,792` telemetry messages, `115/115` frames/detections, and
  `64.002 Hz` no-op cadence. Reset/arm/setpoint/disarm counts and cadence
  MAVLink setpoints are zero. Shadow/parity SHA-256 values are
  `04320a87...`/`a365ac34...`.
- Preregister exactly one bounded VQ2 Training attempt under
  `vq2_n483_fixed_state_gate2_bounded_003`. Wrapper
  `scripts/run_windows_vq2_gate2_policy_n483_bounded.ps1` has SHA-256
  `1d0217bcc60a435e9ce98120155c5b8a7c8651fb24b6217fc9bf90eacb238fc3`.
  Reuse the active Training process, send one detected command-`31000` reset,
  request `80 Hz`, cap at `14 s`, and require index `2`, finish `-1`,
  clean safety/streams/rate, one exit disarm, and passive disarm proof. Any
  failure rejects N483 without unchanged retry. Submission remains forbidden.

### VQ2 N483 rejection and corrected-bearing Puffer successor — N483--N520, 2026-07-27

- The user requires PufferLib for deployed VQ2 control because the classical
  controller lineage did not transfer. This is now a hard constraint: every
  flown four-action vector must be the complete output of a recurrent Puffer
  checkpoint, followed only by the fixed documented action decoder. Classical
  control is permitted only for offline labels, rewards, and diagnosis; it may
  not emit, blend, clip selectively, schedule, or override a deployed action.
- N483 ran exactly once and is rejected without unchanged retry. It passed
  official Gate 1 at `3.251771926 s`, then collided with ID `1002` at index
  `1` around `5.735 s`; impact was `0.145923`, finish remained `-1`, and the
  run was invalid. Transport itself passed: `363` commands at `63.822 Hz`,
  zero rate violations, a detected reset with all `60` calibration samples,
  and exactly one exit disarm. Attempt/passive SHA-256 values are
  `8f167767...`/`9377a16f...`; the passive probe sent zero packets and proved
  base mode `65`, system status `3`.
- N519 source-locks the failure using only recorded public observations,
  camera-derived poses, official status, and Puffer actions. All `333` actions
  (`189` index-0 and `144` index-1) replay within `6.705523e-7`. At sample
  `241`/`4.266 s`, the raw aperture changes from the stale vector
  `[13.247626,7.208333,-2.778962] m`, yaw error `0.49832`, to the new nearly
  centered vector `[30.025565,1.078125,0.259298] m`, yaw error `0.035891`.
  Preserving only the legal predictor's forward range and adopting that public
  bearing reconstructs corrected initial Gate-2 world vector
  `[14.518822083,0.418616153,0.472708623] m` from rounded trace fields. The
  registered training vector `[14.51880584,0.41894794,0.47249277] m` agrees
  within `0.000332 m`. N519 report SHA-256 is `ab6fcb57...`; its focused suite
  plus evaluator/runtime coverage passes `42/42`.
- N485--N493 close the simple altitude-floor, lateral-plant-only, bearing-
  reseed-only, and teacher-gain-transfer explanations. The old N474 Puffer
  still crashes on the corrected geometry, while the offline teacher passes;
  this is a checkpoint training problem rather than authorization for a
  classical runtime action.
- N495 collects `320/320` successful corrected-geometry teacher episodes at
  perturbation scale `0.05` (`216,588` records, dataset SHA-256
  `89e0d5c2...`). N496's first decoder family yields one exact `128/128`
  child, but N498 rejects it at only `58/128` scale-`0.05` and `64/128`
  scale-`0.10`. N499 shows start, gate, and plant perturbations each expose the
  same edge-crossing weakness.
- Lower decoder regularization improves exact radial error to about `0.225 m`
  but reaches only `118/128` at scale `0.05`. N505 therefore collects a wider
  `512/512` scale-`0.10` teacher corpus (`346,311` records, dataset SHA-256
  `1c27b0c9...`). Retain the N506 stress-only decoder:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n506_corrected_geometry_stress_decoder/stress_only_ridge_1e-7.bin`,
  SHA-256
  `55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f`.
  Teacher-free screens pass `128/128` exact, `125/128` disjoint scale-`0.05`,
  and `104/128` scale-`0.10`, all with zero crash/timeout. Exact radial error
  is `0.472141 m`; disjoint radial aggregate is `0.439108 m`.
- N509--N518 recurrent-layer, full-network, DAgger-label, assisted-trajectory,
  and gate-feature encoder refinements all fail to beat N506. The strongest is
  only `124/128`; the direct DAgger children collapse as low as `6/128` and
  introduce three crashes. Freeze these branches and do not promote them.
- N520 verifies the actual N506 composite on Linux and Windows controller
  Python. N294 still reproduces all `195` prefix actions within
  `4.805625e-7`; the official N483 index-1 observation selects N506's whole
  action `[0.313981891,-0.478270173,0.146595865,0.000246137]` exactly.
  Callable SHA-256 is `f54c885e...`; Linux/Windows replay SHA-256 values are
  `19b96abd...`/`166b0685...`.
- No post-N483 FlightSim command has been sent. N506 is the retained offline
  candidate, not yet authorized for actuation. Next prepare and pass a
  source-hashed zero-command Windows shadow in recovered VQ2 Training, with
  every lifecycle/control count exactly zero. Only after that evidence may a
  separate bounded Gate-2 attempt be preregistered. VQ2 Submission remains
  forbidden.

### VQ2 N521 shadow and N522 Gate-2 preregistration — 2026-07-27

- The responsive v3391 process was recovered without relaunch. Only the
  visible third `AI-GP VIRTUAL QUALIFIER R2 - TRAINING` row was selected;
  receive-only status then proved index `0`, finish `-1`, base mode `193`, and
  system status `4`. Readiness/screenshot SHA-256 values are
  `60e02e79...`/`9cbdd6fe...`. Submission was not selected.
- Zero-command N521 `vq2_n521_corrected_geometry_shadow_005` passes. It replays
  `529/529` complete Puffer actions at `52.821 Hz`, maximum action error
  `4.392170e-7`, with all nine deployment hashes exact, `1,805` telemetry
  messages, and `114/114` frames/detections. No-op cadence is `64.004 Hz` with
  zero violations and zero MAVLink setpoints; reset, arm, control-setpoint,
  and disarm counts are zero. Report/parity SHA-256 values are
  `92053ff9...`/`88f34aa3...`; shadow wrapper SHA-256 is `f6b5576e...`.
- Preregister exactly one bounded VQ2 Training attempt under
  `vq2_n522_corrected_geometry_gate2_bounded_004`. Wrapper
  `scripts/run_windows_vq2_gate2_policy_n522_bounded.ps1` has SHA-256
  `274abd7fdd8010c422e27afcec3a3ef795c5758f501c644af6c416104d1d3cb8`.
  It pins the N294/N506 whole-Puffer composite, N519 public corrected vector,
  N521 shadow evidence, callable, runner, and acceptance contract. Reuse active
  Training, send one detected MAVLink-`31000` reset, request `80 Hz`, cap at
  `14 s`, and require target/min/stop official index `2`, finish `-1`, no
  collision/invalid/stream/rate failure, one exit disarm, and passive disarm
  proof. Any failure rejects N522 without unchanged retry. VQ2 Submission is
  forbidden.

### User redirection: emergent visual Puffer only — 2026-07-27

- N522 ran exactly once and is rejected without unchanged retry. It passed
  official Gate 1, remained at index `1`, and collided with ID `1002` before
  Gate 2. The report/passive SHA-256 values are `bbb5dd70...`/`233e226d...`.
  Transport was valid (`457` commands at `64.0 Hz`, zero violations), reset
  was detected, and one exit disarm plus command-free mode/status `65/3` were
  proved. The failure is control transfer, not transport.
- The user explicitly rejected another calibrated-detector/predictor successor
  and requires emergent behavior. Close the N480--N522 explicit Gate-2 vector,
  camera-pose detector, kinematic predictor, classical teacher, and phase-
  specific action lineage. Preserve it only as historical evidence.
- The successor must be a recurrent end-to-end visual Puffer policy. Its actor
  sees pixel-level official-camera representations, body rates, reliable
  actuator/motor feedback, previous action, and recurrent history only. It may
  not receive detector geometry, reconstructed pose/velocity/attitude, course
  vectors, kinematic state, or official gate index. Official status is judge/
  reward/stop evidence only. Privileged state is training-only for reward,
  critic, latent reconstruction, and world-model targets.
- Every deployed vector must be a complete neural output with no classical
  blend, channel override, or runtime planner. Prefer direct actuator output
  only after live legality/semantics are proved; otherwise output the complete
  supported body-rate/thrust vector. The detailed actor-legality contract is
  `docs/vq2_emergent_puffer_goal_prompt.md`; the authoritative paper-guided
  execution prompt is `docs/vq2_paper_guided_puffer_goal_prompt.md`.
- Freeze FlightSim actuation. First build and train the pixel-input Puffer
  environment/policy, prove held-out visual and dynamics generalization,
  export parity, benchmark actual-Windows inference above `50 Hz`, and pass a
  zero-control VQ2 Training shadow. Submission remains forbidden.

### N523 Informed-Dreamer visual foundation — 2026-07-27

- The successor mechanism is now specifically Informed Dreamer, not an
  optional model-free PPO fallback. PufferLib supplies the native environment,
  rollout/replay path, and deployed recurrent policy integration.
- A separate `drone_race_vision` binding preserves the historical 32-float
  ABI. Its transport observation is `4,152` floats: a `4,118`-value legal actor
  slice (`64x64` continuous gate-edge mask, body rates, four motor/actuator
  feedback values, three prior actions, and camera timing) plus `34`
  training-only privileged decoder targets. The actor receives no official
  phase, geometry, pose, or decoded state.
- The renderer adapts Geles et al.'s efficient continuous inner-edge masks,
  five-way edge segmentation, three-action history, and corrupted edge
  randomization. It excludes their explicit perception reward and feed-forward
  PPO actor. It adapts SkyDreamer's discrete RSSM, GRU, privileged decoder,
  reward/continue heads, latent imagination, and deterministic action.
- Native historical and visual C regressions pass; the PyTorch contract/RSSM
  suite passes `8/8`. A CPU smoke completed `640` agent-steps and `3` complete
  world/imagination updates; RTX 3070 completed `576` agent-steps and `2`
  updates. Every reported loss was finite. This proves integration only, not a
  trained or deployable policy.
- Authoritative implementation record:
  `docs/vq2_informed_dreamer_implementation_2026-07-27.md`. Continue offline
  training and disjoint screening. FlightSim actuation and Submission remain
  forbidden.

### Paper-guided hardware-bounded VQ2 execution — 2026-07-27

- The user requires a recurring SkyDreamer paper audit tied directly to the
  VQ2 solve, not a one-time literature summary. The authoritative execution
  prompt is `docs/vq2_paper_guided_puffer_goal_prompt.md`; the older emergent
  prompt remains its actor-legality contract.
- Keep the solution minimal: one learned `64x64` official-frame mask, one
  compact discrete RSSM/GRU, and one actor emitting one complete four-channel
  CTBR vector. Mapped flight plans, gate index, decoded-state runtime feedback,
  planners, classical actions, and checkpoint switching remain forbidden.
- Apply the paper's functional training lessons: distinguish burn-in `16`,
  world-model sequence `64 -> 128 -> 256`, and imagination `16`; raise measured
  replay density; use gate-local/full-start curriculum; and test persistent
  visual corruption, parameter inference, and long-horizon drift.
- Adapt to the measured RTX 3070 8 GiB, four-core CPU, and 7.7 GiB RAM using
  bounded memory-mapped replay, mixed precision, microbatching, gradient
  accumulation, and resumable `0.5M -> 2M -> 8M -> 17M` transition stages.
  Store large artifacts on WSL because Windows `C:` has only about 6.9 GiB
  free. Do not assume the paper's 40 GB A100 resources.
- N528's disjoint and visual-counterfactual evaluation was the first offline
  step and is closed by the N528-N531 correction section below. FlightSim
  actuation and Submission remain frozen.

### Paper-guided VQ2 offline correction — N528-N531, 2026-07-27

- N528 is rejected without unchanged continuation: disjoint deterministic
  Gate 1 is `0/64` with `64/64` low crashes. Counterfactual initial-mask
  blanking changes `64/64` actions (mean absolute delta `0.0084648455`), so
  the failure is policy/control learning rather than a dead visual pathway.
  Evaluation/counterfactual SHA-256 values are `fba84fa...`/`ca0cd299...`.
- N529 separates gradient-free replay context `16`, optimized world sequence
  `64`, and imagination `16`. It performs exactly two finite world/critic and
  zero actor updates at batch one; peak allocated/reserved CUDA memory is
  `65,055,232/90,177,536` bytes. It used the superseded motor schema and is
  temporal-contract evidence only.
- The active schemas are now `vq2_visual_ctbr_v2` and
  `normalized_attitude_ctbr_v1`. One Puffer actor emits the complete four-value
  pitch/roll/collective/yaw vector; the fixed gyro-closed decoder targets the
  proven `SET_ATTITUDE_TARGET` path. Zero normalized action is hover-centered.
- N530 proves CTBR training plus resumable WSL-backed NumPy memmaps under the
  `16/64/16` contract. Its 96-step replay is 25,927,680 bytes; model/metrics/
  replay-metadata SHA-256 values are `6e877f67...`/`002713b3...`/`6acce705...`.
- N531 proves the paper-adapted 70% uniformly randomized gate-local / 30% full-
  start curriculum through the rebuilt CUDA binding. It records 72 completed
  episodes, local fraction `0.7361111045`, zero optimizer steps, and a bounded
  21,606,400-byte replay. Metrics SHA is `088b1382...`. Gate phase/reset
  provenance remain outside the actor slice; admission uses full starts only.
- Focused Python tests pass `13/13` and visual native regressions pass `8/8`.
  Next add AMP, microbatch/gradient accumulation, and interrupted-resume proof;
  benchmark sequence 64 then 128 below 7.2 GiB allocated VRAM before one
  resumable `0.5M` discovery run. No FlightSim command is authorized and
  Submission remains forbidden.

### VQ2 SkyDreamer/Informed-Dreamer audit — N532--N572, 2026-07-27

- The paper-guided branch implements the active legal 4,118-value emergent
  visual successor: recurrent discrete RSSM, training-only privileged decoder,
  255-bin distributional reward/value heads, BF16 microbatching, bounded
  memory-mapped replay, prior-only distillation, and the pinned DreamerV3
  all-time imagination-start rule. Peak measured CUDA memory stayed below
  `120 MB`, so local hardware capacity is not the active blocker.
- A direct audit of the paper's pinned DreamerV3 commit proves default actor
  imagination starts from all batch/time posteriors (`imag_last=0`), not only
  sequence endpoints. It also confirms reward/value `symexp_twohot` heads with
  255 bins, loss scales dynamics/reward/representation `1/1/0.1`, and free
  nats `1.0`. Detailed authority and PDF renders are in
  `docs/skydreamer_paper_review_2026-07-27.md` and
  `docs/assets/skydreamer_v1/`.
- N551 prior-only distillation lowers raw prior KL to `0.000327` and corrects
  exact-start pitch dynamics. N561 source-locks balanced, pre-action-anchored
  causal replay for all four CTBR channels; N563 retains correct pitch but
  decoder drift dominates weak roll/yaw/collective effects.
- N567 reward-only training lowers two-hot loss to `0.01258` but preserves the
  wrong action ordering. Early 64 Hz VQ2 rewards are mostly `1e-5..1e-3`, while
  the nearest nonzero pinned support bins are `+-0.1705577`. N569 reward scaling
  by 64 still reverses pitch and is rejected.
- N570's training-only decoded progress plus effort weights
  `[0,0.025,0.06,0.5]` passes an exact-start four-channel causal audit. N572
  nevertheless fails `0/64` from uninterrupted native starts: recurrent action
  means become positive pitch/collective, the vehicle flies backward, climbs
  to mean `z=60.21 m`, and times out with zero crash. Checkpoint/evaluation
  SHA-256 values are `0a6d77f3...`/`577ebc6e...`.
- N572 rejects that checkpoint and N549's stale Gaussian replay as the actor-
  start distribution; it does not close the emergent visual architecture.
  SkyDreamer's all-time starts assume replay continually refreshed by the
  current task policy. Continue the 4,118-value visual actor through alternating
  native collection/model/actor learning with current-policy provenance and
  replay-age metrics. Carry N570's centered effort objective into that online
  loop. Detector geometry, reconstructed state, official phase, classical
  teacher actions, imitation, and runtime overrides remain forbidden. No new
  FlightSim command or bounded attempt is authorized; Submission remains
  forbidden.

### VQ2 paper-guided actor frontier — N573--N591, 2026-07-27

- N573 proves fail-closed resume refresh; N574 turns the complete bounded
  replay over to current-policy data before one optimizer update. Resume,
  replay provenance, age, and turnover metrics are now source-locked.
- The pinned DreamerV3 actor uses the slow target critic, separately centered/
  scaled advantages, all `B*T` posterior starts, and cumulative predicted-
  continuation weights. The local implementation now matches those estimator
  properties; focused Dreamer/replay tests pass `33/33`.
- N581 is the first correct forward-control checkpoint in the emergent branch.
  It reaches Gate 1's plane in all 64 disjoint full starts with zero crash, but
  remains `0/64`; mean radial error is `3.03748 m`, dominated by `3.02240 m`
  vertical error.
- N582--N585 establish the horizon-64 replay-wide causal target
  pitch/roll/collective/yaw `-0.5/0/-0.1/0`. Collective `-0.1` beats zero on
  `512/512` stochastic starts, but only by mean return `0.0022983` against
  roughly `2.58` p10-p90 spread.
- N586 proves all `1,024` paper-matched starts fit at `1,007,411,200` allocated
  CUDA bytes. N587 uses 16 full-start updates at `1,020,971,520` bytes. N588
  then records the first unassisted emergent-policy Gate-1 passes: `11/64`,
  zero crash/timeout. Vertical error improves to `2.71434 m`, but negative roll
  raises absolute lateral error to `1.18906 m`; N587 is not deployable.
- N589 confirms zero roll/yaw remain replay-wide optima. N590's continuation-
  weighted first step improves roll but fails yaw. N591 repeats all starts four
  times (`4,096` rollouts) at `3,788,349,440` bytes peak, but roll/yaw still
  move away from zero and collective moves the wrong way. Reject both children.
- Retain N587 only as the offline `11/64` frontier. Hardware is not the active
  blocker. Before another native child, audit pinned stateful normalizers and
  the actor-head coupling, then preregister the smallest training-only method
  that preserves already-correct roll/yaw neural outputs while improving pitch/
  collective. The deployed policy must still emit one complete four-channel
  recurrent Puffer action with no runtime override.
- No post-N522 FlightSim command has been sent in this branch. Live actuation
  remains frozen; no bounded Training attempt is authorized, and VQ2
  Submission remains forbidden.

### VQ2 restricted actor/reward frontier — N592--N612, 2026-07-28

- N592 proves that directly reinterpreting the legacy actor as the pinned
  bounded Normal is not semantics-preserving and still corrupts roll/yaw.
  N593-N594 constrain training to neural pitch/collective mean rows; the actor
  still emits one complete recurrent four-channel vector at runtime.
- N595 rejects N594 at `0/64`, zero crash/timeout, but improves full-start
  terminal radial/vertical/absolute-lateral error to
  `2.19095/2.17037/0.25744 m`. Every episode reaches the Gate-1 plane and
  misses above it.
- N596-N601 reject longer frozen-world updates, a collective-only continuation,
  stale-replay explanations, and effort-coefficient changes. N604 is the
  retained mixed current-policy world parent, SHA-256 `44ed81a3...`; it records
  eight native centered-gate reward events with zero actor updates.
- N605-N606 prove decoded x/phase event proxies remain zero in imagination.
  N607 finds the refreshed learned reward weak but correctly ordered. N608's
  first collective-only update moves negative; N609's 16 updates reverse.
- N610's 2/4 branches pass direction while its 8 branch is damaged. N611
  preserves N610 and recovers only the missing calculation. The recovered
  eight-update branch is clean but fails direction at `-0.0100942`; N610's
  original rule selects four updates at `-0.0136196`. N611 model/report hashes
  are `05a4591a...`/`18209284...`.
- N612 screens the selected child once on N595's exact offset-700 seeds. It is
  `0/64`, zero crash/timeout. Vertical/radial/lateral errors improve only
  `0.021101/0.021123/0.001130 m`, ending at
  `2.149274/2.169831/0.256309 m`. Evaluation SHA-256 is `c692f810...`.
  Reject the selected child, do not screen the 2-update sibling, and close the
  N608-N612 fixed update-count branch.
- Retain N587 as the `11/64` completion frontier and N595/N612 as paired
  vertical-miss evidence. Next run a read-only, source-locked learned-reward
  collective gradient-SNR audit across normalization and imagination-sample
  count before another actor child. No post-N522 FlightSim command has been
  sent; live actuation and Submission remain forbidden.

### VQ2 reward/event-support frontier — N613--N623, 2026-07-28

- N613's fixed N604 gradient-SNR matrix rejects one/four samples and legacy/
  percentile normalization: every condition is only `1/8` negative with a
  positive mean/median virtual collective step.
- N615 measures correct local one-step reward direction on `98.63%` of starts
  but only `7.18e-9` advantage for collective offset `-0.10`. N604 contains
  eight gate events and only `8.95` expected supervised exposures.
- N616 validates a Puffer-only native event collector. N617 captures 128 valid
  17-step gate-event windows from all 64 agents and all six ordered phases;
  dataset SHA is `6a36b805...`. No optimizer, teacher, analytic action, or
  FlightSim path is involved.
- N618-N621 isolate event-world integration. Inherited Adam is wrong; fresh
  Adam at 25% events and `4e-5` slightly harms dense state; 12.5% events loses
  direction. One 25%-event step at `3e-5` improves event phase/reward and stays
  within the 1% dense gate.
- N622's independent 4/16/64 bracket improves phase MAE but worsens event reward
  at every count, predicts zero crossings, and increasingly corrupts dense
  privileged loss. N623 then shows the only dense-safe four-update diagnostic
  reverses local collective reward credit and scores `4/8` negative in every
  gradient condition. Reject all N618-N622 children.
- Next separate the objectives: phase-only RSSM/prior/privileged-decoder
  learning on N617 plus dense N604 replay with reward/actor frozen, then a
  separately gated frozen-latent reward-head calibration. Repeat multi-seed
  gradient SNR before any actor update. N587 remains the `11/64` Gate-1
  completion frontier. No post-N522 FlightSim command has been sent; actuation
  and Submission remain forbidden.

### VQ2 transition/representation diagnosis — N624--N632, 2026-07-28

- N624's isolated one-step phase regression improves event next-phase MAE but
  not event delta; N625's 4/16/64 bracket never predicts a crossing and longer
  updates create global non-event phase drift. N626/N627 crossing-BCE weights
  `0.10/0.07` improve aggregate event errors but increase dense delta error
  `16.67%/20.83%`. Close direct phase-regression/BCE training.
- N628 fits a group-disjoint linear event probe to deterministic N604 prior
  features. Held-out AUC/AP are only `0.5310/0.2396`; only `8/35` events rank
  above all four same-window negatives. Privilege counterfactual error is
  exactly zero.
- N629 localizes image, sensor, fused encoder, posterior, and prior features;
  no frozen linear representation passes. Best AUC is `0.5645`. N630 is a
  preserved read-only metric-shape failure; N631 is the corrected run.
- N631 proves the true continuous forward label is coherent: it is strictly
  decreasing and final-ranked in `35/35` held-out windows. N604's decoded
  forward correlation is only `0.0547`, its MAE is `216.6` true steps, and it
  is monotonic in `0/35`. The privileged pose/rate/action oracle remains
  training-only and final-ranks `35/35`.
- N632 shows no frozen image/encoder/posterior/decoder-hidden representation
  supports held-out continuous distance. Best correlation is `0.1005`; no
  output-head calibration is admitted. The next child must learn near-plane
  legal representation from N617 training groups, with validation/test groups
  disjoint and fixed N604 dense feature/action preservation. Reward and actor
  learning remain forbidden until continuous held-out and dense gates pass.
- N587 remains the `11/64` Gate-1 completion frontier. No post-N522 FlightSim
  command has been sent; live actuation and Submission remain forbidden.

### VQ2 successful-prefix representation frontier — N633--N669, 2026-07-28

- N633--N639 reject short-window posterior/sequence integration. Trust-region
  children preserve float32 dense actions but improve held-out plane
  correlation only by about `3.5e-5`; BF16's apparent gains are quantization.
  N644--N646 then reject frozen boundary state, scalar mask geometry, and a
  small nonlinear five-point probe on the original six-phase event corpus.
- N648 recovers 23,553 broad samples from N604 replay. N649 proves raw masks
  contain coarse range signal (`0.4548` held-out correlation), but phases 3/5
  remain ambiguous; N650 proves N604 recurrent state does not fix them.
  N651/N652 actor-frozen full-start world refreshes and N653's recurrent audit
  also fail. Do not resume any N633--N653 child.
- N654--N658 validate exact deterministic N587 event collection and isolate
  one hard latent near-tie. N659 collects 160 disjoint real Gate-1 transitions;
  N661 retains 128 at a strict `0.001` replay bound. N662 shows N587's exact
  carried state has only `0.0841` plane correlation; its apparently strong
  zero-state full-prefix result is an artificial recurrent-age clock. N663's
  nonlinear near-plane visual audit also fails.
- N664 source-locks N652's 125k-transition full-start replay: 23,402 Gate-1
  samples from 234 groups. N665 finds strong coarse mask range correlation
  (`0.844`) but `0.749 m` near-plane MAE and chance direction. N666's 64-step
  GRU improves to `0.510 m/58.3%` but fails; N652 has only one gate event, so
  this failure-heavy replay/probe route is closed.
- N667 collects 192 disjoint deterministic N587 successful prefixes of 321
  steps at offset 2000 with no crash or skipped history. N668 replay-filters
  128 from `189/192` eligible; dataset SHA is `c81768f0...`. Selected replay
  maxima are `0.000605/0.0000752/0` for deterministic/logits/mismatch.
- N669 defeats event-clock leakage with seven 128-step crop offsets and whole-
  event splits. Its selected mask-only recurrent probe passes untouched test:
  correlation `0.960515`, overall/near-plane MAE `0.249514/0.240912 m`, and
  direction `0.802309`. Tail-only correlation is below `0.019`. Legal visual
  history is sufficient when successful transition support is present.
- N669 probe weights are discarded and no model is promoted. Next design one
  actor-frozen N587 encoder/RSSM integration using only N668 training groups,
  exact legal crop boundaries, and fixed successful-prefix plus broad N652
  action anchors. Keep prior/decoder/reward/continuation/actor/critics exact;
  do not read N668 test labels until validation admission.
- N587 remains the `11/64` offline frontier. No post-N522 FlightSim packet was
  sent. Native policy screening, FlightSim control, and Submission remain
  unauthorized.

### VQ2 integrated successful-prefix representation — N670--N672, 2026-07-28

- Runtime authority remains one recurrent 4,118-input visual Puffer actor
  emitting the complete four-channel CTBR vector. Privileged forward distance
  is a training-only label; test/native/FlightSim paths remain frozen.
- N670 performs one source-locked soft-posterior representation step from
  N587. Process and exact-boundary gates pass, but the frozen N587 latent has
  only `-0.0629` validation correlation and the `1e-6` update changes it by
  just `0.000087`. Dense/action preservation and progress admission fail.
  Reject child/report SHA `e57c1079...`/`8ad66c8b...`.
- N671 jointly trains only encoder, RSSM sequence/posterior, and a training-
  only plane probe for 400 steps, with actor weights and task heads exact.
  Continuous-child correlation reaches `0.7392`, but numerical latent bounds
  select step 50 at `0.4392`; near-plane MAE never falls below `2.08 m` and
  direction peaks at `0.5798`. Reject child/report SHA
  `6a972d79...`/`4ddc35ab...`.
- N672 removes numerical latent constraints while retaining hard/soft action
  distillation. The unsaved step-400 state reaches `0.942461` correlation,
  `0.302188 m` overall MAE, and `0.803922` direction. It narrowly fails
  near-plane MAE at `0.542766 m`; validation/N652 action RMSE are
  `0.001016/0.001715`, above the `0.001` limits. The fixed selector therefore
  saves step 150, which fails all progress gates. Reject selected child/report
  SHA `2f2817dc...`/`1e4d9a63...`; it is not the high-signal state.
- N672 proves useful crossing-scale history can occupy an actor-functionally
  near-neutral N587 representation. Next preregister an export-only exact
  reconstruction of N672's eight-point trace to recover step 400 as a
  quarantined donor. Then freeze encoder/RSSM/posterior and separately
  calibrate the training-only plane probe plus distill actor outputs to fixed
  N587 actions on N668 training and N652 broad histories. Validation only may
  select; N668 test remains untouched until all original progress and
  `0.001/0.01` action gates pass.
- N587 remains the policy frontier at `11/64`, zero crash/timeout. No post-N522
  FlightSim packet has been sent. Native screening and Submission remain
  unauthorized.

### VQ2 representation recovery and held-out test — N673--N676, 2026-07-28

- N673's export-only attempt is rejected. Although it saves a similar
  step-400 state, its eight-checkpoint CUDA trace differs from N672 by up to
  `0.0160443`, above the preregistered `1e-7`; never use its checkpoint.
- N674 is a fresh seed-674 donor, not a replay claim. It source-locks and saves
  step 350 with `0.934467` validation correlation, `0.492837 m` overall MAE,
  `0.364160 m` near-plane MAE, and `0.809524` direction. Donor/report SHA are
  `2c5326ca...`/`ac46fa2c...`. It is calibration-only.
- N675 freezes every N674 representation tensor and independently calibrates
  the training probe plus actor imitation. Validation progress is
  `0.944367`, `0.316195/0.467785 m` overall/near MAE, and `0.827731`
  direction. N668/N652 action RMSE/max are
  `0.0001029/0.0005120` and `0.0001730/0.0019679`. All validation gates pass;
  checkpoint/report SHA are `ce643319...`/`5bfead2c...`.
- N676 consumes the one authorized N668 test read and formally fails. Test
  correlation/overall MAE/direction pass at `0.931859/0.328685 m/0.807359`,
  and all action gates pass, but near-plane MAE `0.501634717 m` exceeds the
  fixed `0.500000 m` bound by `0.001634717 m`. Report SHA is `aeb6cccf...`.
- Never reread or tune to the N668 test. The next branch must collect a larger
  episode-disjoint deterministic N587 successful-prefix corpus, freeze new
  validation/test groups before training, and use N668 only as consumed
  development data. Increase genuine near-plane training support; preserve the
  original progress and `0.001/0.01` action gates for one new test read.
- N587 remains `11/64`. Native policy screening, FlightSim, and Submission
  remain unauthorized; no post-N522 FlightSim packet has been sent.

### VQ2 disjoint successful-prefix corpus freeze — N677--N680, 2026-07-28

- N677 collects 416 deterministic N587 Gate-1 histories at disjoint episode
  offset 4000. It covers 64 agents and 412 vector steps with zero skipped
  histories, crash, teacher/analytic/sampled action, learning, or FlightSim
  packet. Corpus/report SHA are `0d5bf433...`/`23bb1d6c...`.
- N678 independently replay-filters the source. It finds 411 eligible and
  freezes 400 with selected deterministic/logit maximum errors
  `0.00174588/0.00007132`, zero stochastic mismatch, and action error
  `7.39470e-7`. Corpus/report SHA are `bd79043d...`/`daea39cd...`.
- N679 is rejected without retry because direct CLI import resolution failed
  before any corpus read or output write. Corrected source adds a direct CLI
  smoke test; the focused collector/filter/partition suite passes `10/10`.
- N680 performs the label-blind split using only `vector_step`. It freezes
  272/64/64 train/validation/test events in 268/64/64 whole groups with zero
  overlap. Manifest/report SHA are `91b9acf1...`/`75b70103...`. Test rows have
  not been evaluated and remain sealed.
- Next physically or logically isolate the sealed test rows before any fit,
  audit near-plane support on train/validation only, then repeat the N674 donor
  and N675 frozen-calibration separation at larger support. Preserve the
  `0.80/0.50/0.50/0.75` progress and `0.001/0.01` action gates. N668 remains
  consumed development-only evidence.
- N587 remains `11/64`. No native policy screen, reward/prior continuation,
  post-N522 FlightSim packet, or Submission action is authorized.

### VQ2 sealed partitions and near-balanced donor — N681--N687, 2026-07-28

- N681 exactly materializes the N680 manifest as 272 train, 64 validation, and
  64 sealed-test rows. SHA are `6bb788c0...`/`c8ded2bd...`/`ef43fd77...`;
  no test metric or content-based choice occurs.
- N682 proves every train/validation event has 43--49 near-plane samples, but
  overlapping uniform crops reduce effective near-plane exposure from `15.9%`
  unique to `8.86%` of loss samples. Test remains unopened.
- N683 rejects four-row reference replay at deterministic error `0.00766128`.
  N684 has a report-only axis bug and writes nothing. Corrected N685 proves
  fixed padded reference batch 416 passes at `0.00174588/0.00007123/0`; fixed
  4/64/336 all expose the same outlier. Report SHA is `98c59365...`.
- N686 passes a fixed-416 one-step smoke with inverse exposure and `2.0`
  near-plane plane-loss weighting; its checkpoint is quarantined.
- N687's 400-step donor is rejected. Selected step 400 passes correlation,
  overall MAE, donor near-plane MAE, action, replay, isolation, and resource
  gates at `0.903538/0.421591 m/0.500221 m`, but direction `0.713914` misses
  `0.75`. Child/report SHA are `52048c9b...`/`5febb6fa...`; never use it.
- Next correct the still-unweighted delta objective with inverse temporal-pair
  crop exposure plus the same fixed near-plane rule, smoke once, then run one
  fresh N587 donor only if the smoke passes. Do not relax gates or open test.
- N587 remains `11/64`; native/reward/prior/FlightSim/Submission remain frozen.

### VQ2 pair-weighted donor and sampler diagnosis — N688--N695, 2026-07-28

- N688 audits 78,064 unique train temporal pairs against 180,880 crop-pair
  exposures. Inverse pair exposure plus the fixed `2.0` either-endpoint
  near-plane multiplier raises expected near-pair share to `27.51%`; report
  SHA is `53af9139...`. N689 passes the one-update implementation smoke;
  report SHA is `48d4ce4f...` and its checkpoint is quarantined.
- N690 proves the pair-weighted objective can satisfy every donor metric. Its
  unsaved step 350 reaches correlation/overall/near/direction
  `0.909126/0.383560 m/0.472684 m/0.758557`, with all preservation gates.
  The old selector instead saves higher-correlation step 400, whose overall
  MAE regresses to `0.513399 m`; reject the checkpoint/report
  `560c921c...`/`808a4e3f...` and never reconstruct step 350.
- N691 changes no model or data. Its fail-closed selector audit proves that
  complete progress admission must outrank a higher single metric and selects
  step 350 on immutable N690 history; report SHA is `ff994c95...`.
- Fresh-seed N692 is rejected despite clean process/action gates: selected
  step 400 is `0.892961/0.423967 m/0.772668 m/0.676711`. N693 then proves its
  sampler saw near/pair signal at the 11th/8.5th lower percentiles. Across 200
  schedules those measures correlate `0.989558`; report SHA is `6689167a...`.
- N694 passes a default-off balanced-sampler smoke. N695 executes exactly
  three epochs: every 1,904 crop appears three times, every event 21 times,
  no crop is unseen, and dense anchors appear 22--23 times. Correlation,
  overall, and near-plane metrics pass at
  `0.880770/0.464102 m/0.503379 m`, but direction is only `0.646205`; reject
  checkpoint/report `0893c8f7...`/`ce8ea421...`.
- Balanced exposure removes a measured confound but is insufficient. Before
  another donor, audit representation-gradient alignment across independent
  random plane-probe initializations on a fixed N681-train batch, then compare
  an ephemeral probe-only balanced warmup with the N587 representation exact.
  Only a source-locked improvement in gradient agreement may admit a default-
  off warmup smoke and one fresh donor. Do not tune seeds or gates.
- The physical N681 sealed test SHA `ef43fd77...` remains unopened. N587 stays
  the `11/64` zero-crash/timeout native frontier. No native policy screen,
  reward/prior/actor continuation, post-N522 FlightSim packet, bounded run, or
  Submission action is authorized.

### VQ2 decoder-free representation continuation — N696--N703, 2026-07-28

- N696 rejects probe-only warmup: median representation-gradient cosine falls
  from `0.028238` to `0.010819`, and sign agreement improves only `0.010500`.
  N698's deterministic actor-insensitive channel cannot reach encoder or
  posterior. N699's actor-null/simplex-tangent channel does reach all three
  components but is formally rejected on a legacy N698 report-field mismatch.
- N700's report-only correction passes and authorizes only an implementation
  smoke. N701 passes that one-update smoke at tangent weight/scale
  `0.01/0.05`; its checkpoint is quarantined.
- N702 is a paired three-epoch donor and is rejected. Correlation/overall/near
  pass at `0.885188/0.456135 m/0.519258 m`, but direction is `0.656994`.
  N703's fixed tangent readout also fails at `0.639137` direction and
  `0.661581 m` near-plane MAE. Close this fixed tangent; never reuse N702 or
  sweep its weight.
- Next audit a train-only, decoder-free, rotation/sign-invariant covariance
  geometry for level and temporal-delta progress. Require nonzero finite
  encoder/sequence/posterior gradients and N587/actor immutability before a
  default-off one-update smoke. A random `SoftPlaneProbe` must not drive the
  representation update; any readout is fit afterward on frozen train features.
- N681 sealed test SHA `ef43fd77...` remains unopened. N587 stays `11/64`.
  Native/reward/prior/actor continuation, post-N522 FlightSim packets, bounded
  Training, and Submission remain forbidden.

### VQ2 invariant progress geometry — N704--N705, 2026-07-28

- N704 passes a train-only decoder-free level/delta covariance audit. Its loss
  is orthogonal-rotation and target-sign invariant, non-collapsed, and reaches
  encoder/sequence/posterior while N587 stays bit-exact. Report SHA is
  `73e70106...`.
- N705 passes the default-off one-update integration with the random probe
  detached from RSSM features. Invariant-only and total RSSM gradient maxima
  match; replay, sampling, action, mutation, resource, and isolation gates all
  pass. Quarantine checkpoint/report `c1f2d0cc...`/`c3e7fdb5...`.
- Before a donor, smoke one deterministic fixed-lambda ridge fitted only on
  frozen N681-train features. Validation may evaluate but never select the fit.
  Only then may one fresh 357-update invariant child start from N587, followed
  by the same frozen readout and unchanged donor gates.
- N681 sealed test remains unopened, N587 remains `11/64`, and native,
  reward/prior/actor, FlightSim, bounded Training, and Submission stay frozen.

### VQ2 invariant donor admission — N706--N710, 2026-07-28

- N706 passes the deterministic fixed-lambda ridge smoke on quarantined N705:
  one float64 lambda `0.001`, train-only fit, identical double-solve hash, no
  model mutation/checkpoint/test/native/live path. Report SHA is `b5f9a26d...`.
- N707 completes exactly 357 balanced updates from N587. Every crop is seen
  three times and every event 21 times; invariant gradients reach all RSSM
  components with the random probe detached. All preservation, action,
  parameter, process, resource, and isolation gates pass. Checkpoint/report
  SHA are `8f415f85...`/`2bb3f31c...`.
- N708 is rejected after a reportless CUDA/DXG termination. N709's CPU recovery
  is rejected on a NumPy-boolean JSON serialization defect after evaluation.
  Neither produced metrics or is retried unchanged.
- N710 fixes only serialization and admits N707. Its fixed train-only ridge
  has coefficient SHA `bd2b1e32...`, 512 active features, and validation
  correlation/overall/near/direction
  `0.957912/0.261334 m/0.443587 m/0.861979`. All source/process gates pass;
  model changes and optimizer/checkpoint/test/native/FlightSim counts are zero.
  Report SHA is `e7e3492f...`.
- Next adapt the N675 calibration/package path to physical N681 partitions,
  N707, and N710. First require a zero-model-update package: refit and store the
  exact fixed ridge, reproduce strict progress gates, and independently verify
  N681-validation plus N652 action RMSE/max `<=0.001/0.01`. N707's recorded
  action drift already satisfies those strict bounds, so do not optimize the
  actor unless a separately diagnosed and preregistered failure requires it.
- Only a bit-exact N707 model with fixed-readout auxiliary metadata and a fully
  passing validation report may authorize one mutation-free physical N681
  sealed-test read. The sealed test remains unopened. N587 stays the `11/64`
  native frontier; native/reward/prior/FlightSim/Submission remain frozen and
  no post-N522 FlightSim packet has been sent.

### VQ2 invariant posterior and prior continuation — N711--N722, 2026-07-28

- N711 packages N707 and the fixed N710 readout with all model tensors exact.
  N712 consumes the physical sealed test exactly once and passes at
  `0.945961` correlation, `0.315909 m` MAE, `0.386704 m` near-plane MAE, and
  `0.857515` direction. The sealed test is permanently consumed; never open,
  hash, score, or tune against it again.
- N713 improves paired native Gate 1 from `11/64` to `13/64`, zero
  crash/timeout. N714 rejects imagination: KL is about `0.179`, categorical
  agreement `43.75%`, fixed-readout bias reaches `+-67 m`, and gate-event
  reward RMSE is about `8.456`.
- N716's prior-only repair reaches KL about `0.0002`, agreement about `97.4%`,
  and action drift below `0.000164` RMSE, with posterior actions exact. N717
  still rejects soft-progress agreement at `0.265..0.284 m` MAE.
- N719 selects progress-level-only at `3e-7`; N720 reproduces the virtual step
  within `4.47e-8`. N721 validates all-horizon improvement to
  `0.1422/0.1433/0.1442/0.1543/0.1840 m`, with every other prior gate passing.
  N720 checkpoint/report SHA are `4d7cc2b6...`/`8cae6c61...`.
- N718 and N722 fail without output and must not be retried unchanged. Retain
  N720 offline, but do not use it for imagination. N723's exact full-train
  one-step aggregate surface improves MAE only `0.140267 -> 0.137356`; larger
  rates regress. Close that objective without a real step. Next audit a
  mutation-free train-only open-loop progress loss at horizons `1,2,4,8,16`,
  restore N720 after every virtual rate, and require improvement at every
  horizon before one real smoke and complete N721 re-admission.
- Reward repair follows only after prior admission; actor/native/FlightSim stay
  frozen. No post-N522 FlightSim packet has been sent and VQ2 Submission is
  forbidden.

### VQ2 N723--N735 closure and solve-first reset — 2026-07-28

- N723--N732 close the aggregate and direct open-loop prior audits. N731 is the
  retained offline diagnostic. N732's held-out direct open-loop MAE at horizons
  `1/2/4/8/16` is `0.135247/0.134998/0.131514/0.132079/0.157686 m`.
- N734 is rejected by N735 because held-out horizon 1 regresses
  `0.0001622075 m` versus N731 despite horizon 2--16 improvements. N735 report
  SHA-256 is
  `46bfe16694965f514b6030f685cea37cb2c7bcd99268b7fb4ab9e327bbaab7f1`.
  Do not resume the branch as N736. Never reopen/hash/score N712's consumed
  physical sealed test.
- The user explicitly superseded the former Dreamer-only, emergent-phase, and
  no-classical-label training requirements. Freeze N523--N735 as research
  evidence. The canonical goal is
  `docs/vq2_paper_guided_puffer_goal_prompt.md`.
- The solve-first actor is a single recurrent full-output Puffer policy:
  official JPEG -> intrinsic normalization -> causal all-red-pixel soft mask ->
  `64x64` -> compact CNN -> one GRU -> Gaussian actor deterministic mean ->
  fixed CTBR wire conversion. No selected contour, explicit geometry, course
  coordinates, pose, gate identity, phase switch, planner, blend, override, or
  fallback may enter deployment.
- Native privileged geometry is authorized only for offline oracle labels,
  curricula, reward, and a separate critic. Train with uninterrupted legal
  histories using BC/DAgger, then teacher-free recurrent PPO. Require true
  `0.75 m` aperture screens, zero runtime teacher blend, and success/collision
  admission before optimizing time.
- No post-N522 FlightSim packet has been sent. Keep live commands frozen until
  the new policy passes native exact/perturbed full-course admission, replay,
  export parity, and zero-command Windows shadow. Submission is forbidden.

### VQ2 solve-first preprocessing — VQ2-SF001, 2026-07-28

- `vq2_sf001_soft_red_mask_contract` implements the new preprocessing-only
  lineage without resuming N736. One official `640x360` JPEG is converted to a
  dense soft red response, area-resized to `64x36`, and placed in rows
  `[14,50)` of a zero `64x64` canvas. The full horizontal field is preserved;
  official `fx=fy=320`, `cx=320`, `cy=180` maps exactly to native
  `fx=fy=32`, `cx=cy=32`.
- The transform is causal and stateless and returns exactly `4096` contiguous
  `float32` values. It contains no contour/component/morphology call, target
  selection, corners, range, bearing, pose, confidence arbitration, gate
  identity, official index, or action path. Simultaneous red candidates are
  retained rather than ranked.
- The focused plus adjacent camera/ABI suite passes `31/31`. All eight retained
  official-v3391-renderer frames produce finite nonempty masks. The full JPEG
  path benchmarks at `9.262 ms/frame` (`107.968 Hz`) in the Linux environment.
  The retained fixtures are VQ1 v3391 renderer evidence, not VQ2 course
  coverage; the previously recorded VQ2 passive JPEG is absent from the
  current worktree and was not fabricated.
- Implementation/test/preregistration SHA-256 values are
  `1c220738ea9105bf35e50631f61df19ac29eb2ddefe338b6caf9530202d6dda9` /
  `baf246d019fc48f91c7ea33cc010a7ad2e1d66e4e9b7ee697c1eadee5ecb872a` /
  `7b9a7b2097bf937c2b017e11fbdd9362e4cd2e8e16853e6b67594d386c6b3305`.
  Report SHA-256 is
  `accacceefc75ddc1cea157cf7f725b9dbe7d9791d6add6ec7829a5493ee0e32f`.
- VQ2-SF001 is admitted only as offline actor preprocessing. It authorizes no
  checkpoint, shadow, reset, arm, bounded attempt, or Submission action. No
  FlightSim packet was sent and the consumed N712 sealed test was not touched.
  Next audit the native full-course oracle at the true `0.75 m` aperture; do
  not collect BC/DAgger labels until the teacher itself passes its reliability
  gate.

### VQ2 solve-first oracle — VQ2-SF002--SF011, 2026-07-28

- SF002--SF007 reject the legacy whole-course spline, direct active-gate P/PD,
  true-relative-velocity damping, and speed-only surfaces. None produced a
  six-gate finish. SF008's minimum-jerk segment reference is mathematically
  continuous but also scores `0/64`; these branches are diagnostic only and
  wrote no labels.
- SF009 adds one default-off, training-only alignment governor. It uses the
  active gate's privileged lateral/vertical error and vehicle velocity to
  reduce forward speed smoothly from `2.0` toward `0.2 m/s` until centered,
  then applies coupled roll/thrust PD with drag, gravity, bank, and pitch
  compensation. It has no next-gate lookahead, spline, phase-specific gain,
  stored plan, or runtime path. The deployed actor never receives this state.
- SF009 passes `64/64` fixed six-gate episodes at the true `0.75 m` aperture.
  SF010 then passes `512/512` independently randomized courses with zero
  collision/miss/out-of-order/timeout or action/wire/thrust/crossing-margin
  violation.
- SF011 is the admitted oracle: `4096/4096` disjoint randomized six-gate
  finishes, mean completion `98.078308105 s`, zero safety/envelope violation,
  and every gate sampled in every episode. Mean crossing radial error is
  `0.042277 m` at Gate 1 and `0.002579..0.005679 m` at Gates 2--6, all well
  below the fixed `0.10 m` admission limit.
- SF011 report SHA-256 is
  `f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e`.
  Executed controller/evaluator SHA-256 values are `47f5c40d...` and
  `98e099a3...`. The screen wrote zero labels and sent zero FlightSim packets;
  N712 remained untouched.
- The oracle is admitted only for offline BC/DAgger label generation. Before
  collecting, prove the saved dataset contains exactly the leading `4118`
  competition-legal values and next-observation executed CTBR labels, never
  the `34` privileged native targets. No student checkpoint, live shadow,
  bounded attempt, or Submission action is authorized yet.

### VQ2 solve-first legal dataset — VQ2-SF012, 2026-07-28

- SF012 runs the unchanged admitted SF009 oracle on `64` new randomized full
  courses, seed `42012`, and passes `64/64` with zero collision, miss,
  out-of-order, timeout, crossing-margin, action, wire-rate, or thrust-envelope
  violation. Mean completion is `98.56494140625 s`.
- The source-locked time-major dataset contains `403722` causal transitions.
  Episode lengths range `5126..7336` steps with mean `6308.15625`; every
  episode has one contiguous valid prefix and exactly one final terminal.
- Persistent actor data is only `mask[4096] uint8 + legal_tail[22] float32 +
  executed_action[4] float32`. It stores zero privileged values. Actions are
  recovered from the newest history slot of the next native observation;
  online terminal-inclusive and independent stored nonterminal history/action
  alignment errors are both exactly zero.
- SF012 report/metadata SHA-256 values are `9bb31d59...`/`21158fbc...`.
  Mask/tail/action SHA-256 values are `a1b94876...`/`b5b4b5cb...`/`a918d0b4...`.
  Collection sent zero FlightSim packets, performed zero student updates, and
  did not touch N712.
- SF012 admits this dataset for offline recurrent BC/DAgger only. Next build
  the smallest `256`-wide legal CNN plus one GRU and one four-channel bounded
  actor head, train with full episode order, then screen teacher-free. No
  checkpoint, shadow, bounded attempt, or Submission action is authorized yet.

### VQ2 variable-gate strategy revision — 2026-07-28

- SF013--SF068 and C001--C014 continued after SF012 in standalone
  `docs/vq2_sf*.md`/`docs/vq2_c*.md` records. Frontier: SF063/SF064 pass
  Gate 1 `512/512`, zero crash, but under-turn Gate 2 by `8.19 m`; SF068
  whole-actor PPO reached the Gate-2 plane only on an unconsolidated sampled
  trajectory while eroding Gate 1; C014 closed interpolation and recommends
  phase-local recurrent PPO on the measured true/alias fixture.
- Superseding corrections: the VQ2 course shows about `11` gates from the
  start line (all prior screens assumed `6`; engine supports `num_gates` up
  to `16` per instance), and the corrected mask-gap diagnostic (true `64 Hz`)
  measures maximum mask-empty gaps of only `359 ms` — the failure mode is
  post-crossing re-targeting ambiguity, not long blindness.
- The authoritative execution prompt is now
  `docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md`: variable
  `5..12` gate courses with a count-agnostic phase scalar (normalize by the
  fixed cap `16`), oracle re-admission per count, regenerated legal corpus,
  `>=256`-step BPTT recurrent BC with transition oversampling, DAgger on
  visited states, then phase-local PPO only if needed. Offline stages run on
  rented remote compute; handoff commit is `508286e`. All legality, evidence,
  and freeze contracts stand: FlightSim frozen after N522, Submission
  forbidden, N712 sealed test never reopened.

### VQ2 variable-gate environment — VG001/VG001B, 2026-07-29

- VG001 extends the existing SF010/SF011 course randomizer with default-off
  per-vector-environment gate counts. Enabled instances assign one fixed count
  in `5..12`, uniformly by instance index; 512 environments contain exactly
  64 of each count and episode reset does not resample it. Optional bounded
  validation enforces finite ordered, non-overlapping, reachable geometry
  while retaining the historical randomization streams and default behavior.
- The sole new public phase encoding is
  `clamp(active_gate_index, 0, 16) / 16`, held at `4 Hz`. Gate count is not an
  actor input. The legal `4096` mask and `22`-value tail are byte-identical in
  a five-versus-twelve-gate fixture with identical visible state; only public
  progress and episode termination may reveal advancement.
- Native regressions pass, including `4096` course constructions over counts
  `5..12`, and the focused Python suite passes `36/36`. The frozen SF012
  collector correctly fails its old source lock after these native changes;
  never relabel that six-gate artifact as the variable-count corpus.
- The original VG001 binding smoke passed the count distribution but was built
  in bf16 and is rejected as legal ABI evidence; preserve report SHA-256
  `0f67ff81...`. VG001B rebuilds with `--float`, proves
  `precision_bytes == 4`, and passes 128 forced episodes with exactly `0.125`
  mass per count before and after reset. Float extension/report SHA-256 values
  are `847d77ee...`/`0845ea46...`.
- VG002 is the next blocking gate: the unchanged training-only oracle must pass
  at least `507/512` randomized courses independently at each count
  `{5,8,11,12}`, true radius `0.75 m`, with zero collision and all fixed
  safety/envelope checks. Do not collect the new corpus or train a student
  before that aggregate passes. FlightSim remains frozen after N522, N712's
  consumed sealed test must never be reopened, and Submission is forbidden.
- Remote execution is source-locked and count-boundary resumable. Before any
  episode, `state.json` records the Git commit, source/extension hashes,
  runtime versions, seeds, and full contract. Count/aggregate reports are
  atomic and immutable; `--resume` accepts only the exact completed prefix and
  runs its first missing count. Once state exists, the Vast wrapper must not
  apt-install, pip-install, test, or rebuild. Sync state, reports, and runner
  log back after every count and before stopping/destroying the instance.

### VQ2 variable-gate oracle admission — VG002, 2026-07-29

- On-demand Vast instance `46201898` supplies one RTX 4090, 32 effective CPUs,
  65,547,376 KiB RAM, and 100 GB disk at `$0.334444/hour`. Bootstrap 001 failed
  closed before `state.json` or an oracle episode on a missing executable bit;
  preserve log SHA `40478684...`. Commit `84f4f080...` changes only the two
  executable modes and is the accepted recovery source.
- Remote preflight passes both native suites, float32 `sm_89` CUDA build,
  backend/precision assertions, and `34/34` tests. Extension SHA-256 is
  `ce5f1eac...`; runtime is Python `3.12.3`, Torch `2.10.0a0`, CUDA `13.1`.
- VG002 passes `512/512` independently at counts `5`, `8`, `11`, and `12`.
  Every safety/envelope count is zero. Mean completion is
  `79.078003/136.792145/194.592773/213.846924 s`; worst mean crossing radial is
  `0.042401 m`, below the preregistered `0.10 m` limit.
- Count report SHA-256 values are `23e5bab7...`/`c8b6b3e7...`/
  `bc785ff8...`/`36df6587...`. Aggregate/state SHA-256 values are
  `e649a20a...`/`e8060b45...`; local independent verification passes the full
  report hash chain, sources, runtime, seeds, metrics, and safety fields.
- Stage 0 is admitted. Proceed to one newly tagged 256-episode mixed-count
  SF012-lineage corpus, fixed per-instance counts `5..12`, legal mask/tail/held
  `/16` phase plus executed actions only, at least 1.5 million transitions.
  FlightSim remains frozen after N522, N712 remains permanently closed, and
  Submission is forbidden.

### VQ2 variable-gate legal corpus — VG003, 2026-07-29

- Source commit `cb63c138...` preregisters exactly one 256-agent/episode
  mixed-count collection at seed `429030`. Bootstrap 001 supplied an incorrect
  full expected-commit string and exited before state, extension build, test,
  or episode; its preserved log SHA is `7a0f7e18...`. Bootstrap 002 used the
  actual commit, passed both native suites and `43/43` Python tests, and built
  float32 `sm_89` extension SHA `52d60c20...`.
- VG003 is admitted: all `256/256` randomized courses finish with zero crash,
  timeout, missed/out-of-order gate, or crossing/action/wire/thrust-envelope
  violation. Each count `5..12` contributes exactly `32` episodes and
  successes; mean gates passed is `8.5`.
- The corpus contains `2,396,907` causal records (`15,266` maximum vector
  steps; episode min/mean/max `4,396/9,362.918/15,266`). Executed-action
  history alignment error is exactly zero. Held `/16` progress has zero
  off-tick change/decrease or encoding error and exactly `1,920` expected
  nonterminal increments.
- Final storage is `16,437,452,416` bytes: mask/action/tail/terminal/valid
  SHA-256 values are `463e0028...`/`c76fee3c...`/`fd28285d...`/
  `17c11ae...`/`c7e70a84...`. Stored training-only privilege and total-count
  values are both zero. Metadata/report/state SHA-256 values are
  `7a5d6359...`/`b8069d05...`/`27ef6b72...`; an independent local verifier
  accepts the full manifest, report/state chain, metrics, phase, and safety.
- Stage 1 is admitted. Stage 2 may train one 4,119-input CNN + single 256-wide
  GRU student with recurrent windows at least 256 steps and at least 3x
  transition-window exposure, then run one teacher-free 256-course held-out
  variable-count screen. FlightSim remains frozen and Submission forbidden.

### VQ2 recurrent BC padding-audit rejection — VG004, 2026-07-29

- VG004 source commit `3416a5fe...` passed its `14/14` focused tests and wrote
  source/runtime-locked epoch-zero state, then failed during epoch 1. The
  phase audit treated the required zero-filled invalid rows after a shorter
  episode as a public-phase decrease. This is an audit bug, not corpus
  corruption; VG003 already proved zero decreases on every valid prefix.
- Reject VG004 permanently. State/log SHA-256 values are `a078b441...` and
  `9adaabf1...`. Persisted completed epoch and optimizer updates are zero,
  though unpersisted first-epoch updates may have occurred in process memory.
- VG005 is authorized as one new tag with no reused state. It changes only
  decrease auditing to mask the current invalid row and adds a regression for
  phase `[0,1/16,0]`, valid `[1,1,0]`; all model, data, BPTT, oversampling,
  split, seed, and optimizer choices remain fixed. This is the first distinct
  Stage-2 failure. A second distinct failure requires diagnosis before another
  training variation.

### VQ2 variable-gate recurrent BC — VG005, 2026-07-29

- Corrected commit `25121c6b...` passed `16/16` tests and audited the entire
  VG003 layout before epoch zero: `1,920` valid held-phase increments, zero
  encoding error, and `1,511,189` invalid padding rows explicitly excluded.
- VG005 completed all 12 preregistered epochs in `636.501 s`, with `38,622`
  optimizer updates. Every epoch proves exactly `3.0x` exposure for all
  `1,680` transition agent-windows. The fixed validation selector chooses
  epoch 12 at weighted MSE `5.7281488e-5`; its 240 transition rows have mean
  four-channel MSE `0.00492927`.
- Checkpoint/report/completed-state SHA-256 values are `f686a35d...`/
  `bdcd2b38...`/`8d0d5908...`; the checkpoint has `778,184` parameters and
  one 256-wide GRU. Independent local loading verifies model/source/runtime,
  best epoch, history, dataset phase audit, exposure, optimizer counts, and
  zero actor privilege/teacher/FlightSim/N712/Submission safety fields.
- VG005 is the only Stage-2 candidate. Preregister one teacher-free exact
  VG006 screen over 256 new courses: 64 each at counts 5/8/11/12, at least
  90% aggregate full-course success and zero crash. It is not authorized for
  live or Submission use.

### VQ2 variable-gate teacher-free rejection — VG006, 2026-07-29

- Source commit `988b50fc...` passes both native suites and `22/22` focused
  tests, rebuilds the float32 extension to SHA `52d60c20...`, and screens the
  sole VG005 epoch-12 checkpoint on 64 new courses each at counts 5/8/11/12.
- VG006 is rejected without retry: `0/256` full courses. Behavior is count-
  invariant and localized at the Gate-1 -> Gate-2 transition: every count
  averages `1.03125` gates, `61/64` pass Gate 1, and `5/64` pass Gate 2; no
  course reaches Gate 3. Gate-1 mean crossing radial is `0.468--0.471 m`, and
  Gate-2 successful crossings are centered near `0.035 m`.
- Count 5/11 each record one crash; count 8/12 record zero, for two crashes in
  256. All other failures are missed-gate terminals. Transport is exact:
  executed-action error `0`, zero nonfinite/action/phase faults, and all
  teacher, student-update, FlightSim, N712, and Submission safety fields zero.
- Count report SHA-256 values for 5/8/11/12 are `7e9a75c2...`/
  `f33fb70c...`/`a19bbc31...`/`0b78cff2...`; aggregate/state hashes are
  `1f848400...`/`300df9e6...`. This is the first model acceptance failure;
  VG004 was a pre-acceptance audit implementation rejection. The prompt
  explicitly routes a single localized transition failure to Stage-3 DAgger.
  Preserve VG003 clean anchors and collect source-balanced VG005-visited
  Gate-2 failure states with oracle query labels; do not rerun VG006.

### VQ2 variable-gate DAgger collection rejection — VG007, 2026-07-29

- Source commit `77534145...` preregistered 256 new mixed-count rollouts at
  seed `429047`, driven entirely by VG005 with the SF016-admitted oracle used
  only as an offline query label. Bootstrap 001 stopped before state or action
  because the remote clone lacked the exact historical SF016 report; its log
  SHA is `5a41eae6...`. Supplying that immutable report did not change source.
- The actual VG007 collection is permanently rejected. It terminated all 256
  episodes but produced `60,887` records, below its arbitrary `100,000` floor;
  lengths are `113/237.840/1,436` min/mean/max. Retained legal-array audit
  proves one final terminal per episode, zero held-phase off-tick change,
  decrease, skip, or encoding error, and finite/enveloped oracle labels.
- Held public progress shows 242/256 episodes reached phase 1 and 18/256
  reached phase 2, yielding `27,982` phase-1 and `1,279` phase-2 records. The
  rejected collector failed to persist native metrics and executed-action
  parity, so this evidence cannot claim record volume was the only overall
  failed predicate. Its state/collection-log/postmortem SHA-256 values are
  `3bba5b43...`/`fe8726fb...`/`9f494eaa...`.
- Never train on, finalize, resume, or retry VG007. VG008 corrects the sampling
  design by using 512 new episodes/seed `429048`, retaining the 100,000-record
  floor, and persisting every predicate in a JSON rejection report before any
  failure. It does not weaken the crash, reach, transport, layout, or phase
  gates. No FlightSim packet was sent; N712 and Submission remain forbidden.

### VQ2 Stage-3 two-failure stop — VG008, 2026-07-29

- VG008 source commit `6cd6c1a5...` ran 512 new exact-uniform mixed-count
  episodes at seed `429048`. Bootstrap 001 used a mistyped expected full
  commit and stopped before state/build/test/action (log SHA `7a0f7e18...`);
  bootstrap 002 used the exact commit, passed native suites and `25/25` tests,
  and executed the sole rollout.
- Retained legal arrays prove `127,886` records, one final terminal per all 512
  episodes, lengths `111/249.777/2,048`, and phase histogram
  `[63,340,59,296,5,250,0,...]`. 485/512 reach phase 1 and 32/512 phase 2,
  passing the volume and reach gates. Phase cadence/monotonicity/encoding and
  oracle-query finite/action-envelope checks all pass.
- The computed admission Boolean was nevertheless false. Its intended full
  rejection report then failed before state update because the loop rebound
  the `output` Path parameter to a `RecurrentActorOutput`, causing
  `output.with_name()` to raise. The state remains `collecting`, but VG008 is
  rejected and may not be resumed or trained on. Log/state hashes are
  `94529c46...`/`ab1d4a76...`.
- Strong source-locked diagnosis is the erroneous zero crossing-margin gate:
  native code marks any accepted crossing above `0.50 m` radial even though
  the true aperture is `0.75 m`; the same VG005 policy measured
  `0.421875--0.4375` violation rates in VG006. This quality diagnostic
  contradicts deliberate DAgger collection of imperfect student states. Crash
  and terminal-inclusive transport metrics were lost with the failed report,
  so the diagnosis is explicitly an inference, not reconstructed fact.
- VG007 and VG008 are two distinct Stage-3 acceptance/evidence failures. Per
  the active prompt, stop before a third collection variation. Diagnosis JSON
  SHA is `f9d99a43...`; it specifies a future new-tag correction only if an
  explicit superseding decision authorizes it. FlightSim remains frozen after
  N522, N712 closed, and VQ2 Submission forbidden.

### VQ2 persistent authorization and VG009 preregistration — 2026-07-30

- The user explicitly superseded the preceding two-failure stop through the
  persistent goal prompt, SHA-256 `052ba7cb...`. The terminal objective is a
  verified competition-legal VQ2 R2 Submission finish, not an offline
  checkpoint or Training milestone. Continue evidence-driven new variations;
  never retry a rejected configuration unchanged. The prompt gives standing
  authority for command-free shadows and uniquely tagged bounded VQ2 Training
  only after their gates, and eventual Submission only after the complete
  three-finish promotion and source-lock gates.
- VG009 is preregistered at new tag
  `vq2_vg009_variable_gate_dagger_round1_corrected_512`, seed `429049`. It
  retains 512 exact-uniform count-5--12 episodes, the 100,000-record floor,
  Gate-1/Gate-2 reach, crash, hard safety, transport, terminal-layout, and held
  public-phase predicates. VG007/VG008 arrays remain quarantined.
- The causal source correction renames the recurrent actor result so it cannot
  shadow the output path and writes the complete predicate map to both the
  rejection report and terminal state before raising. It records native
  `crossing_margin_violation` diagnostically but does not reject legal
  `0.50--0.75 m` crossings inside the true `0.75 m` aperture. Crash and
  out-of-order/action/wire/thrust envelope checks remain admission gates.
- Collector/runner/test/preregistration SHA-256 values are
  `4f52642a...`/`8053e7c0...`/`d5105a8c...`/`530ac99e...`. Both native
  regression suites pass and the focused plus adjacent Python suite passes
  `28/28`; runner shell syntax also passes. The collector source-locks the
  persistent prompt and VG008 diagnosis before its first offline action.
- Preserved instance `46201898` could not restart because its host resources
  were unavailable. Replacement instance `46256686` supplies one RTX 4090,
  48 effective/96 visible CPUs, 128 GB RAM, and 100 GB storage at about
  `$0.3615/hour`. Bootstrap 001/002 failed before state/action on missing
  `omp.h`/`ccache`; log hashes are `b1828937...`/`97e857c...`. Bootstrap 003
  built the float32 backend, then failed before state/action because Torch's
  >=32-thread AMD GEMM path introduced at most one float32 ULP in an exact
  phase-zero transfer test; log SHA is `ecc0e47b...`. The source-locked runner
  now checks OpenMP/ccache and caps only Python preflight reductions to 16
  threads, where transfer is bit-exact. CUDA collection remains uncapped.
- The subsequent VG009 result is recorded below. This stage is offline native
  collection only; VQ2 live commands and Submission remain gated, and N712
  remains permanently closed.

### VQ2 corrected DAgger collection admitted — VG009, 2026-07-30

- Bootstrap 004 on source commit `7e0e4c76...` passed the float32 build and
  remote `28/28` preflight, then executed the sole seed-`429049` rollout with
  `resume_count=0`. VG009 is admitted with `123,992` legal query records from
  all `512` terminal episodes in `241.125994 s`; episode length min/mean/max is
  `111/242.171875/2,048`.
- Gate-1/Gate-2 reach is `483/512` (`94.3359375%`) and `32/512` (`6.25%`).
  There are `11/512` crashes (`2.1484375%`), below the fixed 5% collection
  gate. Out-of-order, action-envelope, wire-rate, and thrust-envelope metrics
  are zero, executed-action error is exactly zero, and gate-count mass is
  exactly uniform over 5--12.
- Crossing-margin rate is `240/512` (`46.875%`) and is correctly recorded with
  `admission_predicate=false`; it does not weaken the true `0.75 m` aperture.
  Public-phase records are `[63279,58370,2343,0,...]` with zero off-tick
  change, decrease, skip, or encoding error. Every episode has one contiguous
  valid prefix and one final terminal.
- Report/state/metadata/runner-log SHA-256 values are `72c1e41d...`/
  `88851898...`/`da20bd4e...`/`398b02cc...`. The full 4.2 GB dataset was
  synced locally; independent verification recomputed every file hash, both
  report chains, layout, record count, phase histogram, and all 20 admission
  predicates. Admission evidence SHA-256 is `71c4898f...`.
- The paid replacement instance is stopped with GPU cost zero and disk
  retained. VG009 authorizes only a new VG010 preregistered source-balanced
  recurrent refit starting from VG005, using VG003 clean anchors and VG009
  states at equal source objective weight, followed by a new teacher-free
  screen. Teacher plant actions, student updates during collection, FlightSim
  packets, N712 access, shadow, Training, and Submission actions remain zero.

### VQ2 source-balanced recurrent refit preregistration — VG010, 2026-07-30

- Preregister exactly one offline fit tagged
  `vq2_vg010_variable_gate_source_balanced_refit_001`, seed `429050`. It starts
  from frozen VG005 SHA `f686a35d...`, reads only admitted VG003 clean anchors
  and admitted VG009 visited-state labels, and never reads VG007/VG008.
- Every optimizer update pairs one clean and one failure-state chunk and gives
  each source exactly `0.5` normalized objective weight; raw clean record
  volume cannot dilute VG009. Recurrent state follows genuine episode
  boundaries through 256-step BPTT windows. Public-phase transition windows
  receive three updates from the same incoming state.
- Retain twelve epochs, four agents per source, AdamW `3e-5`, weight decay
  `1e-5`, gradient cap `1.0`, smoothness `1e-4`, and action weights
  `(1,1,4,1)`. VG003 agents 224--255 and VG009 agents 448--511 are disjoint
  validation sets. The unchanged VG005 parent is selection epoch zero.
- Numerical admission requires a child with lower equal-source validation
  score than VG005, strictly lower VG009 validation MSE, clean validation MSE
  at most `0.02`, exact equal-weight audit, and at least three transition
  exposures. Admission authorizes only a newly preregistered teacher-free
  screen; it authorizes no live action.
- The source-locked runner verifies every parent/dataset/admission hash,
  build/runtime prerequisites, both native regression suites, float32 vision
  ABI, and focused tests before epoch zero. Once resumable state exists, it
  performs no build, install, or tests. Completed resume also binds current
  commit/sources, checkpoint, report, and completed state.
- Trainer/runner/test SHA-256 values are `afc7ea57...`/`76178b04...`/
  `c8d0b306...`; preregistration SHA is `1de4c1ff...`. The
  authoritative VQ2 executable remains untouched at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`;
  teacher plant actions, FlightSim packets, N712 access, shadow, Training, and
  Submission authorization remain zero.

### VQ2 source-balanced recurrent refit admitted — VG010, 2026-07-30

- Source commit `67872b79...` passed remote native/vision regressions, a fresh
  float32 CUDA build/ABI check, and `22/22` focused tests before epoch zero.
  The sole seed-`429050` run completed all 12 epochs without resume in
  `3699.850431 s`, with `83,449` optimizer updates.
- Epoch 12 is selected. Equal-source validation weighted MSE falls from the
  VG005 baseline `0.2285099178` to `0.003846616384` (`59.4054x`). VG009
  validation falls from `0.4569625541` to `0.007483940194` (`61.0591x`), while
  clean validation is `0.000209292575`, safely below the fixed `0.02` cap.
- All 12 epochs have exact zero source-weight difference and minimum `3.0x`
  transition exposure. The actor has zero privileged input/teacher blend;
  FlightSim packets, sealed-test accesses, and Submission authorization are
  zero.
- Checkpoint/report/completed-state/runner-log SHA-256 values are
  `a0934753...`/`dfe943b5...`/`c7ae30ca...`/`a283a1fa...`. The completed
  resume verifier and an independent local audit both reproduce the full hash
  chain. Admission evidence JSON SHA is `c0fbfc72...`.
- The previously partial local VG003 corpus was recovered exactly from
  retained instance `46201898`: compressed archive SHA is `9c41a835...`, all
  five array hashes match metadata, and the local loader independently sees
  `2,396,907` records plus all `1,920` legal phase increments. Both Vast
  instances are stopped after evidence sync.
- VG010 is numerical fit evidence only. It authorizes one separately
  preregistered teacher-free deterministic screen on new courses; it does not
  authorize shadow, VQ2 Training, or Submission.

### VQ2 teacher-free variable-course screen preregistration — VG011, 2026-07-30

- Preregister exactly one resumable screen tagged
  `vq2_vg011_variable_gate_recurrent_teacher_free_256` for VG010 checkpoint
  SHA `a0934753...` and report SHA `dfe943b5...`. Run 64 fresh terminal
  episodes independently at counts 5/8/11/12, seeds
  `429056/429059/429062/429063`, for 256 courses total.
- VG010 emits every full four-channel action as its deterministic recurrent
  mean. Teacher blend/control, sampling, clipping, phase switch, scheduling,
  and fallback remain absent. Actor input stays the legal 4,119-value ABI with
  causal 4 Hz held official progress and zero total-count/native state.
- Admission requires at least 231/256 ordered full-course successes, zero
  crash/out-of-order/action/wire/thrust violation, finite enveloped actions,
  executed-action error at most `5e-5`, and exact held-phase transport.
  Crossing margin is reported with `admission_predicate=false`: the diagnostic
  radius is `0.50 m`, while the unchanged true aperture is `0.75 m`.
- The generic VG006 evaluator retains strict crossing-margin behavior by
  default; a thin VG011 wrapper explicitly binds the candidate, fresh seeds,
  runner/preregistration/evidence sources, and diagnostic-only setting. Its
  focused contract passes `10/10`, direct CLI invocation and shell syntax pass.
- Base/wrapper/runner/test/preregistration SHA-256 values are `dc91107e...`/
  `19b97cbc...`/`39fa5994...`/`fbae6b53...`/`552b1e09...`. VG011 remains
  offline-only; rejection routes to a new source-balanced DAgger round, while
  admission authorizes only later offline promotion gates.

### VQ2 teacher-free screen rejected — VG011, 2026-07-30

- Source commit `58b7f727...` passed both remote native suites, a fresh
  float32 CUDA build/ABI check, and `27/27` tests, then ran the sole fresh
  256-course screen. VG011 is rejected with `0/256` finishes, `193/256`
  crashes (`75.390625%`), `63/256` misses, and zero timeout.
- Per-count crash/miss values are 5g `45/19`, 8g `50/14`, 11g `49/15`, and
  12g `49/15`; mean gates passed are `0.671875/0.6875/0.6875/0.703125`.
  Only `176/256` episodes pass Gate 1 and none pass Gate 2, so failure is
  localized before course length matters.
- Transport is exact: executed-action error, phase off-tick/decrease/skip,
  non-finite action, and action-envelope faults are zero. The `0.0390625`
  crossing-margin rate is diagnostic and not causal. This is policy
  covariate shift, not checkpoint execution or native transport divergence.
- Aggregate/state/log SHA-256 values are `3e470b18...`/`f9e6b8c4...`/
  `a93947ab...`; all four count hashes and the terminal chain verify locally.
  Rejection evidence SHA is `53d5ed65...`. The replacement GPU is stopped
  after sync.
- Never retry VG011 unchanged. The next authorized repair is a newly tagged
  VG010-visited-state DAgger collection, followed by a refit that preserves
  both VG003 clean anchors and the earlier VG009 failure distribution. Live
  activity remains forbidden.

### VQ2 VG010-visited DAgger round 2 preregistration — VG012, 2026-07-30

- Preregister exactly one offline collection tagged
  `vq2_vg012_variable_gate_dagger_round2_vg010_visited_512`, seed `429064`.
  Run 512 exact-uniform count-5--12 terminal episodes for at most 2,048 steps
  each, with a 150,000-record floor and at least 60% Gate-1 reach. VG010 SHA
  `a0934753...` emits every deterministic four-channel plant action; the
  SF016 oracle is query-label only and student updates are zero.
- VG011 predicts about 188,820 records at this population. Its `193/256`
  crashes and zero Gate-2 reaches are the failure distribution VG012 must
  label, so crash and Gate-2 reach are recorded as diagnostics rather than
  corpus-admission predicates. This deliberate change affects collection
  only; all later teacher-free admission still requires zero crash and
  reliable full-course completion.
- Hard admission retains exact uniform count mass, 512 single-final-terminal
  layouts, record/length bounds, nonzero phase-1 records, zero out-of-order
  and action/wire/thrust envelope metrics, action-history error at most
  `1e-7`, and exact causal held-phase monotonicity/timing/encoding. Privileged
  native state and total gate count never enter stored student observations.
- The generic collector now exposes source-bound candidate/evidence and
  diagnostic-policy configuration while preserving VG009 defaults. The VG012
  wrapper locks VG010 checkpoint/report/admission, VG011 report/rejection,
  SF016, the goal prompt, oracle, runtime, compiled extension, and all source
  paths before its first action.
- Base/wrapper/runner/test/preregistration SHA-256 values are
  `2ad03c78...`/`baeb921d...`/`08759b2a...`/`30d7ff13...`/`94dcd090...`.
  Focused and adjacent tests pass `36/36`; Python compilation, direct CLI,
  shell syntax, and diff hygiene pass. No VG012 rollout has run yet. FlightSim
  remains frozen, N712 remains closed, and shadow, Training, and Submission
  remain forbidden.

### VQ2 VG012 postcollection failure and VG013 recovery — 2026-07-30

- VG012 ran once on retained RTX 4090 instance `46201898` from source commit
  `107963410014...`. Remote native/vision regressions, float32 CUDA build/ABI,
  and `36/36` tests passed before seed `429064` began. The 512-episode rollout
  reached writer finalization, but accepted-report serialization failed because
  `post_gate_1_records_present` remained a NumPy boolean. Never invoke VG012
  resume: its generic collecting-state path would delete the finalized arrays
  and rerun the policy unchanged.
- The finalized corpus has `171,903` records with length min/mean/max
  `110/335.748046875/2,048`. All 512 valid prefixes are contiguous and have
  one final terminal. Phase records are `[94697,77121,85,0,...]`; 371/512
  (`72.4609375%`) reach Gate 1 and 2/512 (`0.390625%`) reach Gate 2. All 373
  phase increments are monotone/exact `/16`; query labels are finite and
  inside `[-1,1]`.
- Array SHA-256 values for action/mask/tail/terminal/valid are
  `a7e5c358...`/`1f2c8590...`/`80b8fabf...`/`a17bc26a...`/`3c18a5a8...`.
  Metadata/state/log/archive hashes are `e7bfdba1...`/`75a89042...`/
  `c860ba57...`/`c2f868dc...`. The 51 MB post-exit preservation archive was
  copied and independently extracted locally; both paid instances are stopped.
- The generic collector now converts every predicate to a native Python bool
  before either report path; its fixed SHA is `cf0b8f6d...`. This fix does not
  authorize a second VG012 rollout.
- Preregister one deterministic recovery tagged
  `vq2_vg013_vg012_postcollection_recovery_001`. It recreates the same native
  seed/config and replays each recorded nonterminal action from the next legal
  action-history row. VG010 advances only to prove action parity and derives
  the one unavailable terminal action per episode. No new course is sampled.
- VG013 admission requires every original hard collection predicate, exact
  active/terminal layout, exact stored mask bytes, tail/action/executed-action
  error at most `1e-7`, all 171,903 actions and 512 terminals, exact phase
  records, and finite enveloped actor output. Recovery/runner/test/prereg hashes
  are `070f9b8d...`/`8511c9b1...`/`97c6bbe7...`/`8263105f...`; focused tests
  pass `40/40`, both native suites pass, and the full local corpus audit passes.
  VG013 has not run. FlightSim, N712, shadow, Training, and Submission remain
  forbidden.

### VQ2 VG012 corpus recovered and admitted — VG013, 2026-07-30

- VG013 ran once without resume on source commit `7a9a376a...`. Remote native/
  vision regressions, float32 build/ABI, and `40/40` tests passed before the
  recovery state. Full array hashing and the deterministic native replay then
  admitted the corpus; no VG012 policy rollout was repeated.
- All `171,903` actions replay exactly: 171,391 come from the next stored legal
  action-history row and 512 terminal actions are source-derived by VG010.
  Active rows, terminal events, stored mask bytes, legal tail, held phase,
  VG010 action, and executed-action errors are all exactly zero. Replayed phase
  records match `[94697,77121,85,0,...]` exactly.
- Native outcomes are diagnostic: `354/512` low crashes (`69.140625%`),
  `150/512` misses, `8/512` timeouts, 371 Gate-1 reaches and 2 Gate-2 reaches.
  Out-of-order/action/wire/thrust-envelope metrics are zero. All 20 original
  corpus predicates and all 14 recovery predicates pass.
- Recovery report/state/log SHA-256 values are `38358a58...`/`e78d48da...`/
  `eb0692c6...`; independent admission evidence SHA is `5c9d0e4b...`. The paid
  instance is stopped after sync. VG013 admits training data only and
  authorizes one separately preregistered refit preserving VG003, VG009, and
  recovered VG012 at explicit source weights. It does not admit VG010 or any
  live activity.

### VQ2 three-source recurrent refit preregistration — VG014, 2026-07-30

- Preregister one epoch-resumable fit tagged
  `vq2_vg014_variable_gate_three_source_refit_001`, seed `429065`, starting
  from VG010 SHA `a0934753...`. Use only VG003 clean, admitted VG009, and the
  VG013-recovered VG012 arrays; VG007/VG008 remain quarantined and the failed
  VG012 state is never resumed or rewritten.
- Every update pairs one recurrent chunk from each source and normalizes each
  loss independently before exact `0.50/0.25/0.25` clean/VG009/VG012 mixing.
  The clean stream defines an epoch; failure streams cycle only at episode-
  batch boundaries. Use 12 epochs, four agents per source, 256-step BPTT,
  three transition-window exposures, AdamW `2e-5`, weight decay `1e-5`,
  gradient cap `1.0`, smoothness `1e-4`, and action weights `(1,1,4,1)`.
- Reserve the final 32 clean agents and final 64 agents from each failure
  source for validation. Select minimum weighted `0.50/0.25/0.25` validation.
  Numerical admission requires a child better than VG010 overall and on
  recovered VG012, with clean and VG009 validation each at most `0.02`, exact
  source weights, and at least `3.0x` transition exposure.
- The dataset loader now accepts an explicitly hashed external admitted report,
  allowing VG012 arrays to remain immutable while using the VG013 recovery
  report. Loader/trainer/runner/test/preregistration hashes are
  `a3e4fb3e...`/`4795040a...`/`a30aecf4...`/`397c3b11...`/`9855d318...`.
  Focused tests pass `27/27`, both native suites pass, parent loading and the
  full recovered-dataset loader/hash/phase audit pass. VG014 has not run and
  authorizes no live activity.

### VQ2 three-source recurrent refit admitted — VG014, 2026-07-30

- VG014 completed its sole seed-`429065` run from source commit `92d77178...`
  with no resume. All 12 epochs and `88,890` optimizer updates completed with
  exact `0.50/0.25/0.25` source balance and minimum `3.0x` transition-window
  exposure.
- Epoch 8 is selected. Its source-balanced validation weighted MSE is
  `0.016582464965`, improving VG010's `0.145574128220` baseline by `8.78x`.
  Recovered-VG012 improves from `0.574393987536` to `0.058610854629`;
  VG009 improves from `0.007483940194` to `0.006772136153`. Clean validation
  is `0.000473434539`, below the fixed `0.02` cap.
- Checkpoint/report/completed-state/runner-log SHA-256 values are
  `60d69a1f...`/`067d9c19...`/`8d585f66...`/`56207a53...`. Remote and local
  completed-output verifiers pass, and synced hashes are exact. Admission
  evidence JSON SHA-256 is `073fc31e...`.
- Both Vast instances are stopped after artifact sync. VG014 is numerical fit
  evidence only and authorizes one separately preregistered fresh teacher-free
  deterministic screen. FlightSim, shadow, Training, and Submission remain
  forbidden.

### VQ2 fresh teacher-free screen preregistration — VG015, 2026-07-30

- Preregister exactly one resumable screen tagged
  `vq2_vg015_variable_gate_recurrent_teacher_free_256` for VG014 epoch 8,
  checkpoint SHA `60d69a1f...`, report SHA `067d9c19...`, and admission SHA
  `073fc31e...`. Run 64 fresh terminal episodes independently at counts
  5/8/11/12, seeds `429071/429074/429077/429078`, for 256 courses total.
- VG014 emits every complete four-channel action as its deterministic recurrent
  mean. Teacher blend/control, sampling, clipping, phase switch, scheduling,
  and fallback remain absent. Actor input stays the legal 4,119-value ABI with
  causal 4 Hz held official progress and zero total-count/native state.
- Admission requires at least 231/256 ordered full-course successes, zero
  crash/out-of-order/action/wire/thrust violation, finite enveloped actions,
  executed-action error at most `5e-5`, and exact held-phase transport.
  Crossing margin remains diagnostic-only at radius `0.50 m`; the unchanged
  true aperture is `0.75 m`.
- Base/wrapper/runner/test/preregistration SHA-256 values are `dc91107e...`/
  `261eaee1...`/`762ddc6c...`/`463fcc39...`/`234bb8ef...`. Focused and adjacent
  tests pass `27/27`; direct CLI invocation and runner shell syntax pass.
  VG015 is offline-only and authorizes no live activity.
- Bootstrap 001 stopped before build, test, state creation, or policy action
  because the newly added runner's executable bit was not represented in Git;
  exit code is `126`, and log/exit hashes are `86efa14b...`/`703d2c10...`.
  Preserve this preflight failure. Correct only the runner tree mode to `100755`
  and launch bootstrap 002 from the new source commit; the VG015 screen tag and
  all preregistered content hashes remain unused and unchanged.

### VQ2 teacher-free screen rejected — VG015, 2026-07-30

- Bootstrap 002 passed its source-locked preflight and completed all 256 fresh
  courses. VG015 is rejected with `0/256` finishes, `137/256` crashes
  (`53.515625%`), `119/256` misses, and zero timeout. Never retry it unchanged.
- Per-count crash/miss values are 5g `31/33`, 8g `37/27`, 11g `34/30`, and
  12g `35/29`. Gate-1 reach is `246/256`, Gate-2 reach is `2/256`, and no
  episode reaches Gate 3. Relative to VG011, Gate-1 reach improves from 176
  and crashes fall from 193, but failure remains localized to the Gate-2
  transition and is independent of course length.
- Transport is exact: executed-action error, phase off-tick/decrease/skip,
  phase-encoding, non-finite action, out-of-order, and action/wire/thrust
  envelope faults are zero. The `0.0703125` crossing-margin rate remains
  diagnostic and is not causal.
- Aggregate/state/log/exit SHA-256 values are `bb655666...`/`85d9d19b...`/
  `2de58ce7...`/`53c234e5...`; all count hashes and the terminal chain verify
  locally. Rejection evidence JSON SHA-256 is `568f948e...`. The GPU is stopped
  after sync.
- The next authority is a newly preregistered VG014-visited-state DAgger
  collection retaining VG003, VG009, and recovered-VG012 anchors. FlightSim,
  shadow, Training, and Submission remain forbidden.

### VQ2 VG014-visited DAgger round 3 preregistration — VG016, 2026-07-30

- Preregister exactly one offline collection tagged
  `vq2_vg016_variable_gate_dagger_round3_vg014_visited_512`, seed `429079`.
  Run 512 exact-uniform count-5--12 episodes for at most 1,024 steps, with a
  350,000-record floor and at least 90% Gate-1 reach. VG014 SHA `60d69a1f...`
  emits every deterministic four-channel plant action; the SF016 oracle is
  query-label only and student updates are zero.
- The shortened 16-second horizon retains the causal prefix through VG015's
  Gate-1 pass near step 233 and the failing Gate-2 phase while preventing long
  post-miss divergence from dominating labels. Crash, Gate-2 reach, and
  horizon timeout remain corpus diagnostics; later teacher-free admission is
  unchanged and still requires zero crash plus reliable full-course finish.
- Hard admission retains exact uniform count mass, 512 single-final-terminal
  layouts, record/length bounds, nonzero phase-1 records, zero out-of-order and
  action/wire/thrust-envelope metrics, query finiteness/envelope, action-history
  error at most `1e-7`, and exact causal held-phase timing/encoding. Privileged
  native state and total gate count never enter stored student observations.
- Generic/wrapper/runner/test/preregistration SHA-256 values are `cf0b8f6d...`/
  `c988ed48...`/`7991d1b1...`/`e1f79b79...`/`3357b8aa...`. Focused and adjacent
  tests pass `36/36`; direct CLI and runner shell syntax pass. VG016 grants no
  live authority.

### VQ2 transition DAgger corpus admitted — VG016, 2026-07-30

- VG016 completed its sole seed-`429079` run from source commit `cc8ae76e...`
  without resume. It produced `370,989` legal records across 512 exact-uniform
  count-5--12 episodes; length min/mean/max is
  `245/724.587890625/1,024` and all 20 hard predicates pass.
- Phase records are `[122224,244180,4585,0,...]`. All 510 increments are
  monotone/exact `/16`; action-history error, phase timing/encoding error,
  non-finite query labels, and query/action/wire/thrust envelope faults are
  zero. Stored privileged and total-count values are zero.
- Diagnostic outcomes are 82 low crashes, 251 misses, and 179 horizon
  timeouts. Gate-1 reach is `491/512` (`95.8984375%`), Gate-2 reach is
  `19/512` (`3.7109375%`), and Gate-3 reach is zero. These are precisely the
  early transition states selected for labeling and are not policy admission.
- Report/metadata/state/archive/log/exit SHA-256 values are `bcd86c8b...`/
  `4f075d2d...`/`67fd34e4...`/`e318732a...`/`224bbc5a...`/`9a271f2a...`.
  The full 2.1 GB arrays were synced via the 116 MB archive and locally pass
  every hash, shape, dtype, record, terminal-layout, and phase audit. Admission
  evidence JSON SHA-256 is `eb6c4f56...`; both GPU instances are stopped.
- VG016 authorizes only a new source-balanced recurrent refit preserving
  VG003, VG009, and recovered-VG012 while adding VG016. FlightSim, shadow,
  Training, and Submission remain forbidden.

### VQ2 four-source recurrent refit preregistration — VG017, 2026-07-30

- Preregister one epoch-resumable fit tagged
  `vq2_vg017_variable_gate_four_source_refit_001`, seed `429080`, starting from
  VG014 SHA `60d69a1f...`. Use only admitted VG003 clean, VG009, recovered
  VG012, and VG016 datasets; quarantined and failed collector states remain
  excluded.
- Every update pairs one recurrent chunk from all four sources and normalizes
  each loss independently before exact `0.40/0.15/0.15/0.30` clean/VG009/
  VG012/VG016 mixing. The clean stream defines 12 epochs; use four agents per
  source, 256-step BPTT, three transition exposures, AdamW `2e-5`, weight decay
  `1e-5`, gradient cap `1.0`, and smoothness `1e-4`.
- Reserve final 32 clean and final 64 agents from each DAgger corpus. Select
  minimum weighted four-source validation. Admission requires a child better
  overall and on VG016, clean/VG009 each at most `0.02`, recovered VG012 at
  most `0.10`, exact weights, and at least `3.0x` transition exposure.
- Trainer/runner/test/preregistration SHA-256 values are `aeb629cd...`/
  `3a06b184...`/`670c6dd9...`/`422e59b9...`. Focused and adjacent tests pass
  `31/31`; parent loading and the full VG016 loader/hash/phase audit pass.
  VG017 has not run and authorizes no live activity.

### VQ2 four-source recurrent refit admitted — VG017, 2026-07-30

- VG017 completed its sole seed-`429080` run from source commit `68e024c8...`
  with no resume. All 12 epochs and `94,428` optimizer updates completed with
  exact `0.40/0.15/0.15/0.30` source weights and minimum `3.0x` transition
  exposure.
- Epoch 11 is selected. Balanced validation improves from `0.112781651689` to
  `0.019213320459` (`5.87x`), while VG016 improves from `0.342616097521` to
  `0.037651743717` (`9.10x`). Recovered VG012 also improves to
  `0.043122754397`; clean/VG009 are `0.001008161865/0.006974129592`, both well
  within their fixed caps.
- Checkpoint/report/state/log/exit SHA-256 values are `6769476f...`/
  `476c7306...`/`4a723fd5...`/`5244cbd3...`/`9a271f2a...`. Remote and local
  completed-output verifiers and sync parity pass. Admission evidence JSON
  SHA-256 is `df664cba...`; both GPU instances are stopped.
- VG017 is numerical evidence only and authorizes one separately preregistered
  fresh teacher-free screen. FlightSim, shadow, Training, and Submission
  remain forbidden.

### VQ2 fresh teacher-free screen preregistration — VG018, 2026-07-30

- Preregister exactly one resumable screen tagged
  `vq2_vg018_variable_gate_recurrent_teacher_free_256` for VG017 epoch 11,
  checkpoint/report/admission SHA values `6769476f...`/`476c7306...`/
  `df664cba...`. Run 64 fresh terminal episodes independently at counts
  5/8/11/12 with seeds `429085/429088/429091/429092`.
- VG017 emits every complete four-channel action as its deterministic recurrent
  mean. Teacher blend/control, sampling, clipping, phase switch, scheduling,
  fallback, total-count input, and privileged state remain absent.
- Admission requires at least 231/256 ordered finishes and zero crash,
  out-of-order, non-finite, action/wire/thrust-envelope, action-history, or
  held-phase transport fault. Crossing margin remains diagnostic-only at
  `0.50 m`; true aperture remains `0.75 m`.
- Base/wrapper/runner/test/preregistration SHA-256 values are `dc91107e...`/
  `2eaacd3c...`/`f02586e8...`/`0a6e6d4b...`/`4b0e77b2...`. Focused and adjacent
  tests pass `27/27`; CLI and runner syntax pass. VG018 is offline-only.

### VQ2 teacher-free screen rejected — VG018, 2026-07-30

- The sole source-locked screen completed all 256 fresh courses and is
  terminally rejected: `0/256` finishes, `168/256` crashes (`65.625%`),
  `88/256` misses, and zero timeout. Never retry VG018 unchanged.
- Gate-1 reach is `254/256`, Gate-2 reach is `4/256`, and one episode reaches
  Gate 3. Count-specific crash/miss values are 5g `36/28`, 8g `37/27`, 11g
  `48/16`, and 12g `47/17`; `143/168` crashes are low crashes. The same early
  transition failure occurs at every requested course length.
- Transport is exact: executed-action error, phase off-tick/decrease/skip,
  phase encoding, non-finite action, out-of-order, and action/wire/thrust
  envelope faults are zero. The `5/256` crossing-margin flags remain
  diagnostic and are not causal.
- Aggregate/state/runner-log SHA-256 values are `40be8bf0...`/`8b106a11...`/
  `c82644ad...`; all count hashes and remote/local parity verify. The detached
  launch's auxiliary exit tracker was malformed before the runner began and
  contains only a newline (`01ba4719...`); the screen result is unaffected,
  and a no-rollout completed-resume audit captured exit `2` with log/exit hashes
  `40be8bf0...`/`53c234e5...`. Rejection evidence SHA is `b0d7ddb3...`.
- Relative to VG015, reach improves marginally but crashes rise by 31, so the
  next authority is a new VG017-visited early-transition DAgger corpus under a
  fresh tag/seed, preserving VG003/VG009/recovered-VG012/VG016 anchors.
  FlightSim, shadow, Training, and Submission remain forbidden.

### VQ2 VG017-visited DAgger round 4 — VG019, 2026-07-30

- Preregister one resumable offline collection tagged
  `vq2_vg019_variable_gate_dagger_round4_vg017_visited_512`, seed `429093`.
  Run 512 exact-uniform count-5--12 episodes for at most 1,024 steps. Require
  at least 300,000 records and 95% Gate-1 reach.
- VG017 SHA `6769476f...` emits every deterministic full-output recurrent plant
  action. The SF016 oracle supplies query labels only. The unchanged legal
  4,119-value observation includes the causal held `/16` public phase but no
  native state or total-count value.
- VG018's 168 crashes and near-zero later-gate reach are the distribution being
  labeled, so crash/Gate-2/miss/timeout remain corpus diagnostics. Hard gates
  retain exact episode/layout/count mass, finite enveloped labels, action-
  history parity, held-phase timing, and zero ordering/action/wire/thrust fault.
- Generic/wrapper/runner/test/preregistration SHA-256 values are
  `cf0b8f6d...`/`61b6e594...`/`b838bcd7...`/`03b6a5c5...`/`b7ab6673...`.
  Focused and adjacent tests pass `36/36`; both native regression suites pass,
  and CLI/runner syntax checks pass. VG019 is offline-only and, if admitted,
  authorizes only a five-source refit retaining all four prior anchors.

### VQ2 VG017-visited DAgger corpus admitted — VG019, 2026-07-30

- Bootstrap 001 stopped before state/action because the fresh worker lacked
  frozen VG006/SF016 evidence required by the test shard; log/exit hashes are
  `24bd2b03...`/`4355a46b...`. After transferring only those exact reports,
  bootstrap 002 passed both native suites and `36/36` tests and ran once from
  commit `965b1a23...`.
- VG019 admitted `475,453` legal records from 512 exact-uniform count-5--12
  courses. Episode length min/mean/max is `298/928.619140625/1,024`; all 20
  hard predicates pass. Phase records are `[132553,340705,2195,0,...]` with
  521 exact held `/16` increments.
- Diagnostic outcomes are 170 low crashes, 77 misses, and 265 horizon
  timeouts. Gate-1 reach is `508/512`, Gate-2 reach is `13/512`, and Gate-3
  reach is zero. Action-history/phase errors, non-finite labels, and hard
  ordering/action/wire/thrust faults are zero.
- Report/metadata/state/archive/log/exit SHA-256 values are `2079ce9d...`/
  `773273fc...`/`e003edf3...`/`b9632600...`/`3d7847eb...`/`9a271f2a...`.
  The full 2.1 GB corpus was synced through the 131 MB archive; every array
  hash, shape/dtype, record/layout, terminal, phase, and action-envelope audit
  passes locally. Admission evidence SHA is `9b110d50...`.
- Instance `46275514` is stopped with retained disk after sync. VG019 admits
  training data only; next authority is a separately preregistered five-source
  refit preserving VG003/VG009/recovered-VG012/VG016 and adding VG019.
  FlightSim, shadow, Training, and Submission remain forbidden.

### VQ2 five-source recurrent refit — VG020, 2026-07-30

- Preregister one 12-epoch fit from VG017, seed `429094`, pairing VG003,
  VG009, recovered VG012, VG016, and VG019 chunks on every update at exact
  normalized weights `0.35/0.10/0.10/0.15/0.30`.
- Use four agents per source, 256-step BPTT, four transition exposures, AdamW
  `1e-5`, weight decay `1e-5`, gradient cap `1.0`, and smoothness `1e-4`.
  The lower learning rate protects Gate 1 while the new VG019 states and higher
  transition exposure target the persistent Gate-2 failure.
- Select minimum fixed-weight five-source validation. Numerical admission
  requires an overall and VG019 improvement, clean/VG009 MSE at most `0.02`,
  each later source at most `0.10`, exact weights, and `>=4.0x` transition
  exposure. Admission can authorize only a fresh offline screen.
- Trainer/runner/test/preregistration SHA-256 values are `3421998f...`/
  `b44c5572...`/`eaab4e5e...`/`3e7849d3...`. Focused and adjacent tests pass
  `36/36`, all five dataset loaders/hash/phase audits pass, both native suites
  pass, and CLI/runner syntax passes. VG020 authorizes no live activity.

### VQ2 VG020 epoch-3 throughput supersession / VG021 — 2026-07-30

- VG020 ran cleanly on source commit `83a48e63...` through atomic epoch 3,
  then was intentionally stopped before any completed epoch was lost. Epoch 3
  is the new best at fixed-weight MSE `0.018124410839263784`; clean/VG009/
  VG012/VG016/VG019 values are `0.0015846671/0.0072325038/0.0459751276/
  0.0335440650/0.0240580149`. Exact weights and `4.0x` transition exposure
  pass. The state/log hashes are `3d4694ba...`/`6f68f2ac...`; intentional exit
  is `143`. VG020 is incomplete and grants no admission.
- The measured cause is host decode and underfilled CUDA work: the trainer used
  about 32 CPU cores while 8/10 GPU samples were `0--1%`. Production-shape
  local probes show bitwise-identical device mask decode at `20.42 -> 1.65 ms`
  median and bitwise-identical same-horizon grouping at `87.73 -> 31.20 ms`
  forward/backward median.
- VG021 preregisters one exact-state continuation tagged
  `vq2_vg021_five_source_accelerated_continuation_001`. It binds parent state
  SHA `3d4694ba...`, preserves model/optimizer/scaler/all RNG/history/config,
  and may execute epochs 4--12 only after actual-chunk decode/output/state/
  loss/gradient parity and at least `1.5x` throughput. No padding, recurrent
  horizon change, source-weight change, or optimizer step before parity is
  allowed.
- Loader/trainer/parity/runner/recurrent-test/refit-test/preregistration hashes
  are `a9bfc4a7...`/`3e80c744...`/`9689d3ac...`/`7cf0e97c...`/
  `a17a898a...`/`cb6efe56...`/`b86f202b...`. The exact suite reached `39`
  passing tests plus one corrected documentation-string assertion; the focused
  correction passes and remote source-lock reruns the full suite before parity.
- Vast resource discipline is corrected: historical instances `46201898`,
  `46256686`, and `46275514` are `exited/stopped` with retained disks. Only
  active worker `46280503` remains running. Epoch states/logs are synced under
  persistent local `/root/vq2_evidence_sync`. FlightSim packet count remains
  zero; shadow, Training, and Submission remain unauthorized.

### VQ2 bitwise device-decode continuation — VG022, 2026-07-30

- VG021 source batching is rejected before any optimizer step. Four-source
  grouping changed real FP16 actor outputs by up to `0.0013427734`. Pairwise
  grouping made mean/pre-tanh/state/loss exactly equal, but gradient error
  `1.52587890625e-5` exceeded the frozen `1e-5` cap and speedup `1.4892539843`
  missed the frozen `1.5x` floor. Never retry VG021 unchanged; local diagnostic
  evidence SHA is `442c33c3...`.
- Preregister one causally distinct continuation tagged
  `vq2_vg022_five_source_device_decode_continuation_001`. Remove grouping
  entirely and preserve the original five actor calls/order. The sole runtime
  optimization transfers compact uint8 masks before exact CUDA float expansion
  and removes redundant copies of already-contiguous arrays.
- Before epoch 4, the locked original Vast runtime must prove bitwise actor
  observations, outputs, states, loss, and every gradient on actual next-epoch
  chunks, zero optimizer/state writes, and `>=1.15x` median time for a complete
  transition step (five decodes plus four unchanged exposures). Any failure
  rejects VG022 without unchanged retry.
- Loader/trainer/checker/runner/recurrent-test/refit-test/preregistration hashes
  are `7d156466...`/`c9500787...`/`4c4be374...`/`54cba718...`/
  `a17a898a...`/`8de22270...`/`60486654...`. Focused source/artifact tests pass
  after a corrected wrap-independent documentation assertion; compilation and
  runner syntax pass. The remote runner reruns the exact full suite before the
  parity gate. No live authority exists.
- Bootstrap 001 from commit `e7ae962f...` failed closed before tests, parity,
  state, or optimizer work because the runner compared abbreviated package
  labels instead of the migration state's exact runtime strings. Log/exit
  hashes are `67806dd0...`/`4355a46b...`; failure evidence SHA is
  `a84ba484...`. Recovery reads the exact expected runtime from the hash-bound
  migration state. Never retry bootstrap 001 unchanged.
- Bootstrap 002 from commit `60c4b812...` also failed before tests/parity/state/
  optimizer work: its exact-value dictionary omitted the canonical manifest's
  `machine=x86_64` field. Log/exit hashes are `97c406f3...`/`4355a46b...`;
  evidence SHA is `789ad575...`. Bootstrap 003 removes duplicate manifest logic
  and calls the trainer's `runtime_manifest()` directly. Never retry bootstrap
  002 unchanged.
- Bootstrap 003 from commit `2a87bde1...` passed both native suites, a fresh
  float32 SM89 build, and `39/39` tests. Its zero-step parity gate admitted with
  bitwise observation/decode/output/state/loss/gradient equality and a
  `5.218184021x` complete-transition speedup (`1.684904829 -> 0.322891033 s`).
  It then failed before VG022 state creation or optimizer work because loading
  the migration payload directly onto CUDA moved the CPU Torch RNG ByteTensor.
  Parity/log/exit hashes are `f36c3b11...`/`f11e56dd...`/`4355a46b...`;
  failure evidence SHA is `3a3f6d49...`. Bootstrap 004 loads the complete
  payload on CPU, after which model/optimizer loaders migrate parameter state
  and CPU/CUDA RNG restoration retains the required tensor representation.
  Never retry bootstrap 003 unchanged.

### VQ2 VG022 accelerated refit admission / VG023 screen — 2026-07-30

- Bootstrap 004 from source commit `5cc119c6...` passed both native suites, a
  fresh float32 SM89 build, and `40/40` tests. Its zero-step parity rerun is
  bitwise exact for observations, decode, actor outputs, recurrent state, loss,
  and every gradient, with `5.176540700x` complete-transition speedup. Parity
  report SHA-256 is `2bc7ac8c...`.
- Loading the complete migration payload on CPU preserved CPU/CUDA RNG tensor
  representation while model weights and Adam moments migrated correctly.
  VG022 atomically continued the exact VG020 epoch-3 state through epochs
  4--12. The nine accelerated epochs took `6,812.492 s` total, about `12.62`
  minutes each versus VG020's `77--80` minutes.
- Epoch 9 is selected at fixed-weight MSE `0.015374308139712155`, improving the
  five-source baseline by `3.036x`. Clean/VG009/recovered-VG012/VG016/VG019
  values are `0.001098835/0.005883390/0.042725731/0.030032797/0.018746281`;
  all caps pass. All 12 source-weight audits and exact `4.0x` transition
  exposures pass after `126,455` optimizer updates.
- Checkpoint/report/completed-state/log/exit SHA-256 values are `bd3f93d4...`/
  `23d28182...`/`96c5e83e...`/`bdf48bda...`/`9a271f2a...`. Remote and canonical
  local completed-output verifiers pass, and every epoch state was synced.
  Numerical admission evidence SHA-256 is `ac93a00d...`.
- Preregister exactly one fresh screen tagged
  `vq2_vg023_variable_gate_recurrent_teacher_free_256`: 64 terminal episodes
  each at counts 5/8/11/12, seeds `429101/429104/429107/429108`. VG022 emits
  every deterministic recurrent four-action output; teacher blend/control,
  sampling, clipping, fallback, and privileged input remain zero.
- Admit VG023 only at `>=231/256` ordered finishes with zero crash, ordering,
  action/wire/thrust, non-finite, action-history, or public-phase fault. The
  `0.50 m` crossing margin remains diagnostic at the real `0.75 m` aperture.
  Generic/wrapper/runner/test/preregistration hashes are `dc91107e...`/
  `fdacf95f...`/`b03974cb...`/`c922b944...`/`20e4c142...`. VG023 is offline
  only; FlightSim, shadow, Training, and Submission remain unauthorized.

### VQ2 VG023 rejection / VG024 visited-state collection — 2026-07-30

- VG023 is terminally rejected: `0/256` finishes, `193` crashes (`91` low,
  `101` lateral/XY, `1` high), `63` misses, and zero timeouts. Gate-1 reach is
  `255/256`, Gate-2 reach is `26/256`, and no course reaches Gate 3.
- Relative to VG018, Gate-2 reach rises from `4` to `26`, but crashes rise by
  `25`; the crash distribution shifts from `143` low crashes to `101` lateral
  crashes. Counts 5/8/11/12 reach Gate 2 on `5/9/6/6` courses, so the frontier
  is transition-local rather than course-length-specific. The stronger turn
  acquires Gate 2 more often but crosses the lateral-safety frontier and does
  not consolidate downstream progress.
- Executed-action error, phase off-tick/decrease/skip, raw phase encoding,
  non-finite action, ordering, action/wire/thrust envelope faults are all zero.
  Aggregate/state/log/exit hashes are `2a3023fd...`/`eb2ddaa8...`/
  `058497f9...`/`53c234e5...`; completed-resume verification returns `2`
  without state change. Rejection evidence SHA is `88f403f0...`; never retry
  VG023 unchanged.
- Preregister one resumable VG024 corpus tagged
  `vq2_vg024_variable_gate_dagger_round5_vg022_visited_512`, seed `429109`:
  512 exact-uniform count-5--12 episodes, 1,024-step bound, at least 300,000
  records, and at least 95% Gate-1 reach. VG022 emits every deterministic plant
  action; the SF016 oracle labels only the newly visited higher-turn/lateral-
  recovery states.
- Crash/Gate-2/crossing-margin rates remain corpus diagnostics; layout/count/
  phase/action-history/label/action-wire-thrust/ordering predicates remain hard.
  Generic/wrapper/runner/test/preregistration hashes are `cf0b8f6d...`/
  `6b8063bb...`/`574ad9ac...`/`caf21cb6...`/`fb7875d4...`. If admitted, VG024
  authorizes only a source-balanced refit retaining all five prior anchors.
  Live authority remains zero.

### VQ2 VG024 admission / VG025 six-source refit — 2026-07-30

- VG024 admits `512,351` legal records from 512 exact-uniform count-5--12
  VG022-visited episodes in `270.226356 s`. Episode lengths are
  `309/1000.685547/1024` min/mean/max; phase records are
  `[135829,375514,1008,0,...]`, with `519` loader-audited increments.
- It reaches Gate 1 on `510/512` and Gate 2 on `12/512`; diagnostics are `28`
  low crashes, zero XY/high crashes, `9` misses, `475` horizon timeouts, and
  `7` crossing-margin violations. All 20 hard corpus predicates pass. Phase,
  ordering, action-history, action/wire/thrust envelope, non-finite query,
  teacher-emission, and FlightSim faults are zero.
- Report/metadata/state/archive/runner-log hashes are `5d36213f...`/
  `dfe64745...`/`3eaeae4d...`/`4b15dad4...`/`0c4442d5...`. Completed-resume
  exits zero and leaves state unchanged. The 148 MB archive is hash-exact
  locally; every 2.1 GB array hash, shape, dtype, record count, and terminal
  layout independently verifies. Admission evidence SHA is `3b31cb2e...`.
- Preregister VG025 tag `vq2_vg025_variable_gate_six_source_refit_001`: start
  from VG022 epoch 9, seed `429110`, six epochs, learning rate `5e-6`, 256-step
  BPTT, and four transition exposures. Every update contains clean/VG009/
  recovered-VG012/VG016/VG019/VG024 at exact weights
  `0.30/0.05/0.05/0.10/0.15/0.35`.
- Numerical admission requires balanced and VG024 improvement plus clean/prior
  validation caps `0.02/0.02/0.06/0.05/0.04` and VG024 cap `0.10`. Trainer/
  runner/test/preregistration hashes are `a3142bf3...`/`9dc71bb9...`/
  `b25f18c1...`/`b57da8f7...`; both native suites and `47/47` focused tests
  pass. VG025 is offline-only and grants no live authority.

### VQ2 VG025 numerical admission / VG026 accelerated screen — 2026-07-30

- VG025 completed all six source-locked epochs once in `5,275.614272 s` with
  `64,060` optimizer updates. Epoch 5 is selected at source-balanced validation
  MSE `0.01212692862693193`, improving the `0.028955003783597753` baseline by
  `2.387662x`. Its newly visited VG024 validation improves from `0.0582276441`
  to `0.0100127269` (`5.815363x`).
- Selected clean/VG009/recovered-VG012/VG016/VG019/VG024 values are
  `0.001472449/0.006805756/0.042591180/0.032214192/0.016596491/
  0.010012727`; all frozen caps pass. Every epoch proves exact six-source
  weights and `4.0x` transition exposure. The actor remains the unchanged
  4,119-input recurrent full-output Puffer policy with zero teacher blend.
- Checkpoint/report/completed-state/log/exit SHA-256 values are `85671c31...`/
  `b9d22d28...`/`f4a1b84f...`/`5d868cb5...`/`9a271f2a...`. Remote and local
  completed-output verifiers pass, both completed-resume checks return zero and
  leave the state hash unchanged, and every epoch boundary is mirrored locally.
  Admission evidence SHA-256 is `633a34d5...`.
- Preregister one fresh screen tagged
  `vq2_vg026_variable_gate_recurrent_teacher_free_256`: counts 5/8/11/12,
  64 terminal episodes each, fresh seeds `429111/429114/429117/429118`, at
  least `231/256` ordered finishes, and zero crash or hard transport/phase
  fault. Crossing margin remains diagnostic at the real `0.75 m` aperture.
- Before any admission episode, require a bounded 4-vs-32-thread parity probe:
  64 count-5 agents for exactly 128 steps and 8,192 actions per run, exact
  discrete fields, numeric error at most `1e-7`, and zero optimizer/state/
  teacher/FlightSim work. A local dry check with the frozen historical actor
  was exact; only the source-locked VG025-bound remote report is authoritative.
- Wrapper/parity-checker/runner/wrapper-test/parity-test/preregistration hashes
  are `e0df8de3...`/`48985cdf...`/`13771ba9...`/`720fcb6b...`/
  `7cb6c936...`/`155396ec...`. Both native suites, compilation/shell checks,
  and `40/40` focused tests pass. VG026 is offline-only; shadow, VQ2 Training,
  and Submission remain unauthorized.

### VQ2 VG026 rejection / VG027 long-horizon DAgger — 2026-07-30

- VG026 is terminally rejected without unchanged retry. It records `0/256`
  finishes, `168` crashes (`72` low and `96` lateral/XY), `88` misses, and no
  timeout. Gate-1/Gate-2/Gate-3/Gate-4 reach is `256/97/2/0`; count-5/8/11/12
  Gate-2 reach is `30/22/22/23` of 64. This improves VG023 Gate-2 reach by 71
  and crashes by 25, but remains far outside deployment safety.
- Transport and policy execution are exact: maximum action error, public-phase
  off-tick/decrease/skip, raw phase error, ordering, non-finite, action/wire/
  thrust envelope, teacher-action, and optimizer faults are all zero. Similar
  reach across course counts localizes the failure to the post-Gate-1 turn and
  recovery rather than total course length.
- The source-locked 4-vs-32-thread parity probe is numerically exact, but the
  32-thread wall-time speedup is only `1.019054x`. The 51,987-step screen takes
  `10,815.929732 s` at `4.806522` vector steps/s, so this Vast host's native
  rollout remains CPU-bound despite additional OpenMP threads.
- Aggregate/state/bootstrap-log/exit hashes are `324634e2...`/`de296580...`/
  `cb5ef584...`/`53c234e5...`; completed-resume returns `2` and leaves state
  unchanged. Rejection evidence SHA is `866e240c...`.
- Preregister one VG025-visited collection tagged
  `vq2_vg027_variable_gate_dagger_round6_vg025_visited_512`, seed `429119`:
  512 exact-uniform count-5--12 episodes, a 4,096-step bound, at least one
  million labels, `>=95%` Gate-1 reach, `>=20%` Gate-2 reach, and at least one
  stored phase-2 record. VG025 supplies every deterministic plant action;
  SF016 supplies training labels only.
- Wrapper/runner/test/preregistration hashes are `cc630062...`/`1094f89f...`/
  `b2e8d62d...`/`adcfa165...`. Focused and adjacent tests pass `42/42`, both
  native suites pass, and compilation/shell syntax pass. VG027 may authorize
  only a separately preregistered source-balanced refit retaining all prior
  anchors. FlightSim, shadow, Training, and Submission remain unauthorized.

### VQ2 VG027 admission / VG028 seven-source refit — 2026-07-31

- VG027 completed its sole seed-`429119` collection from source commit
  `8045af9b...` without resume. It admits `1,541,730` legal records from 512
  exact-uniform count-5--12 episodes in `1,099.075820 s`; episode length
  min/mean/max is `564/3011.19140625/4096`.
- Ordered Gate-1/Gate-2/Gate-3/Gate-4 reach is `512/197/2/0`. The corpus
  contains `368,731` phase-2 records and `711` exact phase increments.
  Crashes/misses/timeouts are `154/137/221` training-distribution diagnostics.
  All 22 hard layout, phase, ordering, finite-action, envelope, and transport
  predicates pass; teacher actions, student updates, FlightSim packets, and
  sealed-test accesses are zero.
- Report/metadata/state/archive/admission hashes are `08c78d66...`/
  `cf1e9ba2...`/`1bc0c0ac...`/`c078a2df...`/`de2437f3...`. Completed-resume
  exits zero and leaves output unchanged; the local archive, arrays, hashes,
  shapes, dtypes, record count, and terminal layout verify exactly.
- The 48-hour competitive-lap sprint deadline is
  `2026-08-02T00:22:45-04:00`. Its authoritative goal prompt SHA-256 is
  `03f085d3...`. It supersedes older Submission authority: VQ2 Submission is
  forbidden without a new explicit user instruction.
- Preregister exactly one VG028 seven-source recurrent refit from admitted
  VG025 epoch 5, seed `429120`, six epochs, eight-agent source chunks,
  256-step recurrence, AdamW `5e-6`, and four transition exposures. Every
  update uses clean/VG009/recovered-VG012/VG016/VG019/VG024/VG027 at exact
  weights `0.25/0.04/0.04/0.08/0.12/0.17/0.30`.
- Numerical admission requires overall and VG027 validation improvement plus
  preservation caps `0.02/0.02/0.06/0.05/0.04/0.04/0.10`, finite metrics,
  exact source weights, and `4.0x` transition exposure. Trainer/runner/test/
  preregistration hashes are `ceeddef8...`/`3584a80c...`/`77ae61f0...`/
  `ce95bc2d...`. VG028 authorizes only a separately preregistered staged
  teacher-free screen; shadow, Training, and Submission authority remain zero.

### VQ2 VG028 admission / VG029 paired diagnostic — 2026-07-31

- VG028 completed its sole seed-`429120` six-epoch refit from source commit
  `b3403a3e...` in `3,012.894098 s` and `34,449` optimizer updates. Epoch 6 is
  selected at fixed seven-source validation MSE `0.030373104180`, improving
  the `0.108328904468` parent baseline by `3.566606x`.
- Newly visited VG027 validation improves from `0.332380123454` to
  `0.070030132050` (`4.746244x`). Selected clean/VG009/recovered-VG012/VG016/
  VG019/VG024 values are `0.002798320/0.007910041/0.036716251/0.035400363/
  0.016201518/0.012371892`; every frozen cap passes. All six epochs prove exact
  source weights and `4.0x` transition exposure.
- Checkpoint/report/completed-state/runner-log/exit SHA-256 values are
  `ec1206b5...`/`b51b7a9b...`/`096c7939...`/`5c76eb7f...`/`9a271f2a...`.
  Remote completed resume returns zero and leaves all three artifacts
  unchanged; the checkpoint is tensor-exact to the saved selected-best state,
  all report values are finite, and every epoch boundary is preserved locally.
  Admission evidence SHA-256 is `3282c98f...`.
- Bootstrap 001 failed before tests, output creation, or optimizer work because
  the new entrypoints lacked executable Git modes. Repair commit `b3403a3e...`
  changes modes only; bootstrap 002 is the sole training trajectory. Failure
  log/exit hashes are `5c7c4924...`/`703d2c10...`.
- VG029 preregisters a same-seed paired count-5 diagnostic: 32 episodes each
  for VG025 and VG028, seed `429121`, four threads, `2,560` maximum steps,
  concurrent execution, deterministic complete recurrent outputs, and zero
  teacher action. Candidate promotion requires hard transport parity, no
  Gate-1 or crash regression, at least one Gate-3-or-finish signal, and strict
  downstream improvement over VG025.
- Component/comparator/runner/test/preregistration hashes are `f6ae14ae...`/
  `42e9a4d7...`/`0bb86a7e...`/`7048db0c...`/`e308a618...`; parent/candidate
  manifest hashes are `2a5d5dc1...`/`473bc93b...`. VG029 can authorize only a
  separately preregistered full offline screen. Shadow, VQ2 Training, and
  Submission remain unauthorized.

### VQ2 VG029 admission / VG030 count-11 diagnostic — 2026-07-31

- VG029 completes once from source commit `02b7c773...` on 32 identical fresh
  count-5 courses per actor, seed `429121`, four threads, and a 2,560-step
  horizon. VG028 versus VG025 Gate-1/Gate-2/Gate-3/Gate-4 reach is
  `32/25/3/1` versus `32/13/0/0`; mean ordered gates improves
  `1.40625 -> 1.90625`, while crashes improve `5 -> 2`.
- Both actors have zero teacher, optimizer, FlightSim, action-envelope,
  phase, ordering, or transport fault. Aggregate/parent/candidate/state/log
  hashes are `5bbc44b9...`/`c6e06a20...`/`0caecd37...`/
  `bb25b3f0...`,`5b59b6e5...`/`a40fb507...`; completed resume exits zero and
  leaves every output unchanged. Admission evidence SHA is `42ac6e99...`.
- Before spending the full 256-course screen, preregister VG030: a same-seed
  paired count-11 diagnostic on 32 episodes per actor, seed `429122`, four
  threads, and 2,560 steps. Retain the VG029 gates: zero hard fault, no
  Gate-1/crash regression, nonzero Gate-3-or-finish signal, and strict
  downstream improvement over VG025.
- Generalized component/comparator/runner/test/preregistration hashes are
  `925ee589...`/`53d30a1e...`/`8a4b19ed...`/`b337cbea...`/`b0c5559f...`;
  parent/candidate manifest hashes are `929f1063...`/`d89d641b...`. VG030 is
  offline-only. Shadow, VQ2 Training, and Submission remain unauthorized.

### VQ2 VG030 admission / VG031 full screen — 2026-07-31

- VG030 completes once from source commit `c263c735...` on 32 identical fresh
  count-11 courses per actor, seed `429122`, four threads, and 2,560 steps.
  VG028 versus VG025 Gate-1/Gate-2/Gate-3 reach is `32/25/4` versus
  `32/7/0`; mean ordered gates improves `1.21875 -> 1.90625`. Both have two
  crashes, while VG028 misses/timeouts are `6/24` versus `14/16`.
- Both actors pass every policy, phase, ordering, action-history, envelope,
  and transport predicate. Aggregate/parent/candidate/state/runner hashes are
  `9c065836...`/`7783e697...`/`5af2b623...`/
  `64b3b0b5...`,`be910868...`/`bfe5d41b...`; completed resume exits zero and
  changes no terminal artifact. Admission evidence SHA is `d44cacc1...`.
- Preregister exactly one VG031 full screen of VG028: 64 terminal episodes
  each at counts 5/8/11/12, fresh seeds `429123/429126/429129/429130`, four
  native threads, and the existing terminal horizon. Require at least
  `231/256` ordered finishes and zero crash or hard policy/phase/action fault.
- Wrapper/runner/test/preregistration hashes are `2930c82f...`/
  `ff27bca2...`/`e59fd9d3...`/`62f15082...`. VG031 is offline-only and grants
  no shadow, VQ2 Training, or Submission authority.

### VQ2 VG031 early rejection / VG032 higher-phase DAgger — 2026-07-31

- VG031 is terminally rejected and must never resume or retry unchanged. Its
  completed count-5 prefix records `5/64` full finishes, `27` crashes
  (`24` low, `3` lateral), `30` misses, and `2` timeouts. Gate-1 through
  Gate-5 reach is `64/54/18/16/5`; mean gates passed is `2.453125`.
- Early stop is mathematically terminal: even 192 perfect remaining episodes
  could produce at most `197/256` finishes, below the preregistered
  `231/256` floor, and zero-crash admission was already impossible. The
  screen consumed `4,804.816625 s`; stopping preserved several GPU-hours.
- Action history, public phase, ordering, policy envelopes, and transport are
  exact. Count report/state/runner-log/exit hashes are `b15de9cd...`/
  `371a1803...`/`8bdf0801...`/`9d9b1872...`. Source-bound early-rejection
  evidence SHA-256 is `aed9cc06...`.
- Preregister exactly one VG032 VG028-visited collection, seed `429131`: 512
  uniform count-5--12 one-shot episodes, 4,096-step bound, at least one
  million labels, `>=95%` Gate-1, `>=50%` Gate-2, and `>=10%` Gate-3 reach,
  plus nonzero phase-2 and phase-3 records. VG028 emits every deterministic
  recurrent plant action; SF016 supplies stored labels only.
- Manifest/collector/runner/test/preregistration hashes are `d6da45a8...`/
  `0428813e...`/`e0278410...`/`b644e910...`/`84150960...`. VG032 may
  authorize only a separately preregistered all-anchor recurrent refit.
  FlightSim, shadow, Training, and Submission remain unauthorized.

### VQ2 VG032 admission / VG033 eight-source refit — 2026-07-31

- VG032 completed once from source commit `04ee2edb...` without resume. It
  admits `1,636,103` legal records from 512 exact-uniform count-5--12
  episodes in `1,094.757103 s`; episode length min/mean/max is
  `564/3195.513672/4096`.
- Ordered Gate-1/Gate-2/Gate-3/Gate-4 reach is `512/438/67/19`. Phase
  2/3/4 contributes `734,846/67,253/13,244` records. All 24 hard corpus
  predicates pass; crash/miss/timeout `59/214/239` remain training-
  distribution diagnostics.
- Report/metadata/state/bootstrap/completed-resume-exit hashes are
  `f725afd9...`/`b4d43848...`/`b9a1a9fb...`/`c7db59e3...`/`9a271f2a...`.
  Completed resume is byte-identical to the report and leaves state unchanged.
  Small evidence is hash-exact locally; retained arrays remain on the worker.
  Admission SHA-256 is `7887a69a...`.
- Preregister exactly one VG033 six-epoch recurrent refit from VG028 epoch 6,
  seed `429132`, eight-agent source chunks, 256-step BPTT, AdamW `5e-6`, and
  four transition exposures. Every update uses clean/VG009/recovered-VG012/
  VG016/VG019/VG024/VG027/VG032 at weights
  `0.25/0.03/0.03/0.06/0.08/0.08/0.12/0.35`.
- Admission requires strict balanced and VG032 validation improvement, exact
  weights, `4.0x` transition exposure, finite metrics, and per-source caps
  `0.02/0.02/0.06/0.05/0.04/0.04/0.10/0.12`. Core/trainer/runner/core-test/
  trainer-test/preregistration hashes are `c76902c8...`/`cb2f4287...`/
  `3e85240b...`/`065f147c...`/`335a5e10...`/`bc013468...`. VG033 can
  authorize only a fresh paired offline diagnostic; no live authority exists.

### VQ2 VG033 admission / VG034 paired count-5 diagnostic — 2026-07-31

- VG033 completed its sole seed-`429132` six-epoch refit from source commit
  `0f76d945...` in `3,407.805697 s` and `35,070` optimizer updates. Epoch 6
  is selected at balanced validation MSE `0.029448763167`, improving the
  VG028 baseline `0.057872192702` by `1.965182x`.
- VG032 validation improves `0.122915181429 -> 0.042820791151`
  (`2.870456x`). Selected clean/VG009/recovered-VG012/VG016/VG019/VG024/
  VG027 values are `0.004864186/0.007957505/0.030903256/0.038210876/
  0.017964366/0.014296200/0.060050992`; every cap passes. All six epochs
  prove exact source weights and `4.0x` transition exposure.
- Checkpoint/report/completed-state/runner-log/exit hashes are
  `56a8e3b8...`/`033a7f1f...`/`b7bd08d7...`/`aa993727...`/`9a271f2a...`.
  Completed resume is byte-identical to the report and state-immutable;
  local checkpoint tensors exactly equal the saved epoch-6 best state.
  Admission evidence SHA-256 is `dbf0fbb4...`.
- Preregister VG034: one concurrent same-seed paired count-5 diagnostic of
  VG028 versus VG033, 32 episodes each, seed `429133`, four threads, and
  2,560 steps. Qualify only with zero hard fault, no Gate-1/crash regression,
  nonzero Gate-3-or-finish signal, and strict downstream improvement.
- Parent/candidate manifest, runner, dedicated test, and preregistration
  hashes are `601a1706...`/`3dc5a6d7...`/`a941ec38...`/`13f33e5b...`/
  `65195bab...`. VG034 is offline-only; no shadow, Training, or Submission
  authority is granted.

### VQ2 VG034 admission / VG035 paired count-11 diagnostic — 2026-07-31

- VG034 completed once from source commit `0ac081df...` on 32 identical fresh
  count-5 courses per actor, seed `429133`, four threads, and 2,560 steps.
  VG033 versus VG028 Gate-1/Gate-2/Gate-3 reach is `32/31/8` versus
  `32/25/3`; mean gates improves `1.90625 -> 2.21875` and crashes improve
  `2 -> 1`.
- Both actors pass every deterministic action, history, public-phase,
  ordering, envelope, and transport predicate. Aggregate/parent/candidate/
  parent-state/candidate-state hashes are `8c6202d2...`/`a65ef909...`/
  `b4be8478...`/`5aa29e80...`/`56f8f063...`; completed resume is
  byte-identical. Admission evidence SHA is `858adb9e...`.
- Preregister VG035: 32 same-seed fresh count-11 episodes per actor, seed
  `429134`, four threads, and 2,560 steps. Retain the VG034 gates: no
  Gate-1/crash regression, nonzero Gate-3-or-finish signal, strict downstream
  improvement, and zero hard fault.
- Parent/candidate manifest, runner, dedicated test, and preregistration
  hashes are `de4dcb1f...`/`479cdc9c...`/`c9a17edf...`/`d4e22f00...`/
  `c7182d10...`. VG035 may authorize only a fresh full offline screen; live
  authority remains zero.

### VQ2 VG035 admission / VG036 full screen — 2026-07-31

- VG035 completed once from source commit `dca257f1...` on 32 identical fresh
  count-11 courses per actor, seed `429134`, four threads, and 2,560 steps.
  VG033 versus VG028 Gate-1/Gate-2/Gate-3 reach is `32/29/5` versus
  `32/25/4`; mean ordered gates improves `1.90625 -> 2.0625`, and crashes
  improve `2 -> 0`.
- Both actors pass every deterministic action, history, public-phase,
  ordering, envelope, and transport predicate. Aggregate/parent/candidate/
  parent-state/candidate-state hashes are `6554d6dd...`/`196e39b3...`/
  `ad09077f...`/`e5ac67ae...`/`370b728a...`; completed resume is
  byte-identical. Admission evidence SHA is `7c7676b7...`.
- Preregister exactly one VG036 full screen of VG033: 64 terminal episodes
  each at counts 5/8/11/12, fresh seeds `429135/429138/429141/429142`, four
  native threads, and the existing terminal horizon. Require at least
  `231/256` ordered finishes and zero crash or hard policy/phase/action fault.
  Stop after a completed count if admission is mathematically impossible.
- Wrapper/runner/test/preregistration hashes are `0cddd9b1...`/
  `eb9217be...`/`4c8209cc...`/`2e3e49fe...`. VG036 is offline-only and
  grants no shadow, VQ2 Training, or Submission authority.

### VQ2 VG036 infrastructure abort / VG037 bounded-thread screen — 2026-07-31

- VG036 was stopped before its first count completed and has no policy result.
  Its final runner command omitted the `OMP_NUM_THREADS=4`/
  `MKL_NUM_THREADS=1` limits used by both fast paired diagnostics, creating a
  machine-wide PyTorch CPU pool and reproducing VG031's multi-thousand-second
  count behavior. Do not resume or retry VG036 unchanged.
- Both native suites, the SM89 float32 build, and `46/46` focused tests passed.
  No count or aggregate report was written; only source-bound state and log
  artifacts exist, SHA-256 `9e98f004...`/`e0c33fe3...`. Abort evidence SHA is
  `c5dc3287...`; it has no policy conclusion and sent zero FlightSim packets.
- Preregister VG037 on fresh seeds `429143/429146/429149/429150`. It retains
  VG036's 64 episodes at each count 5/8/11/12, four native vector threads,
  2,560 steps, `231/256` finish floor, and zero-crash/hard-fault gate, while
  binding both global thread limits before every evaluator process.
- Wrapper/runner/test/preregistration hashes are `5c6f9278...`/
  `f86f65a4...`/`d75ecf54...`/`1eef10b5...`. VG037 remains offline-only;
  shadow, Training, and Submission authority are zero.

### VQ2 VG037 rejection / VG038 higher-phase DAgger — 2026-07-31

- VG037 is terminally rejected and must never resume or retry unchanged. Its
  completed fresh count-5/count-8 prefix has `14/128` full finishes,
  `45` crashes (`43` low, `2` lateral), `47` misses, and `22` timeouts.
  Ordered Gate-1 through Gate-8 reach is
  `128/126/85/71/31/16/6/3`; mean ordered gates is `3.640625`.
- Even 128 perfect remaining episodes could produce at most `142/256`
  finishes, below the fixed `231/256` floor; zero-crash admission is also
  impossible. Action history, public phase, ordering, policy envelopes, and
  transport are exact. Count-5/count-8/state/log/evidence hashes are
  `affe2363...`/`ac92ef6b...`/`3d454acf...`/`85de99e7...`/`bb14e1e1...`.
- Preregister exactly one VG038 VG033-visited collection, seed `429151`: 512
  uniform count-5--12 one-shot episodes, 4,096-step bound, at least one
  million labels, Gate-1/2/3/4 reach floors `95%/90%/40%/20%`, and nonzero
  phase-2/3/4/5 records. VG033 emits every deterministic recurrent plant
  action; SF016 supplies stored labels only.
- Manifest/collector/runner/test/preregistration hashes are `63012cd2...`/
  `c8b3b9ee...`/`ab184a7d...`/`8015f08e...`/`8cb8d18b...`. VG038 may
  authorize only a separately preregistered all-anchor recurrent refit.
  FlightSim, shadow, Training, and Submission remain unauthorized.

### VQ2 VG038 floor rejection / VG039 calibrated collection — 2026-07-31

- VG038 is terminally rejected and its arrays are quarantined; never resume,
  retry, or fit them unchanged. It completed 512 fresh uniform count-5--12
  episodes in `81.327282 s` and produced `1,922,029` legal labels. Every
  layout, phase, action-history, query, ordering, envelope, and transport
  predicate passed.
- Phase-2/3/4/5 record counts are
  `1,061,932/139,254/28,334/272`. Only the preregistered Gate-3/Gate-4 reach
  floors failed: observed `26.5625%/10.546875%` versus `40%/20%`. Rejection/
  state/log hashes are `7f91cb84...`/`876fea58...`/`22ed7d79...`.
- Preregister VG039 on fresh seed `429152` with no VG038 array reuse. Retain
  512 uniform one-shot episodes, 4,096 steps, one million labels, and nonzero
  phase-2--5 records; calibrate Gate-1/2/3/4 floors to
  `95%/90%/20%/5%`, all below the source-locked VG038 observations.
- Manifest/collector/runner/test/preregistration hashes are `900db1bd...`/
  `4eb20af4...`/`eadb9181...`/`d2c12dbf...`/`868ac0ac...`. VG039 is
  offline-only and may authorize only a separately preregistered refit.

### VQ2 VG039 admission / VG040 nine-source refit — 2026-07-31

- VG039 completed once from source commit `bb53f682...` in `99.634472 s`,
  admitting `1,901,389` fresh legal labels from 512 exact-uniform count-5--12
  episodes. Episode length min/mean/max is `582/3713.650391/4096`.
- Ordered Gate-1/2/3/4/5 reach is `512/487/146/45/3`; phase-2/3/4/5 has
  `1,022,122/158,365/26,044/240` records. All 27 hard corpus predicates
  pass. Report/metadata/state/log hashes are `a0bdcd0d...`/`89134752...`/
  `41702ed8...`/`40017203...`; completed resume is report-exact and leaves
  state unchanged. Admission evidence SHA is `e01eb2e2...`.
- Preregister VG040: one six-epoch recurrent refit from VG033 epoch 6, seed
  `429153`, nine-agent source chunks, 256-step BPTT, AdamW `5e-6`, and four
  transition exposures. Every update uses clean/VG009/recovered-VG012/VG016/
  VG019/VG024/VG027/VG032/VG039 at exact weights
  `0.22/0.02/0.02/0.04/0.05/0.05/0.08/0.12/0.40`.
- Admission requires strict nine-source and VG039 validation improvement,
  exact weights/exposure, finite metrics, and per-source caps
  `0.02/0.02/0.06/0.05/0.04/0.04/0.10/0.12/0.12`. Trainer/runner/test/
  preregistration hashes are `22682e75...`/`386e660b...`/`15470bc9...`/
  `f7993252...`. VG040 remains offline-only.

### VQ2 VG040 admission / VG041 paired count-5 diagnostic — 2026-07-31

- VG040 completed its sole six-epoch seed-`429153` refit from source commit
  `faa7e1b2...` in `3,698.851735 s` and `35,061` optimizer updates. Epoch 3
  is selected at balanced validation MSE `0.028067853488`, improving the
  nine-source VG033 baseline `0.052488560618` by `1.870060x`.
- VG039 actor-visited validation improves
  `0.093892966977 -> 0.033389153715` (`2.812080x`). Every nine-source cap,
  source-weight audit, and `4.0x` transition-exposure predicate passes.
  Checkpoint/report/state/runner-log hashes are `c2a015f3...`/`0f380319...`/
  `22387d91...`/`26d73809...`; completed resume is byte-identical to the
  report and leaves state unchanged. Admission SHA-256 is `53110ead...`.
- Preregister exactly one VG041 concurrent paired count-5 diagnostic of
  VG033 versus VG040: 32 episodes per actor, fresh seed `429154`, four
  threads, and 2,560 steps. Qualify only with zero hard fault, no Gate-1 or
  crash regression, nonzero Gate-3-or-finish signal, and strict downstream
  improvement.
- Parent/candidate manifest, runner, dedicated test, and preregistration
  hashes are `51364059...`/`b2169bc6...`/`190ae655...`/`832ad024...`/
  `4ce5e294...`. VG041 is offline-only; no shadow, Training, or Submission
  authority exists.

### VQ2 VG041 rejection / phase-balanced repair — 2026-07-31

- VG041 completed once on fresh seed `429154` after its preflight correctly
  stopped before rollout because the runner executable bit was absent. Commit
  `3913f47e...` repaired only that metadata; runner content SHA remained
  `190ae655...`, and all 37 remote tests then passed.
- VG040 is rejected for rollout and VG033 remains the parent. VG040 versus
  VG033 Gate-1/2/3 reach is `32/31/6` versus `32/31/8`; crashes regress
  `1 -> 3`, misses regress `0 -> 2`, and mean gates regress
  `2.21875 -> 2.15625`. Both actors retain exact action/history/phase/
  ordering/envelope/transport behavior.
- Aggregate/parent/candidate/state hashes are `38526e26...`/`cc66c27a...`/
  `6808a164...`/`4c895c6a...`,`efbfe244...`; completed resume exits with
  the expected rejection code and is byte-identical to the aggregate.
  Rejection evidence is source-locked; never retry VG041 unchanged.
- Do not collect another broad corpus yet. VG039 already contains
  `158,365/26,044/240` legal phase-3/4/5 records, but whole-corpus sampling
  dilutes them behind `1,022,122` phase-2 records. The next offline branch
  must retain VG033 and explicitly balance later public phases inside VG039.
  No live authority exists.

### VQ2 VG042 phase-balanced refit — 2026-07-31

- Preregister one four-epoch refit from VG033 on fresh seed `429156`,
  AdamW `2e-6`, 256-step causal BPTT, and `4.0x` transition exposure. Keep
  every original VG039 recurrent sequence and apply only public-phase row
  weights `0.25/0.5/1/6/32/32` for phases `0/1/2/3/4/5+`.
- The phase weights change effective VG039 training mass to approximately
  `1.01%/9.27%/32.71%/31.90%/24.83%/0.28%` across phases 0--5. They never
  enter the actor observation or runtime. Outer clean/VG009/recovered-VG012/
  VG016/VG019/VG024/VG027/VG032/VG039 weights are fixed at
  `0.25/0.03/0.03/0.06/0.08/0.08/0.12/0.15/0.20`.
- Admit only with strict balanced/VG039 improvement, no phase-3 regression,
  strict phase-4 improvement, ordinary VG039 MSE at most `0.12`, all nine
  preservation caps, exact source weights, and finite source-bound evidence.
  Numerical admission may authorize only a fresh paired diagnostic.
- Core/nine-source/trainer/runner/test/preregistration hashes are
  `6b0ff918...`/`c959096b...`/`4959f5c2...`/`d9ec5e93...`/`1b81716e...`;
  the repaired preregistration binds the preflight abort. VG042 is
  offline-only; FlightSim, shadow, Training, and Submission remain
  unauthorized.
- The first VG042 preflight on commit `b4251d80...` passed both native suites,
  the SM89 build, and `71/71` tests, then stopped before output/state creation
  because its evaluator referenced `_agent_batches` through the wrong module.
  No optimizer update occurred. Abort evidence SHA is `a66277e6...`; the
  repaired trainer imports the helper from its defining module and directly
  tests that path. Repaired preregistration SHA is `06be595b...`.

### VQ2 VG042 admission / VG043 paired count-5 diagnostic — 2026-07-31

- Repaired VG042 completed its sole four-epoch seed-`429156` refit from
  source commit `b597824d...` in `2,506.644261 s` and `23,343` updates,
  selecting epoch 2. Balanced validation improves
  `0.042006633060 -> 0.032940475596`.
- Phase-balanced VG039 validation improves
  `0.105610140617 -> 0.062242848395`; phase 3 improves
  `0.047932099463 -> 0.033795586811`, phase 4 improves
  `0.123386528242 -> 0.088277508252`, and ordinary VG039 improves
  `0.093892966975 -> 0.043845329419`. All nine caps and hard audits pass.
- Checkpoint/report/state/repaired-log hashes are `876b0ecc...`/
  `2f4c1c9b...`/`3925f815...`/`0a6a2e81...`; completed resume is
  report-exact and state-immutable. Admission SHA is `4b18462a...`.
- Preregister VG043: one concurrent paired count-5 diagnostic of VG033
  versus VG042, 32 episodes each, fresh seed `429157`, four threads, and
  2,560 steps. Require no Gate-1/crash regression, nonzero Gate-3-or-finish
  signal, strict downstream improvement, and zero hard fault.
- Parent/candidate manifest, runner, test, and preregistration hashes are
  `ecb6dc36...`/`c74dd150...`/`197006cd...`/`931631e2...`/`d6250ba7...`.
  VG043 is offline-only; no shadow, Training, or Submission authority exists.

### VQ2 VG043 rejection / update-fraction bracket — 2026-07-31

- VG043 completed once on fresh seed `429157`. VG042 is rejected for rollout
  and VG033 remains the parent: Gate-1/2/3 reach regresses
  `32/31/8 -> 32/28/2`, crashes regress `1 -> 2`, misses regress `0 -> 1`,
  and mean gates regress `2.21875 -> 1.9375`.
- Both actors retain exact deterministic action, history, public-phase,
  ordering, envelope, and transport contracts. Aggregate/parent/candidate/
  state hashes are `074c44e3...`/`d2702ecb...`/`0ddb23d4...`/
  `098be195...`,`410d3c6f...`; completed resume is aggregate-exact. Never
  retry VG043 unchanged.
- The phase-balanced update direction has useful held-out label evidence but
  its full magnitude crosses the rollout stability boundary. Before any new
  gradient training or collection, preregister a small checkpoint
  interpolation bracket from VG033 toward VG042, then require a separate
  fresh-seed confirmation of any selected fraction. No live authority exists.
- VG044 source-locks that bracket at fractions
  `0/0.01/0.025/0.05/0.10/0.20/0.35/0.50`, 64 shared seed-`429158`
  count-5 episodes per fraction, four threads, and 2,560 steps. Gate-1/2
  reach and crashes may not regress; Gate-3 reach or finishes must strictly
  improve. A qualifying fraction remains offline-only until a fresh paired
  confirmation.
- Evaluator/runner/test/preregistration SHA-256 values are
  `e1429cd9...`/`71866d22...`/`5cc56347...`/`5e179943...`.
- VG044 completes all eight fractions and selects alpha `0.10`. On 64
  seed-`429158` episodes, VG033 versus selected Gate-1/2/3 reach is
  `64/62/11 -> 64/63/13`; crashes improve `5 -> 4`, misses `6 -> 3`, and
  mean gates `2.140625 -> 2.1875`. Every hard predicate passes.
- Selected checkpoint/report/state SHA-256 values are `d7ba25a6...`/
  `c6fb4544...`/`c8158c26...`; completed resume is immutable and exact.
  Admission SHA is `85f1baf2...`.
- Preregister VG045 as the sole fresh confirmation: VG033 versus VG044,
  64 concurrent seed-`429159` count-5 episodes each, four threads, 2,560
  steps. Require no Gate-1/2/crash regression and strict Gate-3-or-finish
  improvement. Comparator/runner/test/preregistration hashes are
  `12657474...`/`862e2418...`/`a595f295...`/`5a28bc64...`. No live
  authority exists.
- VG045's formal metrics repeat VG044 exactly, including every aggregate
  metric and action statistic for both actors. Canonical behavior projection
  hashes are identical despite declared seed `429158 -> 429159`: parent
  `d0db42b0...`, candidate `fc25942e...`. `src/vecenv.h` initializes the
  drone-race RNG from environment index, so the training seed is inert here.
  Reject VG045 as non-independent; aggregate/rejection hashes are
  `f386b02d...`/`214c81c1...`.
- The default-preserving repair is VG046: apply native
  `evaluation_episode_offset=8` to both actors, which advances eight resets
  before the measured episode. Require both behavior projections to differ
  from offset zero plus every VG045 rollout criterion. Component/comparator/
  runner/test/preregistration hashes are `85f7e2a5...`/`b5bd643d...`/
  `d9d092b7...`/`ca0649fb...`/`e5a14922...`. No live authority exists.
- VG046 is independent and rejects alpha `0.10` on safety. VG033 versus the
  candidate reaches Gate-1/2/3 `64/58/7 -> 64/59/10`, but crashes regress
  `6 -> 7`; both offset and behavior-distinctness checks pass. Aggregate/
  rejection hashes are `7d95661c...`/`3049a97a...`.
- VG047 brackets smaller alphas
  `0/0.025/0.04/0.05/0.06/0.075/0.085/0.10` on the same offset-8 fixture,
  requiring no Gate-1/2/crash regression and strict Gate-3-or-finish gain.
  First preflight stopped before state after `36/36` tests due missing direct
  import bootstrap; abort SHA is `a918fcb4...`. Repaired wrapper/runner/test/
  preregistration hashes are `0f81b5bb...`/`2449e2c8...`/`425528c5...`/
  `5d718ea0...`. A selection remains
  offline-only pending a different-offset confirmation.
- Repaired VG047 selects alpha `0.025`. On offset 8, Gate-1/2/3 reach is
  `64/58/7 -> 64/58/10`, crashes improve `6 -> 5`, and all hard predicates
  pass. Checkpoint/report/state hashes are `d0bc631e...`/`a27c8384...`/
  `55d330af...`; admission SHA is `3204cee2...`.
- VG048 is the different-fixture confirmation at episode offset `16`, 64
  paired episodes, seed label `429162`, requiring behavior distinctness plus
  Gate-1/2/crash preservation and strict downstream gain. Wrapper/runner/
  test/preregistration hashes are `8c7ce186...`/`347d7a75...`/
  `9481bf5c...`/`84f08384...`. No live authority exists.
- VG048 rejects alpha `0.025`: Gate-3 reach regresses `10 -> 9` and crashes
  `8 -> 11`, though it produces the first Gate-4 reach (`1/64`). Aggregate/
  rejection hashes are `d4b5de1e...`/`a8f5b91f...`.
- VG049 brackets micro fractions through `0.025` on offset 16. Admission
  permits strict Gate-4 gain as downstream progress but still requires
  Gate-1/2 and crash preservation. Wrapper/runner/test/preregistration hashes
  are `5064814b...`/`8ca229f3...`/`74f45ad1...`/`711f7cfa...`. No live
  authority exists.
- VG049 selects alpha `0.0025`: Gate-1/2/3 reach changes
  `64/56/10 -> 64/57/11`, crashes remain `8`, and misses improve `7 -> 5`.
  Checkpoint/report/state hashes are `35b18c0d...`/`8af8ef20...`/
  `ca057d04...`; admission SHA is `f6f81e89...`.
- VG050 confirms on offset 24 with 64 paired episodes. Comparator/runner/test/
  preregistration hashes are `d393bd4f...`/`cec1c76c...`/`7db6a0ee...`/
  `b6c1d414...`. No live authority exists.
- VG050 rejects the interpolation line: Gate-2 reach regresses `61 -> 60`
  and crashes `1 -> 3`. Aggregate/rejection hashes are `86c56e84...`/
  `0cd7e823...`; retain VG033.
- VG051 redirects evaluation to the actual six-gate proxy across episode
  offsets `0/8/16/24`, 64 episodes each. Evaluator/runner/test/
  preregistration hashes are `cd5e7a88...`/`ee7b7e83...`/`e2fe240f...`/
  `cda2963c...`. No live authority exists.
- VG051 completes from exact source commit `62d0ae13...`; the source-locked
  completed resume passes and the local copy matches every remote hash. Across
  256 episodes, VG033 reaches Gate 1--6 at `256/241/52/5/0/0`, with zero
  finishes, `17` crashes, `22` misses, and `2.1640625` mean gates. Every hard
  transport predicate passes. Report/state/runner-log hashes are
  `43da7e9b...`/`0937a715...`/`9c55178d...`; the completed-resume output is
  report-exact. Evidence SHA is recorded separately.
- This establishes Gate 4 -> Gate 5 as the active six-gate bottleneck. Stop
  checkpoint interpolation: VG042's direction is not robust even at a
  `0.0025` fraction. Retain VG033 and next collect training-only late-gate
  local-start states with SF016 oracle query labels while executing only the
  Puffer actor. Preserve the 4,119-value legal runtime ABI and require a
  multi-offset six-gate rollout improvement before any live authority.
- VG052 is that source-locked collection. A default-off native extension adds
  a gate-local sampling range `[min,max_exclusive)`; its nonpositive maximum
  preserves the historical full-range RNG call exactly. Both native regression
  suites pass. VG052 fixes six gates, 512 one-shot agents, 2,048 steps, active
  gate indices `[3,5)`, and offsets `[2,5] m`. VG033 alone emits plant actions;
  SF016 labels are stored only. Require at least 250,000 legal records and
  50,000 each at public phases 3/4 with zero early-phase rows and every hard
  parity/progress/envelope predicate.
- Manifest/collector/runner/test/preregistration hashes are
  `7e246d5e...`/`a3e12b9d...`/`9e84b4fd...`/`72021570...`/`7958566e...`.
  The native C/header/binding hashes are `246997f5...`/`6c1093b1...`/
  `65c1e297...`. VG052 is offline-only and may authorize only a separately
  preregistered trust-region refit; no live authority exists.
- VG052 completes once from commit `e9f9ca24...` in `45.352140 s` and admits
  `548,255` legal rows: phases 3/4/5 contain
  `63,023/255,090/230,142`. All 512 resets use the requested local range;
  all 21 hard predicates pass, action-history error is zero, and teacher
  actions remain zero. The Puffer actor finishes 95 local suffix episodes,
  while `89/217/111` crash/miss/timeout outcomes characterize the training
  distribution and are not deployment admission. Report/metadata/state/log
  hashes are `687141ac...`/`03b71103...`/`5cf79b0b...`/`b5b41f0f...`;
  completed resume is report-exact. Admission SHA is `95c3f6e6...`.
- VG053 preregisters one anchor-heavy trust-region refit from VG033: one epoch,
  AdamW `5e-7`, 256-step BPTT, four transition exposures, and exact ten-source
  weights `0.30/0.03/0.03/0.06/0.08/0.08/0.12/0.15/0.05/0.10`.
  Pre-VG039 anchors retain `0.85` mass and VG052 receives `0.10`. Numerical
  admission requires strict overall/VG052 improvement and every prior cap;
  it authorizes only a fresh multi-offset six-gate screen.
- VG053 trainer/runner/test/preregistration hashes are
  `b7d7753c...`/`92fd257c...`/`a1360cde...`/`f6540ce0...`.
  No live authority exists.
- VG053's first preflight on commit `7f6c7ce8...` passed 29 tests, then
  stopped before output/state creation, baseline validation, or optimizer
  update: the generic dataset audit correctly rejected the intentional VG052
  reset jump from phase zero to 3/4. Abort/log hashes are `01d407b2...`/
  `8966c11e...`. The default-preserving repair enables initial jumps only for
  the explicit VG052 dataset and suppresses that reset jump as a transition;
  all later phase changes remain ordered. Never retry the old source unchanged.
- Repaired VG053 completes once from commit `d3438d82...` in `705.797076 s`
  with `5,907` optimizer updates. Balanced validation improves
  `0.037399526370 -> 0.032892203159`, and VG052 validation improves
  `0.115770637716 -> 0.093695282967`. All ten source caps, exact source
  weights, and `4.0x` transition exposure pass. Checkpoint/report/state/log
  hashes are `f0a9812f...`/`ec8b81b3...`/`1a553be4...`/`89f36999...`;
  completed resume is report-exact. Numerical admission SHA is `3838729e...`.
- VG054 screens the full VG053 update on six gates at offsets `0/8/16/24`,
  64 episodes each and 3,072 steps. Compare against frozen VG051; require no
  Gate-1/2/crash regression plus strict finish/Gate-6/5/4/3/mean progress.
  Full-update failure authorizes only a fraction bracket; success authorizes
  only a larger independent offline screen.
- VG054 manifest/evaluator/runner/generic-test/focused-test/preregistration
  hashes are `001e385a...`/`c0fb4710...`/`8a68ff33...`/`c1cf8ffb...`/
  `952754b8...`/`9835f2f2...`. No live authority exists.
- VG054 rejects the full VG053 update. Across offsets `0/8/16/24`, VG033
  versus VG053 Gate-1--6 reach changes
  `256/241/52/5/0/0 -> 256/226/46/1/0/0`; crashes improve `17 -> 10`, but
  misses regress `22 -> 32` and mean gates regress
  `2.1640625 -> 2.06640625`. Report/state/log hashes are `2124dc08...`/
  `e9c67d4e...`/`2a3d7282...`; rejection SHA is `bb78f928...`. Never retry
  VG054 unchanged and retain VG033.
- VG055 preregisters fractions
  `0/0.025/0.05/0.10/0.20/0.35/0.50/0.75/1.0` on new offsets
  `32/40/48/56`, 32 episodes per offset. Qualification requires hard
  transport, Gate-1/2 and crash preservation, plus strict downstream
  progress over the alpha-zero parent. Evaluator/runner/test/preregistration
  hashes are `0a876b34...`/`8998f64e...`/`625930c2...`/`9cfa9d7d...`.
  A selection remains offline-only pending a larger independent screen; no
  live authority exists.
- VG055 completes once from commit `4cf43b59...` and selects alpha `0.75`.
  Against alpha zero on offsets `32/40/48/56`, Gate-1--6 reach changes
  `128/110/18/0/0/0 -> 128/114/25/1/0/0`, crashes improve `6 -> 5`, misses
  improve `22 -> 17`, and all hard predicates pass. Checkpoint/report/state/
  runner-log hashes are `d4076edd...`/`bb026980...`/`79414b9b...`/
  `3d19ab9c...`; completed resume is report-exact. Admission SHA is
  `ae892d09...`. This sparse result grants no live authority.
- VG056 is the larger independent confirmation: VG033 versus VG055 on
  offsets `64/72/80/88`, 64 shared episodes each. In addition to hard
  transport and Gate-1/2/crash preservation, it requires strict downstream
  gain and nonzero Gate-5 reach to break the established bottleneck.
  Evaluator/runner/test/preregistration hashes are `e354477c...`/
  `8b35810b...`/`e7f6c8b2...`/`483736a7...`. No live authority exists.
- VG056 rejects the VG053/VG055 update line. On 256 new episodes, VG033
  versus VG055 Gate-1--6 reach is
  `255/229/41/1/0/0 -> 255/236/37/1/0/0`; crashes improve `14 -> 11` and
  misses `36 -> 24`, but Gate-3 regresses and Gate-5 remains zero. Report/
  state/log/rejection hashes are `f1613090...`/`af60d5a3...`/`eca6cfca...`/
  `c97082e4...`; completed resume is report-exact. Never retry VG056 or the
  interpolation line unchanged.
- VG057 is a causally distinct learned-policy repair. It loads VG033 into the
  existing `VQ2PhaseResidualActor`, preserves every base parameter exactly,
  and trains only the zero-initialized 1,024-weight phase-gated joint action
  residual on VG039's full-start, actor-visited warmed sequences. Six epochs
  use phase weights `0/0.25/0.5/2/32/32...`; phase 4 directly targets Gate 5.
  The first source completed all six epochs/`5,330` updates, then local
  variable shadowing stopped checkpoint packaging; checkpoint/report were
  never written. State/log/failure-record hashes are `e9e90792...`/
  `4ef9ff5b...`/`627bca44...`. The repaired source reruns deterministically,
  fixes packaging, and selects the best epoch satisfying the stability caps.
- Repaired VG057 trainer/runner/test/preregistration hashes are `4da484f7...`/
  `f5d7014b...`/`040b7703...`/`a5365f3e...`. Numerical admission authorizes only a
  teacher-free residual-scale bracket; no live authority exists.
- Repaired VG057 completes in `59.061321 s` and deterministically selects
  epoch 4 after all `5,330` updates. Base parameters are exact and phase-2/3/4
  validation MSE improves `0.147129/0.048814/0.124329 ->
  0.090564/0.044920/0.080821`; residual L2 is `2.807989`. Checkpoint/report/
  state/log hashes are `cf3825ed...`/`843f44a9...`/`88c00cdd...`/
  `68119d5c...`; completed resume is report-exact and admission SHA is
  `24edfd5c...`. This is supervised evidence only.
- VG058 brackets residual scales
  `0/0.01/0.025/0.05/0.10/0.20/0.35/0.50/0.75/1.0` over fresh offsets
  `96/104/112/120`, 32 episodes each. A scale must preserve Gate-1/2 and
  crashes, pass every hard transport predicate, improve downstream reach, and
  reach Gate 5 beyond the zero-residual parent. Generic/wrapper/runner/test/
  preregistration hashes are `4b27e3cc...`/`4772eff8...`/`8acbc4a8...`/
  `eb56a779...`/`f7d853d9...`. No live authority exists.
- VG058 rejects every shared-residual scale. Alpha `0.05` is the best local
  signal, changing Gate-1--6 reach `127/118/27/2/0/0 ->
  127/114/29/3/0/0` and crashes `5 -> 4`, but it regresses Gate 2 and never
  reaches Gate 5. Full scale collapses Gate-2/3 reach to `97/3`. Report/state/
  log/rejection hashes are `c7bcfc83...`/`246fe5d9...`/`c4a19f89...`/
  `3ffd88cf...`; completed resume is report-exact. Never retry unchanged.
- VG059 introduces an indexed learned residual table inside the recurrent
  Puffer actor. Only public index 4 receives `(4/16)*VG057_weight`; all other
  heads are zero, so action and recurrent state through Gate 4 remain exactly
  VG033. Physical scales `0/0.1/0.25/0.5/0.75/1/1.5/2/3/4` are screened on
  256 fresh episodes each at offsets `128/136/144/152`. Actor/wrapper/runner/
  focused-test/actor-test/preregistration hashes are `f66116b6...`/
  `6ef13f27...`/`3ca3b058...`/`a2e9c547...`/`190b0919...`/`c11356fe...`.
  Require exactly equal Gate-1--4 reach and new Gate-5 reach; no live authority.
