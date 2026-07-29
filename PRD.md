# PRD: Drone Gate Navigation Challenge (Source of Truth)

Last updated: 2026-07-26
Owner: `PufferLib/native-v4-drone-port`

## Source of Truth Policy

- This file is the active product and execution PRD for the race program.
- Legacy hover-era and historical experiment logs remain preserved in git history on the stabilized `3.0` checkpoint branch.
- If conflicts exist between this file and archived material, this file wins.
- The newest explicitly dated decision supersedes older sections even when a
  preserved heading or sentence says “active” or “current.” As of 2026-07-16,
  `AGENTS.md` and `docs/competitive_lap_execution_prompt.md` initialize and
  execute the latest direction; older policy-track sections are provenance.

## Challenge Goal

Build a fully autonomous drone stack that passes all gates in strict order and minimizes valid completion time in the official virtual competition environment through the TS-002 observable interface: MAVLink v2 telemetry/control plus the forward camera stream.

## Program Goal And Completion Contract

Goal:

- Win, or place competitively enough to credibly contend for winning, by submitting a fully autonomous TS-002-compliant drone racing stack that completes the official DCL course in strict gate order with a competitive valid time.
- The primary submission path is the simplest inspectable autonomous controller
  that produces a competitive valid lap through the official interface. A
  deterministic camera/IMU `course-fsm`, a modular hybrid, or learned
  components are eligible; the public rules do not impose an end-to-end neural
  policy. Learning is added only where a frozen failure set proves it is needed.
- Deterministic code may own perception, planning, pitch/roll/yaw/thrust,
  lifecycle, command-rate safety, collision abort, reporting, and MAVLink reset
  throughout the timed run. Human input and privileged simulator state may not.

Completion definition:

- The project is not complete when a native checkpoint scores well. It is complete only when the stack runs through the official SITL/MAVLink + camera interface, without privileged native state, and produces deterministic evidence that it can complete the course.
- A submission candidate exists only after it:
  - maintains heartbeat at `>=2 Hz`;
  - sends accepted commands with an engineering target `>=50 Hz` and strict cap `<100 Hz`;
  - consumes TS-002-confirmed telemetry and camera/perception-derived gate state;
  - passes gates in strict order with no human interaction during the timed run;
  - writes deterministic JSON/CSV reports with completion time, ordered gate passes, invalid-run/crash status, command-rate metrics, telemetry dropouts, and vision-stream health;
  - demonstrates reliability first, then competitive speed.

Competition milestone ladder (submission path):

1. **First-gate proof** — pass the first gate in SITL using telemetry plus camera/perception-derived gate pose. **COMPLETE 2026-07-03** (`logs/sitl/agent_sprint_13_smoke.json`, `official_active_gate_index=1`).
2. **Post-pass survival** — pass gate 1, brake straight, and settle with no collision. **COMPLETE 2026-07-12** (three consecutive official gate-1 passes in attempts 046–048).
3. **Multi-gate proof** — reach official `active_gate_index>=2` with one
   uninterrupted autonomous controller stack from the timed start. A
   deterministic or hybrid stack satisfies this milestone; human handoff does
   not. **COMPLETE 2026-07-13** (`full_policy_rl_stage_c_v6c_gate2_001`,
   accepted, collision-free official index `2`).
4. **Course proof** — complete the official course in strict order with no human input.
5. **Winning proof** — tune completion time only after at least 8/10 valid full-course runs.

The native PufferLib environment remains an optional high-throughput component
training and regression backend; official-interface progress is the final
authority and native promotion is not a prerequisite for the primary path.

Work-continuity rule:

- Always work on the highest-priority unmet competition milestone. Current
  official best is a clean three-gate prefix (`active_gate_index=3`); no
  archived or current run proves Gate 4. The manual Gate-4 controller bracket
  is closed. The active support path is the learned six-gate recurrent policy
  curriculum, followed by native-to-live parity gates before any new official
  flight.
- Do not promote native-only checkpoints, R4/full-course curriculum, Q8/fake-
  quant, or broad neural training ahead of ordered multi-gate official progress
  unless explicitly labeled support-track diagnostics.
- Native training is an optional component-development backend, not a gate that
  blocks testing the deterministic/hybrid official-interface stack. See
  `docs/competitive_lap_execution_prompt.md`.

## Current VQ2 Promotion Decision — 2026-07-26

- VQ2 Training, not Submission, is the active target. Runtime evidence exposes
  camera, finite HIGHRES_IMU accel/gyro, actuator status, heartbeat, and official
  race progress, but no vehicle pose, odometry, or track transfer.
- The deployed controller must be the user's requested recurrent PufferLib
  policy. Native state-feedback is restricted to teacher labels; runtime input
  remains the legal 32-value camera/IMU/previous-action/progress ABI.
- N294 is the first promoted Gate-1 candidate: exact `512/512`, perturbed
  `422/512`, zero crash/timeout, checkpoint SHA-256 `a57ca5f4...`. Its Windows
  shadow passes `520/520` replay, `52.0 Hz` inference, `63.401 Hz` no-op cadence,
  all hashes, healthy streams, and zero reset/arm/setpoint/disarm.
- N295 passed the first VQ2 gate exactly once with the frozen recurrent
  PufferLib policy. Official Gate-1 time is `3.228605508 s`; the index-1 stop
  fired at `3.250 s`. It delivered `207` setpoints at `63.995 Hz`, with zero
  in-run collision, invalid state, command-rate violation, telemetry dropout,
  or drain-limit hit. One reset, exact deployment hashes, and exit/passive
  disarm proof pass.
- N295 cannot be retried unchanged. Before another Training command, diagnose
  its complete recurrent trace and source-lock a separately tagged Gate-2
  continuation experiment. Runtime control remains PufferLib-policy-only;
  VQ2 Submission remains unauthorized.

## Preserved VQ1 Promotion Decision — 2026-07-17

- This section supersedes older “next action” statements below. N104 Candidate
  002 was executed once and rejected after an official Gate-2 collision: Gate 1
  passed, official state remained index `1`, and no finish occurred. Do not
  rerun N104 unchanged. The reproduced historical parent remains the clean
  live-proven three-gate prefix.
- Native N142 is the retained legal Gate-4-through-Gate-6 tail. It passes both
  common and disjoint radius-`0.75 m` validations at `128/128`, zero crash and
  zero timeout. The visible final control is coordinate-free and consumes only
  active-gate relative pose/rate plus observed attitude. Synthetic spline
  coordinates used during N140 development were removed before promotion.
- The deployment composite resets three independent FP32 recurrent policies at
  the Gate-4/5/6 boundaries. Checkpoint SHA-256 values are N112/Gate 4
  `9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca`,
  N118/Gate 5
  `e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30`,
  and N115A/Gate 6
  `4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354`.
- Candidate 003 used the pre-N143 composite once and is rejected without an
  unchanged retry: Gates 1-2 passed, then frame ID `1001` collision occurred at
  active Gate 3/index `2` around `10.266 s`. Gate 4 was never activated and
  finish stayed `-1`. Passive post-stop status proves disarm (`base_mode=65`).
- N143 adds only the already-live-proven Candidate-128/129 observable Gate-3
  lateral intercept. It changes roll only at official index `2`; recurrent
  state and pitch/thrust/yaw remain exact. Formal adapter projection over all
  ten Candidate-129 traces covers `237` Gate-3 samples and all three promoted
  modes with zero violations; the archived batch's accepted result is `8/10`.
- Linux and actual Windows Python replay the same eight-episode, `15,981`-record
  dataset through all three late phases with maximum action error
  `6.616115570068359e-6`; the N143 Gate-3 branch therefore leaves the N142 tail
  unchanged. Updated callable SHA-256 is
  `d0119c0d9fa099cbe08bdbd7072e7089f9961250183b3fa358cc01ce668cf9a4`;
  wrapper SHA-256 is
  `e241e7596c64c07a7df62eadbb2aad878f8c2e9cadfd2ac2a63ef71fa00504db`.
  Focused verification passes `65/65`; both PowerShell entry points parse in
  Windows.
- Zero-command N143 Shadow 003 passed against the reused v3385 process:
  `223/223` phase-0 ticks replay within `2.11e-7`, all 11 manifest artifacts
  match, streams are healthy, and reset/arm/setpoint/disarm are all exactly
  zero. JSON/parity SHA-256 values are `d4be9256...`/`7e5225af...`.
- Before flight, correct the evidence metric exposed by Candidate 003: fixed-
  rate command Hz now uses the first-to-last wire-command interval rather than
  reset/calibration-inclusive attempt time. Since runner/adapter
  hashes changed after Shadow 003, promotion order required zero-command N143
  Shadow 004. It passed `242/242` ticks within `2.41e-7`, all 11 manifest hashes,
  healthy streams, finish `-1`, and reset/arm/setpoint/disarm all zero. Evidence
  hashes are `a31c92ee...`/`fd122b48...`.
- Candidate 004 then ran exactly once and is rejected without an unchanged
  retry. It passed Gate 1, collided with ID `1001`/impact `9.2843` at active
  Gate 2/index `1` around `7.844 s`, and never activated Gate 3/N143. Finish was
  `-1`; exit and passive disarm passed. The corrected metric proves a real
  publisher rate of only `42.128 Hz` over `7.406 s`, not a reporting artifact.
- Repair the transport before another flight: a shorter GIL handoff, Windows
  timing/priority setup, hot deadline wait, and catch-up bounded to `95.2 Hz`.
  Shadow mode now runs the same cadence scheduler against a no-op sink under
  live camera/inference load while still sending zero MAVLink setpoints. Smoke
  acceptance and the independent shadow verifier require `[50,100) Hz`, zero
  rate violations, and zero probe setpoints. Tests pass `92/92` focused Linux
  and `38/38` actual-Windows smoke. Shadow 005 proves cadence `54.569 Hz` over
  `9.969 s` with zero violations/setpoints, but full parity fails because the
  simulator was inactive after Candidate 004 (`race_start=-1`): `96` frames,
  zero gate detections/visible observations. Recover R1 through the existing UI
  helper without relaunch. Zero-command Shadow 006 then fully passes: cadence
  `52.518 Hz` over `10.187 s`, zero cadence violations/MAVLink probe setpoints,
  `42/42` visible ticks within `1.58e-7`, all 11 manifest hashes, healthy
  streams, and lifecycle/control zeros. Evidence hashes are
  `b274a99e...`/`2e38c1df...`.
- Candidate 005 ran exactly once and is rejected without an unchanged retry.
  Actual wire delivery now passes at `50.233 Hz` across `5.375 s` (`271`
  commands, zero rate violations), so publisher cadence is no longer its
  acceptance blocker. It made no official pass and hit frame ID `1001`, impact
  `7.5096`, at active Gate 1/index `0` after about `5.766 s`; finish stayed
  `-1`. The last clean pose was about `4.444 m` forward, `1.451 m` left, and
  `0.076 m` up. Passive follow-up proves disarmed `base_mode=65`, index `0`,
  finish `-1`.
- The complete Gate-3 adapter is explicitly rejected for Gate 1: its counter
  mode would alter `185/397` samples across twelve traces that already passed
  Gate 1. N144 may reuse only its positive adaptive/early/close roll floors at
  official index `0`, leaving the Gate-3 counter-bank disabled there and Gate
  2/index `1` byte-for-byte behavior unchanged. The source-hashed separation
  verifier now passes: all `397` samples from 12 Gate-1-passing reports remain
  exactly unchanged, both 23- and 32-value trace ABIs are covered, and exactly
  six Candidate-005 samples change (`2` adaptive, `4` close) with zero non-roll
  error. It separately records `185` accepted samples the rejected full adapter
  would change. Report SHA-256 is `71eb295c...`. Next activate only this narrow
  index-0 branch, prove Gate 2 and the N142 tail invariant, check Windows parity,
  and require a changed-manifest zero-command shadow. That activation/parity is
  now complete: callable `f4364b03...`, pinned wrapper `2fd967a8...`, Linux
  `127/127`, actual Windows `13/13` plus launcher parses, and unchanged
  `15,981`-record late-tail maximum error `6.616115570068359e-6` in both
  runtimes. The inactive post-Candidate-005 state ignored MAVLink command
  `31000` because no race session was active, so the UI was recovered through
  a recovery-only process relaunch and window-relative login/R1 selection.
  Passive tag `n144_six_gate_composite_shadow_007` then passed in active R1:
  all `35/35` visible Gate-1 samples replay within
  `1.3715932369235719e-7`, all 11 hashes match, no-op cadence is `50.643 Hz`
  over `10.031 s` (`509` samples), and cadence violations plus reset/arm/
  MAVLink-setpoint/disarm counts are zero. Evidence SHA-256 values are
  `066e29d8...`/`bfb7eef6...`. This authorizes exactly one bounded Candidate
  006 under tag `n144_six_gate_composite_gate4_bounded_006`: one normal
  policy-ready MAVLink-`31000` reset, `60 Hz`, at most `30 s`, one attempt,
  and target/min/stop official index `4`. `FullLap` remains unauthorized and
  Candidate 006 may not be repeated unchanged. Candidate 006 ran exactly once
  and is rejected: after the detected normal `31000` reset it made zero passes
  and collided with ID `1001`, impact `9.0863`, at Gate 1/index `0` after about
  `4.734 s`; finish stayed `-1`. Its N144 branch commanded normalized roll
  `0.9`/decoded `0.45 rad` during the approach, so the transferred positive
  floor is not retained. Actual wire delivery was also only `46.204 Hz` over
  `4.610 s` (`214` commands). Passive follow-up proves disarmed `base_mode=65`,
  system status `3`, index `0`, finish `-1`. Freeze live attempts and extend
  the offline source-hashed diagnosis to Candidate 006 before another policy
  mutation or simulator command. The diagnosis now passes. Candidate 006 has
  five recorded N144 roll changes from the recurrent base, including a `0.9`
  bank triggered only `1.49e-8 m` beyond the old `-0.3 m` threshold before
  trace rounding. N145 adds a direct severity requirement of `right<-0.8 m`:
  it changes zero of `397` accepted samples, preserves all six Candidate-005
  severe-miss corrections, and changes zero Candidate-006 samples. Linux/
  Windows separation hashes are `d73b0cb0...`/`28e02251...`; callable hash is
  `9eaab396...`. Windows contention benchmarking keeps the `1 ms` handoff and
  rejects lower switch intervals; of `70/80/90 Hz` requests, only `80 Hz`
  passes all five trials (`52.000–57.860 Hz`, average `55.691`, zero too-fast
  violation). Only the N145 wrapper requests `80 Hz`; other workflows retain
  the shared runner's `60 Hz` default. Linux passes `58/58`, Windows `18/18`
  plus parser checks, and both replay all `15,981` late-tail records with
  unchanged `6.616115570068359e-6` maximum error. Zero-command Shadow 008 was
  blocked before inference by the terminal post-Candidate-006 state: it saw
  zero stationary calibration samples despite `2,183` messages/`100`
  heartbeats. A diagnostic MAVLink `COMMAND_LONG` command `31000` was sent but
  ignored on the `THROTTLE DOWN` screen, so reset success is detected rather
  than inferred from transmission. Pause -> Back To Main Menu preserved the
  responsive PID `24476`; reselecting VQ1/R1 restored active index `0`, finish
  `-1`, base mode `193`, system status `4`. Replacement zero-command
  `n145_six_gate_composite_shadow_009` then passed: all `32/32` samples replay
  (`28` visible), maximum error `1.4837913513143786e-7`, all 11 hashes match,
  and the no-op probe delivered `53.620 Hz` (`538/10.015 s`) with zero cadence
  violations/probe setpoints and zero reset/arm/setpoint/disarm commands.
  Shadow/parity hashes are `f580dd01...`/`ca70caf5...`. This authorizes exactly
  one bounded `n145_six_gate_composite_gate4_bounded_007`: one verified normal
  MAVLink-`31000` reset from active R1, `80 Hz`, at most `30 s`, repeats/
  min-valid `1/1`, target/min/stop official index `4`. Accept only clean index
  `4`, finish `-1`, actual `[50,100) Hz`, all hashes, no collision/invalid/
  dropout/drain issue, exit disarm plus passive proof. No unchanged retry or
  `FullLap` is authorized.
  Candidate 007 has now run exactly once and is rejected. Its sole command
  `31000` reset was detected (`2667002 -> 3405 ms`, race start `3300 ms`) and
  calibration collected `60/60` samples. It made zero official passes and
  remained Gate 1/index `0`, finish `-1`, at the `30.047 s` bound. No collision
  or invalid state occurred. The closest accepted Gate-1 pose was `8.771574 m`
  at `4.094 s`, body `[8.771574,1.973604,-2.508122]`; real-gate detections were
  then lost and final detection age reached `17.266 s`. Transport passed:
  `1,644/29.875 s = 54.996 Hz`, zero violation, `7,559` messages, `3,823` IMU,
  `237` frames, zero dropout/drain hit. Exit and passive disarm passed
  (`base_mode=65`, system status `3`, index `0`, finish `-1`). Freeze live
  attempts and source-hash/replay the full Candidate-007 trace against the
  recurrent base, N145, Candidates 005/006, and retained Gate-1 passes before
  any Candidate 008. `FullLap` remains unauthorized.

## Competition Strategy Reset

Decision as of 2026-07-16 — recovered official Gate-2 lineage:

- Course-count correction: the user confirmed that official VQ1/R1 has **six
  ordered gates**. `active_gate_index=3` is the transition to Gate 4, not the
  final gate. Official course proof requires all six ordered passes plus a
  nonnegative finish timestamp. The four-gate native training course is an
  incomplete development proxy and must not define the official phase count.
  Runs through Candidate 206 used target/min index `3`; they prove only a clean
  three-gate prefix/entry to Gate 4, never Gate-4 completion. Freeze live runs
  until the post-prefix controller and detector state reset independently for
  Gates 4, 5, and 6 under a six-phase observable index. Resume with target/min
  index `4`, then map Gates 5-6 after an authoritative Gate-4 pass.
- Six-phase decoding, per-gate post-prefix resets, per-gate delayed edge reset,
  and an authoritative `--stop-after-official-gate-index` are now implemented.
  Focused tests pass `102/102`; Windows compile/CLI checks pass; callable copies
  match SHA-256
  `00f5d0e3a6288170612a1c3a7e2217239bbb2a4cd9ff6bdbedd9f52ec2e50bb1`.
  Candidate 207 ran the exact safe Candidate-205 controller under the corrected
  contract for five bounded attempts. Attempts 1/3/5 reached index `3` cleanly;
  attempts 2/4 failed at Gate 3; none reached index `4` or set a finish time.
  Close the manual Gate-4 pulse/re-anchor/cooldown bracket. The next mechanism,
  explicitly selected by the user, is one recurrent policy trained across six
  ordered gates. Its deployment input is the exact camera/IMU/action contract
  plus normalized official progress `active_gate_index / 6`; unknown Gate-5/6
  geometry is randomized in training and is never embedded in live control.
  Native completion is only a promotion screen; official six-gate progress and
  `race_finish_time_ns` remain authoritative.
  The first bounded native bootstrap is now measured: the expanded parent
  completed `3.125%` of exact-512 mixed six-gate episodes; a 1M-transition child
  reached `3.906%` but increased mixed crash and is not promotable. On the
  isolated Gate-4 segment stratum it improved completion `4.492% -> 5.859%`
  while slightly reducing crash. Retain it only as the preregistered N002
  segment-training warm start. N002 uses 100% segment starts, `3e-5` learning
  rate, and a 3M-transition cap; accept only at segment success `>=10%`, crash
  `<=5%`, gates `>=4.5`, followed by a non-regressing full-start screen.
  N002 missed all three thresholds and is rejected. Reverse Gate-6 curriculum
  then learned lateral acquisition but exposed low-floor crashes; reward-only
  PPO shaping plateaued. Targeted supervised roll/thrust fitting reduced crash
  from `12.5%` to `0.586%` in the safest child, but its `2.0 m` completion is
  only `35.55%`. The same child completes `56.45%` at a `3.0 m` teaching
  aperture. N009 therefore trains with that aperture plus a terminal radial
  penalty whose reward break-even is near `1.9 m`, then screens unchanged at
  `2.0 m`. N009-N010 did not materially center. A one-step DAgger refresh N011
  remains below threshold at full randomization (`41.80%` success), but proves
  the curriculum at narrower widths: `83.20%` success at 50% width and `59.18%`
  at 75%, with crash below `5%` on both. N012 trains the measured 75%-width rung
  before returning to full width. N012 was flat, but N013's full-action
  supervised fit solved the isolated late gate: exact-512 success `85.35%` at
  75% domain width and `63.87%` at full width, with zero crash and timeout in
  both. N015-N017 showed that phase-local decoder fitting was insufficient.
  The direct full-course branch then established a perfect six-gate expert on
  a deterministic straight-extension, radius-3 homotopy rung. Expert-only BC
  and fully on-policy DAgger failed closed loop, while N033 scheduled-sampling
  DAgger (`25%` student/`75%` expert execution) produced the first accepted
  single recurrent six-gate policy: exact-512 success `100%` at radius `3.0`
  and `96.68%` at radius `2.75`, with zero crash/timeout. The unchanged policy
  scores `79.30%` at radius `2.5` and `19.92%` at radius `2.0`. N036's narrower
  full update and N037 trust-region blends were rejected. Later calibration
  and bounded PPO produced retained parent N058, which completes `412/512` at
  radius `0.95` and `455/512` at radius `1.0`, with zero crash/timeout. The
  N071 environment rung keeps Gates 1/6 at radius `0.75`, Gate 4 at `0.875`,
  and the other gates at `0.95`, where unchanged N058 completes `411/512`.
  N072 did not promote Gate 4 to `0.75`; its best child was `399/512`.
  N073 restored exact common-seed parity for direct FP32 collector loading.
  N074's broad spline teacher failed; N075 is invalid target evidence because
  its command omitted contract settings. Under the corrected contract,
  unchanged N058 completes `97/128`; N076 Gate-4-only feedback reaches at most
  `93/128`, and N077's isolated gate-indexed bias reaches at most `96/128`.
  Reject both mechanisms without fitting or exact-512 promotion. N078 then
  established an accepted observable gate-indexed adapter. N083 retains
  unchanged N058 with a native-index-2 roll `-0.0125`, thrust `+0.00125`
  bias; N084 tightens that gate to radius `0.85` and completes `412/512`, zero
  crash/timeout. N085's `0.825` continuation and N086's second-gate candidate
  (`407/512`) are rejected. N089 adds a center-safe index-1 adapter and scores
  `413/512`; N090-N093 reject tighter local continuations. N094's first
  six-gate anchor-mixture PPO is rejected, and N095 fixes a causal simulator
  flaw by separating late-course position sampling from vehicle/start RNG.
  N096 then trains raw N058 end-to-end over a 50/50 nominal/randomized
  six-gate mixture, with no runtime adapter: retained step `491,520` improves
  randomized success `9/128 -> 12/128`, average gates
  `4.0546875 -> 4.1953125`, and retains fixed N071 at `413/512`, zero
  crash/timeout. This modest native gain is not an official candidate.
  N097's randomized-only late teaching aperture and N098's observable-reward
  correction both regress to a best `11/128`; reject every child and retain
  N096. N098 leaves a default-off, native-tested guard for suppressing only a
  hidden randomized-next-gate reward, but that guard alone is not a promoted
  optimization mechanism. N099-N101 test success-filtered elite action
  perturbation on disjoint training seeds. The collector infrastructure is
  valid, but every source-regularized decoder child regresses. N102 prevents
  prefix leakage with phase-local learned heads and still reaches only
  `11/128` and `4.164062` average gates. Reject the entire mechanism and retain
  N096. No N-series checkpoint is authorized for official
  flight, and synthetic Gates
  5-6 are never treated as official geometry.
  N103 then tests observable-phase-aware recurrent replay with the same single
  full-start actor and exact N096 course/reward contract. All 32 children have
  zero crash/timeout, but best completion is only `11/128` at `4.15625` average
  gates and best progress is `4.1875` at `10/128`. Reject N103 without a fixed
  retention screen, retain N096, and close phase-priority weighting alone. Its
  default-off terminal-prefix-safe implementation remains tested infrastructure,
  not an official candidate.
  N104 is now the retained raw six-gate native policy. It uses legal normalized
  progress plus a six-way active-gate indicator and trains only isolated
  Gate-4/5/6 encoder columns `27..29`; the Gates-1-through-3 encoder path and
  every shared recurrent/decoder parameter remain byte-exact. The initial
  apparent ties were invalid because a stale collector emitted zero phase
  flags. With the rebuilt collector, the zeroed source reproduces N096 exactly
  at `12/128` and `4.1953125`, while N104 step `983,040` reaches `16/128` and
  `4.2109375` and retained step `1,048,576` reaches `17/128` and `4.21875`, all
  with zero crash/timeout. Both promoted checkpoints and the control retain
  fixed N071 at `413/512`, `5.21484375` average gates, zero crash/timeout.
  Retained checkpoint SHA-256 is
  `e248ca36a46d422a18c91afbf98be9b15a1c7191b42e6a189afc6456861b4c9e`.
  N105's prebuilt gain ladder is canceled without a valid screen because its
  rationale was the stale flat result; do not select amplification after N104's
  corrected promotion. This remains native curriculum evidence only. No
  official simulator run is authorized until aperture/domain and native-to-live
  parity gates pass.
  Offline N104 callable parity now passes under both Linux and the actual
  Windows Python runtime on an eight-agent, `2,368`-step trace spanning all six
  phases: maximum action/state errors are `1.73e-6/7.16e-7`, with every reset
  reproduced. This clears checkpoint layout and deterministic recurrent
  inference, not live perception transfer. The Windows worktree still contains
  older runner/callable hashes, so deployment packaging remains a prerequisite.
  N106 rejects immediate deployment: fixed radius `0.75 m` on all six gates
  yields only `62/128`, `3.75` average gates, zero crash/timeout. All failures
  are at active Gate 2 (`24`) or Gate 3 (`42`); every episode clearing them
  finishes. The next bounded continuation should train only their legal phase
  columns `25..26`, preserving Gate 1, N104's learned late columns, and every
  shared recurrent/decoder parameter exact.
  N107 runs that isolated continuation and produces a causal common-128 gain:
  best step `327,680` reaches `74/128` and `4.1328125`, zero crash/timeout,
  with no serialized change outside columns `25..26`. It is rejected because
  exact-512 reaches only `284/512` (`0.5546875`) versus the required
  `410/512`. N108's anchored full-network run from N104 is also rejected: its
  best common-128 checkpoint reaches only `64/128` and `3.828125`. N109's one
  registered higher-learning-rate full-network rung from the causal N107
  parent produces a narrow common-128 winner at step `229,376` (`76/128`,
  `4.1484375`) but fails exact-512 at `288/512` (`0.5625`), `4.017578125`.
  Every result has zero crash/timeout. Reject N107-N109, retain N104, and close
  direct PPO continuation of this direction. Current-source Windows packaging,
  fresh-reset recovery, and passive live Gate-1 parity now pass. The refreshed
  shadow saves all `257/257` recurrent ticks, replays on Linux at max action
  error `4.35e-7`, matches every source/checkpoint hash, and sends zero control
  mutations. Bounded Candidate 001 then passes official Gate 1 at
  `5.032423496 s`, triggers its immediate index-1 stop at `5.016 s`, and
  records no collision/invalid state/dropout/drain/rate violation; post-stop
  `base_mode=65` proves disarm. This is a clean prefix, not a lap. One unchanged
  N104 attempt bounded to target/min/stop index `2` is the next action; no
  Gate-3 extension or full lap is yet authorized.

- The user correctly recalled a prior controller passing Gate 2. The evidence
  exists in the Windows worktree but was absent from the WSL log copy:
  `logs/sitl/official_policy_validation_summary_full_policy_rl_stage_c_v6c_gate2_001.json`.
- The run used `policy-attitude`,
  `scripts/policy_callable_checkpoint.py:infer`, and
  `checkpoints/drone_race_full_policy_stage_c/1783952962041/0000000001998848.bin`
  (SHA-256
  `25b8b886e803387dcf26a9876a2ef3370c959966ef7b74fdb2884bd71c6e7a19`).
  It passed acceptance at official `active_gate_index=2` with two ordered
  passes, no collision, no invalid state, no dropout, and no command-rate
  violation. `race_finish_time_ns=-1`, so it did not finish or lock a lap.
- This accepted artifact supersedes the mistaken 2026-07-16 assumption that
  official Gate 2 was unproven. First reproduce it unchanged using the Windows
  runtime and MAVLink reset `31000`; then retain it as the solved prefix for
  Gate-3 work. Do not replace it with a weaker `course-fsm` lineage merely
  because those logs are present in WSL.
- Exact reproduction `competitive_r1_reproduce_stage_c_v6c_gate2_083` passed on
  2026-07-16 at official index `2`, with acceptance true, no collision/invalid
  state/dropout/rate violation, `56.333 Hz` effective command rate, and no
  finish. The recovered Gate-2 baseline is therefore live and reproducible.
- The next smallest step is the preserved
  `v6c_latest_obsmatch_r135_dropout_e10.bin` Gate-3 checkpoint. Historical
  telemetry proves official index `3` at `9.109 s`; its invalidity came from a
  later collision near `19.3 s`. First reproduce only the non-final Gate-3
  prefix in a 12-second bounded window before changing or retraining it.
- That bounded reproduction passed as
  `competitive_r1_reproduce_gate3_r135_dropout_084`: official index `3`,
  acceptance true, no collision/invalid state/dropout/rate violation,
  `55.833 Hz`, and no finish. The clean three-gate prefix is now live and
  reproducible.
- A complete audit of both WSL and Windows evidence found no official index
  `4` and no nonnegative `race_finish_time_ns`. Gate 4 is the first genuinely
  unsolved segment of the six-gate course, not the finish. Diagnose the retained Gate-3
  policy's delayed post-pass collision offline before another live extension.
- That diagnosis localized the miss: Gate 4 began about `41.74 m` forward and
  `9.07 m` right after index `3`, but the policy crossed its plane still
  approximately `4-5 m` right of center, then reversed and collided. A narrow
  observable-only final-gate handoff now exists in
  `scripts/policy_callable_final_gate_handoff.py`. It preserves the recurrent
  checkpoint for Gates 1-3 and, only at official index `3`, levels roll, yaws
  to camera bearing, and regulates closing speed/vertical error. First compare
  centering in a 12-second non-finishing diagnostic; do not extend it through
  Gate 4 until the finish-lock decision is resolved.
- The first handoff diagnostic aborted at official index `2` after missing Gate
  3 by about `1.07 m`; it never activated final-gate control. Offline replay
  proved the adapter's pre-handoff actions bit-exact with the retained
  checkpoint across all available prefix samples. Permit one identical,
  uniquely tagged reliability replicate. If it repeats the prefix miss, pause
  Gate-4 work and harden Gate 3 before changing final-segment gains.
- The permitted replicate preserved clean official index `3`, but its initial
  final-gate handoff worsened the camera bearing from the `0.4485 rad` reference
  to `0.680 rad`. Evidence identified an inverted v3385 yaw-command sign and a
  motion-filter pose that became stale during the turn. The next bounded child
  changes only index-3 control: raw camera pose, inverted yaw correction, and
  braking until aligned. Its window remains 12 seconds and cannot intentionally
  finish the course.
- That bounded child, `competitive_r1_gate4_handoff_rawpose_087`, passed at
  official index `3` with no finish/collision/health violation. At shutdown its
  raw Gate-4 pose was approximately `25.26 m` forward, `4.89 m` right, and
  `0.1914 rad` bearing: a large improvement over Candidate 086's `0.680 rad`
  and Candidate 084's `0.4485 rad` same-time references. Braking stopped and
  reversed the closure, so Candidate 087 proves the corrected observable-only
  alignment behavior but not a final approach. Retain it as the Gate-4 parent;
  next add an explicit pre-crossing stop guard and test only a bounded approach
  that cannot finish R1.
- The guarded Candidate 088 never reached index `3`: it repeated Candidate
  085's left-side Gate-3 miss and collided at `9.266 s` (ID `1001`, impact
  `10.0640`) with official index `2` and no finish. This establishes a prefix
  reliability blocker: Candidates 084/086/087 passed Gate 3, while 085/088
  failed with the same signature. Pause final-gate tuning. Compare successful
  and failed Gate-3 traces offline and make one narrow prefix correction before
  another live handoff run. A discovered wrapper omission in the new stop-guard
  namespace was fixed and regression-tested; it did not affect Candidate 088
  because the final phase never activated.
- Candidate 089's narrow Gate-3 roll guard confirmed the diagnosis but did not
  yet pass: it improved the last right offset from `-1.324 m` to `-0.688 m`
  while holding `+0.22 rad` roll, then struck the left frame at `9.250 s`.
  The sign and observable gating are retained; the exact `8 m/+0.22 rad`
  values are rejected. The next bounded child changes only to `10 m/+0.35 rad`
  with release at `-0.3 m`, aiming to establish clearance before the plane.
- Candidate 090 made that exact change and passed official Gate 3 cleanly:
  index `3`, no collision/invalid/health violation, no finish, `56.372 Hz`.
  The absolute camera pose did not meet its conservative clearance proxy, but
  official race status is authoritative. Retain Candidate 090 and run an exact
  three-attempt short reliability batch before resuming Gate-4 work.
- The Candidate 091 reliability batch then achieved only `1/3`; failures
  remained left of Gate 3, including one run where the usable active-gate pose
  arrived only near `10 m`. Fixed roll authority is therefore insufficient at
  the inherited roughly `8 m/s` closure. Keep Gate 4 paused and add braking
  only while the observable close Gate-3 left-error condition is active; do not
  escalate roll again.
- Candidate 092 added that braking and reached the best raw Gate-3 position yet
  (`2.11 m` forward, `0.11 m` left), but still hit the frame because it carried
  about `+0.35 rad` bank and large lateral rate into the plane. Position-only
  release is rejected. The next narrow child must use the already-observable
  gate-right rate to counter-bank before crossing, with modestly stronger
  braking; fixed bank must not be increased.
- Candidate 093's rate-aware child failed: its derivative term reduced bank
  while the gate remained `1.57 m` left near the plane. Reject that gain pair
  rather than tuning it. The smallest next strategy is phase-level speed
  control: brake from official index `2` even before a trustworthy close pose,
  then apply Candidate 092's proven position correction at much lower closure.
- Candidate 094 proved that slowing prevents collision, but it stalled at
  official index `2`. Its last raw pose was horizontally centered (`0.17 m`
  left at `7.56 m`), while the filtered pose stayed stale and held full bank
  through an `8.45 s` detection loss. Use raw Gate-3 pose for the deterministic
  subphase, level on dropout/centering, and make a gentle level approach; reject
  the stale filter and `-0.40 rad` close brake.
- Candidate 095 rejected that raw-only subphase. Low-confidence background
  aliases at roughly `30-80 m` displaced the active Gate-3 observation, the
  recurrent thrust stayed near its minimum, and the run ended at official index
  `2` with collision ID `1002`, consistent with ground contact. Restore the
  associated motion-filter observation and leave checkpoint pitch/thrust/yaw
  unchanged.
- Candidate 096 added a filtered-rate bang-bang lateral intercept and reached a
  well-centered pose (`0.24 m` left, `0.11 m` down), but its counter-bank began
  only at the measured `2 m` plane. It hit the Gate-3 frame while still at about
  `+0.45 rad` roll. Retain the intercept structure and move its leveling switch
  earlier.
- Candidate 097 reached official index `3` before later crashing during the
  unvalidated post-pass phase. Because the diagnostic continued for `2.6 s`
  after the official transition and the latest contact was ID `1002`, it is
  evidence of one Gate-3 pass but not a clean promoted run.
- Candidate 098 isolated the same controller in three independent `10 s`
  windows and passed official Gate 3 cleanly in `3/3`: zero collisions, zero
  invalid runs, no finish, effective command rates `56.9-58.1 Hz`, and no
  rate/dropout/drain violation. Its `15 m/+0.45 rad` centering bank plus
  rate-projected `-0.8 m` switch and `-0.50 rad` counter-bank is the retained
  reliable Gate-3 child. The Gate-3 reliability blocker is resolved; resume
  bounded Gate-4 diagnostics with the observable 20 m no-crossing guard, while
  treating long post-index-3 survival as a separate requirement.
- Subsequent larger reliability evidence reopened that blocker. Candidate 101
  exposed a fourth-run high-speed left miss; Candidates 102-106 converted
  contacts into repeated stalls and were closed; a `9.7 m/s` conditional brake
  passed only `3/5` in Candidate 108, and a `9.0 -> 8.5 m/s` hysteretic governor
  passed only `2/5` in Candidate 110. Do not describe Gate 3 as reliability-
  solved from Candidate 098's smaller batch. Candidate 098 remains the best
  simple lateral parent. Candidate 111's `-0.04 rad` far-phase pitch brake then
  failed at Gate 3 and coupled into different recurrent pitch/thrust and
  intercept geometry; Candidate 112's action shadow did not remove that
  physical coupling and stalled. Direct far/close pitch braking and action
  shadow are closed. Gate-4 tuning is paused. The next preregistered child
  changes only lateral timing: apply Candidate 098's `+0.45 rad` bank between
  `20-15 m` for the replay-separated `>=7.0 m/s` high-speed tail. Candidate 113
  exercised that branch above `8.3 m/s` and passed official index `3` cleanly;
  Candidate 114 then passed the exact reliability batch `5/5`, including two
  exercised high-speed and three unexercised normal-speed cases. Combined
  Candidate 113/114 evidence is `6/6` with zero collisions/invalid states or
  finishes. Promote and freeze this Gate-3 prefix; bounded Gate-4 work may
  resume under the index-3/`20 m` no-crossing guard.
- Candidate 115 combined that prefix with the `8 m` raw Gate-4 continuity bound
  and `0.27` wall-clearance hover. It stayed clean for `14 s`; active bearing
  improved to about `0.175 rad`, and later-gate pose jumps did not reverse the
  yaw command. The active near gate then remained cropped at the lower image
  edge, so continuity is a retained safety filter, not an approach solution.
  The next narrow child adds only delayed observable descent after three seconds
  of wall-clearance hover, still under the 20 m guard.
- Candidate 116 proved that delayed `0.22` descent is safe and reduced apparent
  vertical error, but the unchanged `-0.12 rad` brake increased Gate-4 range
  back to about `36.9 m`. Retain descent and change only the delayed recovery
  pitch to `+0.06 rad` in the next guarded diagnostic.
- Candidate 117 added that pitch and collided with ID `1002`; the latest frame
  proved the active near gate was a dominant red component cropped at the right
  edge while the detector selected a tiny downstream gate. Close open-loop
  recovery. Candidate 118 restores Candidate 115's safe controller and changes
  only final-phase detection to reconstruct/prefer dominant cropped edge frames.
- Candidate 118 selected `frame_partial` live and cleanly triggered the hard
  stop at `18.78 m`, validating the detector and guard. Trace review found yaw
  was frozen by a stale first handoff seed: the real Gate-4 pose arrived
  `0.125 s` later with a `9.0 m` jump over the `8 m` bound. Add only a `0.5 s`
  acquisition grace, then restore the permanent bound.
- Candidate 119 validated that grace. Near-boundary Gate-4 geometry improved to
  about `3.5 m` right and `0.17 rad` bearing versus Candidate 118's `9.5 m`/
  `0.43 rad`; late aliases remained rejected and the run stayed safely outside
  20 m. Vertical error around `10.7 m` is now dominant. Candidate 120 adds only
  the already-safe three-second-delayed `0.22` descent, with no recovery pitch.
- Candidate 120 failed at Gate 3 before final logic activated, the first prefix
  failure after 11 clean runs. It crossed `6.5 m/s` at `18.66 m` but the
  configured `7.0 m/s` early bank waited until `16.74 m`. Candidate 121 changed
  only that threshold to the replay boundary `6.5 m/s` and passed the exact
  five-reset Gate-3 reliability gate `5/5`, with zero collision/invalid state or
  finish and healthy `56.381-58.286 Hz` command rates. Promote that threshold.
  Candidate 122 nevertheless exposed a residual severe-left tail: the early bank
  started at `18.46 m`, but the last fresh pose was still `2.59 m` left and it
  hit frame ID `1001` before Gate 4. Final logic remains untested. Candidate 123
  adds only maximum `+0.50 rad` roll when a `<=22 m`, `>=5.0 m/s` observable
  Gate-3 trajectory projects more than `3.5 m` left at the 2 m intercept plane;
  all pitch/thrust/yaw/recurrent/final behavior and the 20 m stop stay frozen.
  Candidate 123 exercised it from `20.43-17.75 m` and passed official Gate 3
  cleanly at `9.157 s`. Candidate 124 then failed promotion at `3/5`; one severe
  path hit frame ID `1001` only `0.195 m` outside the counter threshold after
  the maximum-bank branch released abruptly, and one unrelated run stalled at
  index 1. Candidate 125 changes only that release to a continuous projected-
  miss roll taper, `+0.50 rad` at `-3.5 m` through `+0.45 rad` at `-0.8 m`.
  Candidate 125 exercised the taper and passed official Gate 3 at `9.406 s`,
  with one inherited counter-threshold bounce but no sustained oscillation.
  Candidate 126 met the exact five-reset gate `4/5`. Its failed run was nearly
  centered (`0.16 m` left, `0.03 m` vertical, `0.048 rad` roll) but still hit
  frame ID `1001`; promote only for guarded diagnostics, not final reliability.
  Candidate 127 is the first authorized retry of Candidate 119 association plus
  delayed `0.22` descent under this prefix, with zero recovery pitch and the
  mandatory index-3/`20 m` stop. Candidate 127 failed before activation because
  `4.40-4.90 m/s` closure at about `19.2 m` sat just below the adaptive `5.0`
  threshold, dropping roll to `+0.282-0.297 rad`; it remained `1.26 m` left.
  Candidate 128 changed only that adaptive threshold to `4.0 m/s`; it exercised
  at `19.59 m`/`4.47 m/s` and passed official Gate 3 at `9.266 s`. Candidate 129
  met the unchanged ten-reset gate exactly `8/10`. One failure remained
  `1.36 m` left without counter-bank; the other was nearly centered but had a
  small frame impact. Promote only for the written guarded diagnostic. Candidate
  130 combines this prefix with exact Candidate 119 association plus delayed
  `0.22` descent, zero recovery pitch, and the mandatory index-3/`20 m` stop.
  Candidate 130 safely reduced near Gate-4 down error from about `16.7` to
  `9.2 m` and stopped at the `18.78 m` guard, proving descent; yaw remained
  stale because the 8 m stepwise association can walk through unlogged aliases.
  Candidate 131 adds only a fixed 12 m post-grace anchor bound while retaining
  the 8 m step bound, descent, and guard. Candidate 131 validated it: guard
  geometry improved to `3.19 m` right/`0.158 rad` bearing, but down error stayed
  `9.94 m`. Candidate 132 retains the proven three-second delay and changes only
  post-delay thrust from `0.22` to `0.18`, under the same anchor and hard stop.
  It reduced down error to `5.8 m` but collided with intermediate structure ID
  `1002`; reject it. Candidate 133 midpoint `0.20` also hit ID `1002`, closing
  stronger descent at safe `0.22`. Candidate 134 adds only bounded final lateral
  roll (`kp=0.04`, max `0.15 rad`) to reduce the remaining right error and move
  around the wall; stale/dropout roll is forced level. It cleared the wall and
  reached `1.99 m` down but overshot to `14.76 m` left. Candidate 135 tests the
  interpolated low scale `kp=0.01`, max `0.05 rad`; it reached official index
  `3` but struck the same intermediate structure ID `1002` at `15.750 s`, so
  this scale is below the clearance threshold. Candidate 136 changes only the
  lateral scale to the arithmetic midpoint `kp=0.02`, max `0.10 rad`, preserving
  the exact association, safe descent, stale-roll leveling, and hard stop. It
  was collision-free but the `0.5 s` grace fixed a farther aperture; the active
  right-side gate returned at about `0.8 s`, was rejected, and the raw guard
  stopped early at `17.78 m` with frozen yaw. Candidate 137 changes only grace
  to `1.0 s`, covering that return while ending before the known `~1.85 s`
  later-gate alias onset. Candidate 137 validated it cleanly: the guard fired at
  `16.00 m` forward with `2.225 m` right/`0.138 rad` bearing/`7.650 m` down,
  improving every geometry term over Candidate 131. Candidate 138 repeats this
  exact configuration for five independent resets and requires `4/5` clean
  official-index-3 attempts; individual collision/invalid/finish state fails
  regardless of wrapper gate-progress accounting. Candidate 138 achieved only
  `3/5` clean: one Gate-3 ID-1001 strike and one post-Gate-3 ID-1002 strike.
  The clean final endpoints retained moderate `3.19-3.88 m` right error, so
  Candidate 139 preserves `kp=0.02` and raises only the cap `0.10 -> 0.12 rad`,
  adding early clearance authority above `5 m` error without changing near-gate
  proportional centering. It exercised that cap but still hit ID `1002` after
  prolonged timer-triggered descent, closing cap tuning. Candidate 140 restores
  cap `0.10` and adds only `PUFFER_FINAL_GATE_DESCENT_MAX_FORWARD_M=25`: after
  the retained three-second delay, `0.22` descent is allowed only while the
  associated active gate is also within `25 m`; otherwise hover is retained.
  The feature defaults disabled and the 20 m guard remains unchanged.
  Candidate 140 was clean and held hover to a `19.636 m` guard, but that stop
  preceded the three-second delay, leaving range gating unexercised. Candidate
  141 is an exact five-reset batch requiring `4/5` individually clean attempts
  and at least one long path that holds hover after three seconds while the
  associated gate remains beyond `25 m`; post-Gate-3 ID `1002` is disallowed.
  Candidate 141 reached `4/5` clean but the fifth run exposed stale descent:
  the last accepted pose stayed inside `25 m` while current frames were
  rejected/far, so held `0.22` persisted into ID `1002`. Candidate 142 changes
  only enabled range-gate semantics: dropout, rejection, non-forward, or current
  range above `25 m` restores `0.27`; `0.22` requires a current accepted
  `(0,25] m` pose plus the retained delay/vertical conditions.
  Candidate 142 validated it live: after the delay, rejected/far current poses
  down to `21.1 m` all held `0.27`, with zero stale-descent samples and a clean
  `19.592 m` guard. Candidate 143 repeats the exact controller for five resets,
  requires `4/5` individually clean, and disallows any post-Gate-3 ID `1002` or
  stale descent.
  Candidate 143 promoted exactly `4/5` clean: four final-active attempts had no
  collision/invalid/finish, ID `1002`, or stale descent; the sole failure was
  pre-handoff Gate-3 frame ID `1001`. Retain it as the safest guarded final
  diagnostic, not a completed lap. The no-crossing stop remains installed and
  Gate-3 reliability is still the limiting tail.
  Candidate 144 is the first crossing-capable attempt: exact Candidate 143,
  target/index `4`, `45 s`, no pre-crossing guard, and no controller scalar
  change. The smoke runner now stops immediately on a nonnegative official
  finish time and records `official_finish_stop`; focused verification is
  `71 passed`, with successful Windows compilation. If a finish occurs, archive
  it and use a separate MAVLink-31000 no-flight reset proof before another run.
  Candidate 144 failed before handoff at official index `2`, frame ID `1001`,
  so finish handling was untested. Exact retry Candidate 145 also failed only
  at Gate 3: official index `2`, ID `1001`, impact `1.7020` at `9.125 s`, no
  finish, healthy `57.425 Hz`. Candidate 146 was the final exact sampling retry
  and passed Gate 3. It ran the full `45.0 s` collision-free at official index
  `3`, no finish, healthy `56.711 Hz`; only five 8 Hz final samples descended
  (`0.625 s`) before the current-association fail-safe restored hover and the
  gate left view. Candidate 147 changes only final descent persistence: a
  one-shot `2.0 s` pulse may start after the retained `3.0 s` delay and a
  currently associated `(0,25] m`, vertically low gate, bridges dropout only
  until its fixed expiry, cannot restart, then forces hover. It is a guarded
  `18 s`, target/index `3`, `20 m`-stop diagnostic, not a finish attempt.
  Callable hash is
  `0330567714776a0512f10823f4970cf1ae9d300ad97dbc951a7659fa83035e22`;
  verification is `73 passed` plus successful Windows compilation and matching
  root/Windows hashes.
  Candidate 147 did not activate final control: local vision counted three but
  authoritative state stayed index `2`; Gate-3 ID `1001`, impact `11.0876`,
  ended it at `9.250 s`. Candidate 148 is one exact activation retry, with no
  code/scalar/execution change except its evidence tag. Candidate 147 supplies
  no bounded-pulse evidence.
  Candidate 148 was clean at official index `3`, guard-stopped at `19.2 m`, but
  never triggered the pulse because current association had not reconnected.
  Candidate 149 changes only the diagnostic stop from `20 m` to `15 m`, leaving
  every controller mechanism/scalar and the two-second one-shot cap exact.
  Candidate 149 ran the full `18 s` clean at official index `3` but still never
  re-associated or pulsed, closing guard-only sampling. Candidate 150 adds one
  disabled-by-default, one-shot re-anchor after `0.75 s` of step-coherent
  rejected poses and only once current forward range is `(0,25] m`; large jumps
  reset persistence, dropouts clear it, and a second re-anchor is prohibited.
  It retains the one-shot two-second descent, target/index `3`, `18 s`, and
  `15 m` stop. Hash is
  `2c898fa629708e0c783a579feb47c66ffbf0339b1de292d9cd52b85216db59a0`;
  `75 passed`, Windows compilation succeeds, root/Windows hashes match.
  Candidate 150 passed its custom criteria: full `18 s`, official index `3`,
  zero contact, one coherent re-anchor, one exact two-second descent pulse, then
  23 sampled hover commands. Candidate 151 retains one-shot descent but permits
  at most two `0.75 s` coherent re-anchors within a separate `30 m` re-anchor
  range; descent remains independently gated to `(0,25] m`, so the second
  re-anchor cannot restart it. Hash is
  `a23869da02c2f381d2a54d797e76183d164d3703431928c5877a82bd40593c5c`;
  `76 passed`, Windows compilation succeeds, hashes match. It is still an
  `18 s`, index-3, `15 m`-stop no-finish diagnostic.
  Candidate 151 stalled safely before handoff at official index `2` for the full
  `18 s`, with no collision/invalid/health fault; it provides no two-epoch
  evidence. Candidate 152 is one exact activation retry with only its tag changed.
  Candidate 152 passed Gate 3, but the `15 m` raw guard immediately fired on a
  stale `1.50 m` Gate-3 pose at `9.219 s`, before any final-policy sample.
  Candidate 153 disables only this false-positive diagnostic guard for `18 s`;
  target/index remains `3`, controller exact, immediate finish-stop enabled.
  Candidate 153 passed custom acceptance: full `18 s`, official index `3`, zero
  contact, two coherent re-anchors, one exact two-second pulse, 16 later hover
  samples, and restored live yaw/roll on the second cluster. Candidate 154
  permits at most two two-second pulses with at least `1.0 s` hover between;
  every pulse still needs a fresh associated `(0,25] m` low gate and re-anchors
  remain capped at two. Hash is
  `5636035dafce1f8e98798690765bac2de628e52327a7fdd9570270540658d1ee`;
  `77 passed`, Windows compilation succeeds, hashes match. Run a `22 s`, target/
  index-3, unguarded diagnostic with immediate finish-stop retained.
  Candidate 154 remained clean for `22 s` and reached the best final geometry
  yet (`7.45 m` forward, `2.84 m` down), but only one pulse fired because a third
  coherent cluster exceeded the two-re-anchor cap. Candidate 155 changes
  re-anchor max count to `3` and ends any active pulse immediately on a fresh
  associated pose at the existing `<=1.5 m` vertical tolerance; maximum pulse
  count/duration/cooldown remain `2/2 s/1 s`. Hash is
  `78d3c268676a4f5d42058a770d492c5253f00b44d0449bb163514e513efe0fa7`;
  `78 passed`, Windows compilation succeeds, hashes match. Run `22 s`, target/
  index `3`, no raw guard, immediate finish-stop.
  Candidate 155 failed only at Gate 3: official index `2`, ID `1001`, impact
  `11.0822` at `9.047 s`; final logic was untested. Candidate 156 is one exact
  activation retry with only the tag changed.
  Candidate 156 ran clean at official index `3` for `22 s` but correctly held
  hover because no coherent near cluster qualified. Candidate 157 is an exact
  maximum-five-reset reliability batch. The outer runner now implements
  `--stop-after-official-finish`, records attempts actually run, and halts before
  any post-finish reset/flight. Focused verification is `80 passed`; Windows
  compilation succeeds. No controller tune is allowed within the batch.
  Candidate 157 achieved official index-3 progress on `4/5`, with `3/5` clean
  attempts. The only two-pulse member hit ID `1002` at `18.719 s`, impact
  `0.0462`, immediately after the second full two-second pulse; equal pulse
  durations are rejected. Candidate 158 retains a `2.0 s` first pulse but caps
  later pulses at `1.0 s`; every trigger, cooldown, count, re-anchor, and other
  scalar remains exact. Callable hash is
  `4d150ad4c797fa009187ebc3e37fcc5580c5965e16e3ace6144863248ab45686`;
  focused verification is `81 passed`; Windows compilation succeeds, hashes match.
  Candidate 158 still hit ID `1002` at `17.234 s`, impact `0.7683`, after only
  `0.625 s` sampled time in pulse 2, which began at `24.62 m`; duration alone is
  not the safety boundary. Candidate 159 requires later pulses inside `10 m`
  while retaining their `1.0 s` cap; pulse 1 stays `25 m`/`2.0 s`. Hash is
  `791eb85bff431e9b9167a25ebfce6fb36a91ba4442b4a5f1023ac698cd1ec336`;
  `82 passed`, Windows compilation succeeds, hashes match.
  Candidate 159 used only pulse 1 and still hit ID `1002` at `18.500 s`, impact
  `0.1955`, with the target `21.33 m` forward; the foreground wall makes all
  descent at that range unsafe. Candidate 160 changes primary descent range to
  `10 m`, making every pulse close-only. Re-anchor tracking remains `30 m`; all
  duration/count/cooldown and other controller settings stay exact. Candidate
  160 then passed cleanly at official index `3` for `25 s`, with zero descent,
  contact, invalid state, or finish. It validated hover outside `10 m`, but the
  retained braking pitch prevented closure: the closest accepted low-gate
  family was about `20 m` forward/`10.25 m` down before it receded and left
  view. Candidate 161 adds only a `+0.03 rad` hover-thrust approach after the
  existing `3 s` delay on a current associated low gate in `(10,30] m`.
  Rejection/dropout forces `-0.12 rad` braking plus hover, and all descent stays
  locked to a current associated `(0,10] m` pose. Callable hash is
  `74cd28981020aef99b9bd94626016a25b620e02d12f7bbdeb60b82559f3bc27d`;
  `83 passed`, Windows compilation succeeds, and root/Windows hashes match.
  Candidate 161 passed generic acceptance at official index `3` for `25 s`,
  zero contact/invalid/finish, and `57.2 Hz`. Its approach fired on 22 trusted
  samples, never through dropout, and improved closest accepted final range to
  `16 m`, but did not reach the `10 m` descent zone; no descent occurred.
  Candidate 162 changes only approach pitch from `+0.03` to `+0.06 rad`, the
  existing aligned-forward value, with all safety and prefix settings exact.
  Candidate 162 then failed before final activation at official index `2`, Gate-
  3 ID `1001`, impact `10.6418` at `9.094 s`; the bracket is untested. Candidate
  163 is one exact activation retry with only its evidence tag changed.
  Candidate 163 then ran clean at official index `3` for `25 s`, but 18 trusted
  `+0.06 rad` samples stayed at `24-29.29 m`; the active partial frame exited
  bottom-right, with no descent or finish. Reject the stronger pitch and restore
  `+0.03`. Candidate 164 changes only the first-pulse threshold from `10` to
  `17 m`; later descent remains `10 m`. The new bound is below both rejected
  `21-25 m` starts and immediately above Candidate 161's trusted `16 m` pose.
  Candidate 164 failed before handoff at official index `1`, ID `1001`, impact
  `10.7113` at `6.359 s`; the threshold was untested. Candidate 165 is one exact
  activation retry with only its evidence tag changed. Candidate 165 then ran
  clean at official index `3` for `25 s`, but 20 approach samples stayed at
  `21.6-29.09 m`; no pulse/finish occurred and the threshold remains untested.
  Candidate 166 is an exact maximum-five-reset opportunity batch with no tuning
  and automatic whole-batch stop on any official finish. Candidate 166 exhausted
  five attempts with no finish: attempts 1/3/5 were clean index-3 runs, attempt
  2 was an immediate index-0 ID `1002` reset-state contact, and attempt 4 was
  Gate-3 ID `1001`. No pulse fired. Attempt 3 exposed centered low
  `frame_partial` poses at `16.62 -> 16.30 m` for `0.25 s`, rejected by the fixed
  anchor and too brief for the `0.75 s` broad re-anchor. Candidate 167 adds one
  disabled-by-default close-only re-anchor: `0.125 s`, `(0,17] m`,
  `|right|<=2 m`, low gate, after delay, one use. Callable hash is
  `e59dbe7ccc643ca278bf4e588d9ab123cb03cf703e12c20aa8c2fc1e834486da`;
  `84 passed`, Windows compilation succeeds, root/Windows hashes match.
  Candidate 167 failed before final activation at official index `2`, Gate-3 ID
  `1001`, impact `3.6712` at `9.031 s`; the new branch was untested. Candidate
  168 is one exact activation retry with only its evidence tag changed.
  Candidate 168 also failed at Gate 3, index `2`, ID `1001`, impact `5.5783` at
  `9.188 s`; final logic remained untested. Read-only diagnostics found the
  four-logical-processor Windows host at `93-99%` CPU while recent controller
  delivery had fallen to `39-44 Hz`; the sim is healthy and unrelated apps stay
  untouched. Candidate 169 changes only transient controller-process priority
  to High, leaving code/config and simulator process exact. Candidate 169 raised
  delivery to `49.138 Hz` but missed `>=50 Hz` and still failed at Gate 3, ID
  `1001`, impact `1.0489` at `9.219 s`; do not retain priority as a fix.
  Candidate 170 is an exact maximum-five-reset Candidate-167 batch at normal
  priority, no tuning, with whole-batch stop on any official finish. Candidate
  170 exhausted five attempts with no finish or pulse: three clean index-3
  runs, one Gate-3 ID `1001`, and one collision-free index-1 stall. Clean raw
  minima were `19.64/19.64/24.69 m`; the closest centered post-delay partial
  frame was `19.64/0.92/9.70 m` forward/right/down. Candidate 171 brackets the
  first range to `20 m` while cutting first/later pulses to `0.5/0.5 s`; later
  range stays `10 m`. Use a fixed maximum-five batch, no tuning, batch finish-stop.
  Candidate 171 achieved `4/5` clean and one safe half-second pulse; that attempt
  reacquired near `19.6 m`, with down error improving from about `9.5-9.9` to
  `7.4-8.1 m`, no contact or finish. Candidate 172 changes only later-pulse
  range from `10` to `20 m`, permitting one feedback-separated second half-
  second pulse while retaining maximum count two. Run a fixed maximum-five batch.
  Candidate 172 produced one clean two-pulse attempt: both stages were bounded
  and separated, no contact followed, and the active frame later closed to
  `7.45 m` with about `4.10 m` down. The max-two budget blocked another eligible
  correction; no finish occurred. Candidate 173 changes only maximum pulse
  count from `2` to `3`, in a fixed maximum-five batch with finish-stop.
  Candidate 173 yielded one clean index-3 handoff in five attempts but no pulse
  or finish. The coherent rejected low-gate cluster reached
  `16.62/2.13/7.89 m` and `18.19/2.61/8.95 m` forward/right/down; only the
  `|right|<=2 m` one-use close-re-anchor bound excluded it. Candidate 174 changes
  only that close bound to `3.5 m`, retaining the three half-second stages and
  every other safety, prefix, batch, and finish-stop setting. Candidate 174
  achieved `3/5` clean index-3 runs and one safe half-second pulse, but no
  finish. A later close family could not reauthorize because the fast re-anchor
  had spent its one-use budget. Candidate 175 changes only a new default-
  preserving `PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_COUNT` from `1` to `2`; the
  `3.5 m` bound and three-pulse ceiling remain exact. Callable SHA-256 is
  `1d22e54281dfa50190693c56167dc37da0fe5a1f5d7571429e1497f177b2a351`;
  focused verification is `87 passed`, Windows compilation succeeds, and
  root/Windows hashes match.
  Candidate 175 produced two clean index-3 runs, but their closest final raw
  poses remained `25.95 m`; no pulse or second close budget was exercised, while
  three attempts failed at Gate 3. Candidate 176 retains count `2` and changes
  only the close lateral admission from `3.5` to `4.0 m`, directly covering
  Candidate 174's measured second family at `3.72-3.94 m`. The roll cap, pulse
  rules, prefix, fixed maximum-five budget, and finish-stop remain exact.
  Candidate 176 achieved `3/5` clean index-3 runs and no finish. Its best run
  safely executed all three half-second stages, reached approximately
  `6.31/4.74/3.18 m`, and still had a current associated target after cooldown
  at `7.45/4.72/3.42 m`; only the maximum-three pulse budget blocked another
  eligible correction. Candidate 177 changes only pulse maximum `3` to `4`;
  close count two, `4 m` bound, all stage rules, prefix, fixed maximum-five
  budget, and finish-stop remain exact.
  Candidate 177 produced two clean index-3 runs, but their post-delay minima
  stayed about `27.87/23.35 m`; three attempts failed in the prefix. No pulse or
  fourth-stage evidence occurred. Candidate 178 is one exact tag-only fixed
  maximum-five Candidate-177 opportunity batch, with no tuning and whole-batch
  finish-stop.
  Candidate 178 produced two clean index-3 runs and no finish. Its exercised
  run used two safe stages; roughly `0.25 s` after stage 2, current geometry was
  `18.00/4.47/8.78 m`, but the one-second cooldown blocked stage 3 and the
  target then moved beyond `20 m`. Candidate 179 changes only cooldown from
  `1.0` to `0.25 s`; half-second ceiling, maximum four, fresh authorization,
  range/thrust/association/prefix, fixed maximum-five budget, and finish-stop
  remain exact. Reject on any contact.
  Candidate 179 produced two clean index-3 runs and no finish/contact, but its
  multi-stage run still separated starts by about `2.5 s`; the quarter-second
  cooldown never enabled a stage and gave no vertical benefit. Restore retained
  cooldown `1 s`. Candidate 180 uses the four-stage Candidate-178 parent and
  changes only descent thrust `0.22` to `0.20`; half-second ceiling, max four,
  fresh authorization, range/association/prefix, fixed maximum-five budget, and
  finish-stop remain exact. Reject on any post-handoff contact.
  Candidate 180 achieved one clean index-3 run with two safe `0.20` stages, but
  its best down error was `7.73 m`, worse than Candidate 176's retained
  `3.18 m`; restore `0.22`. Candidate 181 uses exact Candidate-178 pulse
  settings and changes only final lateral roll cap `0.10` to `0.12 rad` to
  counter measured rightward drift before target loss. Gain, stage rules,
  association/prefix, fixed maximum-five budget, and finish-stop remain exact.
  Candidate 181 yielded three clean index-3 runs and no finish/contact, but its
  exercised minima stayed near `17.63 m`; `0.12 rad` did not improve retention,
  so restore `0.10`. Candidate 182 uses exact Candidate-178 settings and changes
  only first-stage ceiling `0.5` to `1.0 s`; later stages remain `0.5 s`, with
  max four, cooldown `1 s`, `0.22` thrust, fresh `(0,20] m` authorization, all
  association/prefix settings, fixed maximum-five budget, and finish-stop exact.
  Reject on any post-handoff contact.
  Candidate 182 achieved `4/5` clean index-3 runs and no finish, repeatedly
  exercising one-second first stages and twice adding a later half-second stage,
  with zero post-handoff contact. Best down error stayed about `6.49 m`.
  Candidate 183 changes only first-stage ceiling `1.0` to `1.5 s`; later stages
  remain `0.5 s`, and max four/cooldown/thrust/range/association/prefix, fixed
  maximum-five budget, and finish-stop remain exact. Reject on any post-handoff
  contact.
  Candidate 183 achieved `4/5` clean index-3 runs and no finish, safely
  exercising 1.5-second first stages with zero post-handoff contact; best close
  geometry improved to approximately `10.94/7.11/4.80 m`. Candidate 184 changes
  only first-stage ceiling `1.5` to `2.0 s`, still fresh/current and `(0,20] m`;
  later stages stay `0.5 s`. Run at most five individually inspected attempts,
  stopping the whole branch immediately on post-handoff ID `1002`/contact or an
  official finish.
  Candidate 184 completed five individually inspected attempts: four clean
  index-3 runs, two full guarded two-second first stages, zero post-handoff
  contact, no finish, and best `10.67/6.03/4.58 m`; close the duration ladder.
  Candidate 185 restores exact Candidate-178 half-second stages and changes only
  `PUFFER_FINAL_GATE_HOLD_PULSE_LATERAL=1`: during an already active bounded
  pulse dropout it holds saved authorized lateral roll under the existing cap,
  then levels at expiry. Callable SHA-256 is
  `75f36a9988109e7b1c38aa8690c79e9fdf2001281cb250fa118c3601191ae1f1`;
  `88 passed`, Windows compile succeeds, runtime hashes match. Fixed maximum-
  five batch, no tuning, finish-stop; reject on post-handoff contact.
  Candidate 185 produced only one clean index-3 run; its `1.45 m` raw minimum
  was stale pre-delay handoff geometry and no pulse fired. Four attempts failed
  in the prefix, leaving saved-roll hold initially untested. Candidate 186
  produced one clean index-3 opportunity and exercised it exactly: one fresh
  `17.28/1.89/9.18 m` `frame_partial` authorized a half-second `0.22` pulse and
  `-0.0378 rad` roll; the roll remained exact through four `40-46 m` downstream
  aperture aliases and leveled at expiry, with zero contact. Target retention
  and geometry did not improve and no finish followed. Keep the option disabled
  by default and close further pulse-authority tuning. The next bounded child
  changes only final-phase detector selection so an available cropped edge
  component cannot lose to a centered downstream aperture; retained control,
  prefix, finish-stop, and reset proof stay exact.
  Candidate 187 preregisters exactly that perception child. Enable only
  `PUFFER_POLICY_FINAL_PREFER_ANY_EDGE_FRAME=1` on exact Candidate-178 control
  with saved-roll hold disabled. The option defaults off, activates only at/
  after raw official index `3`, and prioritizes a red edge component only after
  it passes the existing observable area/fill/size checks; Gates 1-3 and every
  normal detector caller are unchanged. Detector hash is
  `ebcdaae0df17bcb0c3e93ac9666c5574dffae6974b8d51a4ae98d7ba275d650a`
  in root/Windows; focused tests are `86 passed` plus `7 passed` on Windows and
  compilation. Run fixed maximum five, no tuning, whole-batch finish-stop;
  require a nonzero override and improved retention/geometry or finish, and
  reject any false edge capture or post-handoff contact.
  Candidate 187 reached index `3` once and produced `10/3` selections/
  overrides, but zero-delay priority captured the just-passed gate from about
  `1.72 -> 33.23 m`, rotated yaw `1.87 -> 2.28 rad`, then lost detections; one
  estimator collision event also invalidated the run. Reject immediate
  priority. Candidate 188 changes only
  `PUFFER_POLICY_FINAL_ANY_EDGE_DELAY_S=3.0`, retaining prior dominant-edge
  behavior through the existing three-second handoff/wall-clearance interval.
  Verification is `87 passed` root/`35 passed` Windows. Run fixed maximum five
  under `competitive_r1_gate4_any_edge_delay3_batch_188`, no tuning and finish-
  stop; reject any pre-delay override, false edge capture, or contact.
  Candidate 188 produced `3/5` clean index-3 runs with zero post-handoff
  contact and selector/override counts `13/5`, `9/4`, and `14/5`; retain the
  three-second delay. Two clean runs used two half-second pulses, but best
  delayed geometry was only `14.90/2.79/7.03 m`, worse than Candidate 176.
  Candidate 189 changes only later-stage pulse duration `0.5 -> 1.0 s`; the
  first stays `0.5 s`, with maximum four, cooldown/thrust/range/detector/
  association/prefix exact. Run fixed maximum five under
  `competitive_r1_gate4_any_edge_delay3_later1_batch_189`, no tuning, finish-
  stop; reject any post-handoff contact.
  Candidate 189 was preflight-rejected on every wrapper invocation because the
  callable forbids later pulses longer than the first; no smoke/candidate flight
  occurred. A separate reset snapshot proved active `0`, finish `-1`, zero
  collisions, healthy status `4`, and idle motors. Candidate 190 recombines
  independently accepted Candidate-182/188 behavior as uniform `1.0/1.0 s`
  first/later stages; max four/cooldown/thrust/range/delayed detector/
  association/prefix remain exact. Run fixed maximum five under
  `competitive_r1_gate4_any_edge_delay3_uniform1_batch_190`, no tuning, finish-
  stop; reject post-handoff contact or failure to restore hover.
  Candidate 190 produced one clean index-3 run and one safe one-second pulse,
  but best delayed geometry was only `18.38/-3.59/9.68 m`; restore Candidate-
  188 half-second stages. Candidate 191 changes only close re-anchor maximum
  count `2 -> 3`, retaining current/post-delay `0.125 s`, `(0,20] m`,
  `|right|<=4 m`, low-gate, and 8 m step guards. Run fixed maximum five under
  `competitive_r1_gate4_any_edge_delay3_closecount3_batch_191`, no tuning,
  finish-stop; reject false-family admission or post-handoff contact.
  Candidate 191 produced two clean index-3 runs but no third close admission;
  do not retain count three. Its actionable trace was a coherent
  `16.30/4.18/7.82 m` cropped family just outside the `4 m` lateral gate after
  one safe pulse. Candidate 192 restores count two and changes only close right
  bound `4.0 -> 4.5 m`, retaining every current/range/low/step/pulse/delayed-
  detector/association/prefix guard. Run fixed maximum five under
  `competitive_r1_gate4_any_edge_delay3_close45_batch_192`, no tuning, finish-
  stop; reject overshoot or post-handoff contact.
  Candidate 192 produced two clean index-3 runs and one index-3 tiny ID-1002
  contact. The contact run used four half-second stages, reached
  `9.60/2.42/4.02 m`, then fresh `13.15/3.53/0.35 m` vertical alignment after
  hover resumed. Persistent 3-4.5 m right error blocked forward approach.
  Reject `4.5 m` standalone; Candidate 193 changes only lateral gain
  `0.02 -> 0.03` under the same `0.10 rad` cap and all other exact safeguards.
  Run fixed maximum five under
  `competitive_r1_gate4_any_edge_delay3_close45_latkp03_batch_193`, no tuning,
  finish-stop; reject overshoot or contact.
  Candidate 193 produced two clean index-3 runs and no contact, but neither
  recreated Candidate 192's four-stage/vertical state; best delayed geometry
  was only `15.71/2.58/7.49 m`. The lateral bracket is therefore safe but
  untested at its target geometry. Candidate 194 is one exact tag-only maximum-
  five activation retry under
  `competitive_r1_gate4_any_edge_delay3_close45_latkp03_retry_batch_194`, no
  tuning, finish-stop; require comparable four-stage evidence or finish and
  reject any contact.
  Candidate 194 passed official index `3` cleanly in `5/5`, with zero contact
  or invalid state, but produced no finish and zero delayed cropped-edge
  selections/overrides in every run. Four runs used one safe half-second stage;
  closest delayed raw aperture families stayed roughly `19.59-21.33 m`
  forward and `18.35-19.73 m` laterally displaced, so Candidate 192's
  near-finish state was not recreated. Close Candidates 193/194 and the
  `kp=0.03` scalar-retry branch. Retain the `5/5` three-gate-prefix evidence and
  require a new observable acquisition/association mechanism before another
  final-gate batch.
  Candidate 195 restores the accepted Candidate-188/178 `kp=0.02`/right-bound
  `4.0 m` parent and adds one disabled-by-default observable dropout yaw freeze.
  When enabled, a missing/rejected associated final pose holds current
  quaternion yaw instead of the stale absolute setpoint; every associated-pose,
  roll, brake/hover, staged-pulse, range, prefix, and finish rule remains exact.
  Run fixed maximum five under
  `competitive_r1_gate4_dropout_yaw_freeze_batch_195`, no tuning, finish-stop;
  require exercised yaw arrest plus reacquisition/improved geometry and zero
  contact/invalid/health fault.
  Configuration audit supersedes the Candidate-194/195 final conclusions: both
  launch commands omitted `PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX=3`, leaving
  the motion filter active and final edge preference disabled. Their clean
  index-3 runs remain prefix/safety evidence, but are not intended-parent final-
  mechanism evidence. Candidate 196 restores raw index `3` and re-runs exact
  Candidate 193 (`kp=0.03`, `4.5 m`, delayed selector, yaw freeze disabled)
  under tag
  `competitive_r1_gate4_any_edge_delay3_close45_latkp03_rawfix_batch_196`,
  maximum five, no tuning, finish-stop. Require raw pose/observation agreement
  and nonzero selector activation before judging the branch.
  Candidate 196 proved the corrected raw path in two clean index-3 runs with
  delayed edge selections/overrides `26/21` and `19/12`; raw pose and policy
  observation agreed. Neither run pulsed or finished, and best geometry was
  `20.093/2.951/9.952 m`, just outside the close range. Three attempts failed
  before handoff. Close Candidates 193/196 and the exact `kp=0.03` retry.
  Candidate 197 restores Candidate-188/178 raw-index-3 `kp=0.02`/right-bound-
  `4.0 m` control and enables only observable dropout yaw freeze under tag
  `competitive_r1_gate4_raw_dropout_yaw_freeze_batch_197`, maximum five, no
  tuning, finish-stop. Require exercised raw-dropout yaw arrest with safer
  retention/geometry and zero post-handoff contact.
  Candidate 197 passed three clean index-3 final phases with raw yaw freeze
  exercised on all `171/171` sampled dropout frames and no post-handoff
  contact. One run used two safe stages and reached `16.94/2.62/8.18 m`; another
  repeated the detector-quantized `20.093/2.983/9.952 m` state, `0.093 m`
  outside admission. Retain yaw freeze. Candidate 198 changes only both staged-
  descent range ceilings `20.00 -> 20.25 m` under the exact parent, tag
  `competitive_r1_gate4_raw_yawfreeze_range2025_batch_198`, maximum five, no
  tuning, finish-stop; require safe activation at the measured quantized state
  and reject any post-handoff contact.
  Candidate 198 safely exercised `20.25 m`: in one of two clean final phases,
  the second half-second stage started at `20.093/2.260/9.890 m`; a fresh later
  frame reached `19.20/4.11/9.45 m`, just outside the close-right gate. Retain
  the range. Candidate 199 changes close right `4.00 -> 4.25 m` and caps stages
  at three, admitting measured `4.11-4.18 m` evidence while excluding the
  rejected `4.5 m`/four-stage branch. Run tag
  `competitive_r1_gate4_raw_yawfreeze_range2025_right425_max3_batch_199`,
  maximum five, no tuning, finish-stop; require safe third-stage progress and
  zero post-handoff contact.
  Candidate 199 was prefix-limited: four Gate-3 failures and one clean final
  phase with no pulse; its closest cropped state `20.57/4.37/10.16 m` remained
  outside both bounds. Candidate 200 is one exact tag-only maximum-five retry
  under
  `competitive_r1_gate4_raw_yawfreeze_range2025_right425_max3_retry_batch_200`,
  no tuning, finish-stop. Close `4.25 m` if no eligible third stage occurs.
  Candidate 200 safely executed three stages and reached collision-free raw
  poses `2.98/-0.49/0.84 m` then `3.62/0.15/1.12 m`, but normal forward control
  never activated because the identity jump was rejected and the low-gate
  close re-anchor excludes `down<=1.5 m`. Candidate 201 adds only a disabled-
  by-default one-use aligned crossing re-anchor: after the delay, require a
  coherent `0.125 s` raw cluster with forward `<=5 m`, `|right|<=1 m`, and
  `|down|<=1.5 m`; then normal `+0.06` aligned pitch applies. Test/default-check,
  then run `competitive_r1_gate4_crossing_reanchor_batch_201`, maximum five, no
  tuning, finish-stop; reject off-center admission or contact.
  Candidate 201 produced three clean final phases but no raw pose closer than
  `16.0 m`, so the crossing re-anchor was unexercised. Candidate 202 is one
  exact tag-only maximum-five retry under
  `competitive_r1_gate4_crossing_reanchor_retry_batch_202`, no tuning, finish-
  stop. If no `<=5 m` state recurs, stop live sampling and retain the mechanism
  only as guarded/test-proven.
  Candidate 202 produced four clean final phases but no pose closer than
  `16.30 m`; stop rare-state sampling and retain crossing re-anchor only as
  default-disabled/test-proven. Candidate 203 adds one disabled-by-default
  option to set pitch `0 rad` only during an already-authorized bounded descent
  pulse, removing the simultaneous `-0.12 rad` backward brake without changing
  pulse authorization, thrust, yaw/roll, association, range, maximum-three cap,
  prefix, or finish rules. Test first, then run
  `competitive_r1_gate4_level_pitch_pulse_batch_203`, maximum five, no tuning,
  finish-stop; require closer safe retention/progress and zero contact.
  Candidate 203 safely exercised level pitch during two stages but reached only
  `18.38/3.19/9.11 m`; reject and restore disabled. Candidate 204 produced four
  clean official-index-3 runs and one index-3 collision (ID `1001`, impact
  `5.5551`) after three stages, with no finish/crossing sample and best clean
  geometry only `9.29/2.10/5.18 m`; reject the unguarded expanded window.
  Candidate 205 retains Candidate 204 exactly and adds only default-off
  `PUFFER_FINAL_GATE_DESCENT_MAX_ABS_RIGHT_M=4.5`, because pulse admission
  previously checked forward/down but not right error. Run
  `competitive_r1_gate4_pulse_right45_guard_batch_205`, maximum five, no tuning,
  finish-stop; require safe useful activation/progress and zero contact.
  Candidate 205 produced two clean index-3 runs and three prefix failures; both
  clean runs used two in-corridor stages with zero contact, no crossing/finish,
  and best `13.94/3.07/6.45 m`. Retain the guard. Candidate 206 changes only
  cooldown `1.0 -> 0.375 s`, admitting the measured fresh
  `16.62/4.08/8.00 m` state `0.407 s` after stage 2 while retaining the lateral
  guard and maximum-three cap. Run
  `competitive_r1_gate4_guard_cooldown0375_batch_206`, maximum five, no tuning,
  finish-stop; require safe third-stage progress and zero contact.
- The `gate_max_acquire_range_m=50 m` FSM probe was collision-free but stalled
  at official index `1` and is rejected. Its parent default is restored to
  `100 m`; detailed evidence is in
  `docs/competitive_lap_run_ledger_2026-07-16.md`.

Decision as of 2026-07-12:

- Stop per-gate live gain sweeps. Gate 1 is a proven deterministic bootstrap; repeated gate-2 FSM variants remained at official index 1 and collided with the same environment platform.
- Train gate-2 behavior at native throughput, promote checkpoints through deterministic native evaluation, and spend official-simulator cycles only on promoted checkpoints.
- Keep the simulator process alive and use MAVLink `31000` between official attempts. A collision terminates the rollout immediately.
- The policy action contract is absolute pitch/roll/yaw plus hover-centered thrust. Native mode 2 closes those attitudes through the same measured v3385 body-rate plant used by the live runner.

Decision as of 2026-07-13:

- The hybrid FSM → policy handoff is no longer the target architecture. It remains useful for demonstrations, reset-state measurement, and diagnostics only.
- Train a single recurrent policy at native throughput with official-observable inputs. Absolute gate positions may be used internally for simulation and training rewards, but are never policy inputs.
- Use a full-course-first curriculum: every retained trajectory starts at the
  official origin with all four ordered gates and genuine recurrent history.
  Wide full-course apertures provide early terminal credit; then radius,
  close-range dropout, and course jitter are annealed without segment resets.
- The native backend must preserve recurrent state across rollout chunks and
  gates, reset only terminated agents, and train sequences from captured
  rollout-start state. This contract was corrected and partition-invariance
  tested on 2026-07-13.
- The native continuation is machine-driven and atomically resumable. It screens
  before training, uses bounded 0.5M corrections after bootstrap, refuses to
  chain centimeter-only incomplete children, and retains a stage only at exact
  1024-episode success `>=0.90` and crash `<=0.10`. Final promotion remains an
  exact 4096-episode screen with every gate radius set to `0.75 m`.
- Native promotion requires full-start ordered course completion before any new official attempt. See `docs/full_policy_execution_prompt.md`, `docs/full_policy_iteration_report_2026-07-13.md`, and the corrected true-aperture report `docs/full_policy_iteration_report_2026-07-14.md`.

Decision as of 2026-07-14:

- Correct the target-aperture contract: the official `1.5 m` value is the gate's
  inner width. Native `gate_radius` is a radial center-crossing tolerance, so
  promotion geometry is `0.75 m` on every gate. All earlier radius-1.5 results
  are relaxed-curriculum evidence, not official-geometry promotion evidence.
- The radius-1.5 checkpoint that scored `3967/4096` native completions was tested
  once officially and collided with the left post of gate 0. At closest approach
  its camera-relative right displacement was approximately `0.7567 m`, directly
  confirming the corrected half-width boundary. Preserve the artifact at
  `logs/sitl/competition_smoke_policy_full_policy_native_promo_20260714_02_attempt_001.json`
  and do not retry that checkpoint unchanged.
- Preserve the old true-aperture Gate-3 lineage as diagnostic evidence only.
  Its recurrent state was entangled: local and global corrections improved the
  last-gate miss by damaging earlier gates. It is not the active continuation.
- Do not spend another official cycle until a single full-start recurrent policy
  passes all four gates at `0.75 m` and an exact deterministic promotion report
  proves `n == 4096`, success `>=0.90`, and crash `<=0.10`.

Later transfer-parity decision as of 2026-07-14:

- Official full-policy attempt 004 passed gate 0, advanced to
  `active_gate_index=1`, and timed out without collision after losing the
  associated gate-1 image near `4.08 m`. Native training and evaluation must
  model close-range observation loss from gate index 1 at `4.25 m`; the former
  dropout-from-index-2 promotion evidence is superseded.
- The full ordered gate map is present internally in native `env.gates[]` for
  dynamics, rewards, and domain randomization. Absolute gate positions remain
  prohibited policy inputs. The policy boundary is active-gate relative
  pose/motion plus official-observable attitude, race phase, elapsed time, and
  previous action.
- Phase-local corrections recover exact two- and three-gate success above 95%,
  but they alter recurrent state and break the following gate. Stable global
  PPO improves the final-gate miss only by degrading the solved prefix. The
  retained-parent screens rejected all such children; the former true-aperture
  radius state is historical evidence, not the active continuation.
- Stop incremental surgery on the entangled checkpoint. Train a clean single
  recurrent policy from the corrected observation contract using native 1024-
  agent, full-course temporary-aperture curricula for terminal credit. Retention
  always uses a deterministic full-start screen. See the current state and
  continuation rules in `docs/full_policy_execution_prompt.md`.
- The clean-start path is now implemented in
  `scripts/train_full_policy_clean_curriculum.py` with its declarative schedule
  in `config/full_policy_clean_curriculum.json`. It starts without a source
  checkpoint, separates easier training settings from exact target settings,
  persists atomic resume state, and cannot promote a child unless an exact
  1024-episode screen reaches success `>=0.90` and crash `<=0.10`. The final
  v3385 target still requires an exact 4096-episode promotion. Use
  `--max-stages 1` for bounded, resumable training cycles.
- The active full-course bootstrap passed radius `18 m` at exact-1024 success
  `1.0`, crash `0.0`, four ordered gates, and completion time `4.866663 s`.
  Checkpoint
  `checkpoints/drone_race_full_policy_stage_d_gate4/1784059094357/0000000003997696.bin`
  is retained. Two bounded 0.5M children at the pending radius-15 stage reduced
  final-gate radial miss `15.961093 -> 15.844692 -> 15.673779 m` without a
  crash; the latest working checkpoint is
  `checkpoints/drone_race_full_policy_stage_d_gate4/1784059312590/0000000000491520.bin`.
  The authoritative state is
  `logs/drone_race_full_policy_stage_d_gate4/full_policy_coursewide_curriculum_state.json`.
  Neither checkpoint is an official-simulator candidate.

Decision as of 2026-07-16 — competitive qualifier path (supersedes the July 12-14
single-recurrent-policy architecture requirement):

- The product target is a **competitive valid qualifier lap**, not merely native
  success or course completion. Official gate progress and time through the
  TS-002 interface are authoritative.
- The official rules require autonomous flight but do not require an end-to-end
  neural policy. The primary path returns to the existing inspectable
  `course-fsm`/visual-servo stack: camera gate/guidance perception, camera/IMU
  estimation, race-index-aware acquire/align/approach/commit/pass/brake/
  reacquire phases, and gyro-closed body-rate/thrust control. A compact learned
  detector or residual is permitted only after measured component failure.
- The installed official simulator is exactly
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`.
  Its UI shows only `AI-GP Virtual Qualifier R1`; the user confirmed on
  2026-07-16 that only VQ1 is available. R1 is the active competitive target;
  do not reopen VQ2 discovery without new access or a new package.
- Finish-lock decision (2026-07-16): the official FAQ both permits attempts
  throughout the qualification window and describes thousands of completed
  simulator runs used for refinement before stating that the time is locked
  once that run passes all gates. Together with fastest-valid-time scoring,
  this supports a per-attempt timer lock rather than an irreversible first team
  submission. Source: https://www.theaigrandprix.com/. No public VQ1
  leaderboard/cutoff was located. One deliberate local first finish is
  authorized; prove reset-after-finish before the next flight.
- MAVLink command `31000` remains the normal in-place race reset. Preserve the
  healthy v3385 process; relaunch only for recovery. Only official
  `active_gate_index` counts as gate progress, and collision abort is immediate.
- The frozen Jacobian-CMA neural-policy branch completed its bounded search
  without an official Gate-2 crossing and is closed. Do not repeat broad PPO,
  BC/DAgger, policy-space CMA, or unchanged failed configurations. Optimize a
  small named deterministic-controller parameter vector only after reliable
  traversal exists.
- `docs/competitive_lap_execution_prompt.md` is the active copyable execution
  mandate. `docs/course_completion_strategy.md` and
  `docs/full_policy_execution_prompt.md` are retained historical policy-track
  records, not active initialization instructions.

Decision as of 2026-05-21:

- The competition-winning path is a spec-compliant SITL controller first, not native `drone_race` score optimization in isolation.
- Native PufferLib remains valuable as a high-throughput training, regression, and teacher-signal backend, but native success is not promotable unless the behavior transfers through the MAVLink/SITL adapter and official telemetry/camera streams.
- The first competition-facing baseline should be conservative and inspectable: TS-002 camera ingestion, gate detection/pose estimation using known pinhole intrinsics and gate dimensions, telemetry-based stabilization, and bounded MAVLink setpoints at a verified command rate below `100 Hz`.
- Learned policies may augment perception, planning, or control after this baseline exists; they should not be the only path to a valid autonomous run.
- Speed optimization is explicitly secondary to reliable, ordered, spec-compliant gate traversal with correct command-rate, heartbeat, telemetry, and vision-health reporting.

## Current Completion Snapshot

Current qualifier status as of 2026-07-16:

- The installed v1.0.3385 UI exposes only R1, and the user confirmed VQ1 is the
  only available event. R1 is the active target. Unknown finish-time lock
  semantics block only a full-course finish, not the current Gate-2 prefix.
- Gates 1-3 now have a clean accepted R1 prefix:
  `competitive_r1_reproduce_gate3_r135_dropout_084` reached official index `3`
  without collision or health violations. Gates 4-6, full-lap reliability, and
  a competitive official time remain unproven.
- Candidate 087 retained that clean prefix and reduced the raw Gate-4 bearing
  to `0.1914 rad` while holding it `25.26 m` ahead, without collision or finish.
  Corrected raw-pose/yaw-sign alignment is retained; controlled approach and
  crossing remain unproven.
- Candidate 098 first produced `3/3` clean Gate-3 passes, but larger follow-up
  invalidated a reliability claim: Candidate 101 added a high-speed miss, and
  Candidate 108/110 reliability batches passed only `3/5` and `2/5` with their
  respective close-range speed brakes. Gate-3 entry variance is again the
  active blocker. Candidate 098 is the retained simple parent; Gate 4 remains
  paused except for already recorded Candidate 087/100 evidence, and every
  future final diagnostic must keep the 20 m no-crossing guard.
- The active implementation path starts from the recovered Stage-C recurrent
  policy because it owns the strongest official prefix. Deterministic
  `course-fsm` remains a diagnostic/fallback controller, not the retained best.
- Native support-track status: N033 is the first accepted single recurrent
  six-gate policy. N058 is the retained fixed-straight checkpoint parent, with
  exact-512 success `412/512` at radius `0.95` and `455/512` at radius `1.0`,
  zero crash/timeout. N071 retains a mixed-aperture environment at `411/512`:
  gate indices 0/5 radius `0.75`, index 3 radius `0.875`, others `0.95`. N072's
  index-3 target curriculum peaked at `399/512` for radius `0.75` and is
  rejected. N073 restores direct-FP32 collector/native common-seed parity.
  N074's spline expert failed, N075 was invalidated for omitted collector
  settings, and the corrected N076/N077 screens reject Gate-4 feedback and a
  gate-indexed bias (`93/128` and `96/128` versus N058's `97/128`). N078-N084
  then accept a deployable native-index-2 action adapter and tighten that gate
  to radius `0.85`: retained N083 uses roll `-0.0125`, thrust `+0.00125`, and
  N084 scores exact `412/512` with zero crash/timeout. N085 fails at `0.825`;
  N086's proposed Gate-2 addition scores `407/512` and is rejected. N089's
  center-safe two-gate adapter scores `413/512`; N090-N093 are rejected.
  N094-N095 establish a six-gate nominal/randomized anchor mixture and isolate
  course-position RNG from vehicle initialization. N096's retained raw policy
  step `491,520` improves randomized success `9/128 -> 12/128` and retains
  fixed N071 at `413/512`, zero crash/timeout. Checkpoint:
  `logs/drone_race_full_policy_six_gate_bootstrap/ppo_n096_raw_anchor_mixture/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784286271692/0000000000491520.bin`
  (SHA-256
  `d580728192e42647607800083d7c6809f038173672f5033f6ee36fbdfbe68dd9`).
  N096 is the raw randomized-geometry curriculum parent; N089/N084 remains the
  tighter fixed-course adapter parent. N097's widened randomized aperture and
  N098's hidden-next-gate reward suppression both fail the randomized target
  gate (best `11/128` each), so all of their policy children are rejected and
  N099-N102's held-out elite self-imitation ladder also fails; even phase-local
  output isolation tops out at `11/128` and `4.164062`. N096 remains retained.
  N103 phase-priority replay is also rejected: best completion is `11/128`
  with `4.15625` average gates and best progress is `4.1875` with `10/128`,
  all zero crash/timeout. N104 isolated learning to the legal active-Gate-4/5/6
  encoder columns while freezing the solved three-gate prefix and every shared
  recurrent/decoder parameter. The rebuilt-collector rescreen promotes final
  step `1,048,576`: randomized common-128 improves from N096's `12/128` and
  `4.1953125` to `17/128` and `4.21875`, with zero crash/timeout, while exact
  fixed-N071 retention stays `413/512`, `5.21484375`, zero crash/timeout.
  N105 amplification is canceled without a valid screen; its premise was the
  stale collector's false flat result.
  Offline callable replay through Linux and the actual Windows interpreter now
  passes across all six phases with maximum action/state error
  `1.73e-6/7.16e-7`; live observation-domain and deployment-hash parity remain.
  N106 all-gate radius-0.75 screening fails at `62/128`, with every miss
  localized to active Gates 2/3, so official deployment remains blocked.
  N107 improves that target to `74/128` but fails exact retention at `284/512`;
  N108 falls below N107 at `64/128`; N109 briefly improves common-128 to
  `76/128` but fails exact-512 at `288/512` versus required `410/512`. All
  N107-N109 children are rejected, direct PPO continuation is closed, and N104
  remains the native parent. Packaging/hash, fresh-reset recovery, and refreshed
  passive parity now pass for retained N104 with all `257/257` recurrent steps
  reproduced at max action error `4.35e-7` and zero control-state mutations.
  Bounded official Candidate 001 passes index `1` at `5.032423496 s`, stops at
  controller elapsed `5.016 s`, and is clean on every safety/transport metric;
  a passive follow-up proves disarm. One unchanged attempt bounded to official
  index `2` is pre-registered next. It does not authorize a Gate-3 extension or
  full lap.
  This is curriculum evidence only: the two late gates
  are synthetic
  straight-line extensions, and no official flight is authorized until every
  aperture/domain and native-to-live promotion gate passes.

Completed:

- `native-v4-drone-port` is now the GitHub default branch and active source of truth.
- Native v4 baseline exists for `drone` and `drone_race`, including C bindings, CPU smoke/eval paths, local regression checks, and Linux CUDA/NCCL validation on Vast.ai.
- `drone_race` uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate kinematics.
- Race and hover observations are aligned at `23` features, enabling hover-to-race native checkpoint warm starts.
- Staged native race curriculum exists for H1, R1, R3, and R4, with promotion checks that stop on failed thresholds.
- Reproducible checkpoint eval artifacts are supported via `scripts/eval_drone_race_checkpoint.py` with paired JSON/CSV outputs.
- Linux GPU support track ran on Vast RTX 4090: native precheck/validation passed, short `drone_race_competition` train completed, and checkpoint eval artifacts were produced (`logs/drone_race_competition_eval.{json,csv}`).
- Useful native checkpoints exist through H1 and R1; R3 has a reproducible partial-success checkpoint but is not promotion-ready.
- Q8/mixed-precision PufferNet inference scaffolding, export, latency/drift benchmarking, and closed-loop hover comparison harnesses exist.
- MAVLink/SITL adapter scaffold, local dry-run path, TS-002 command-rate validation, and testable inbound telemetry parser/reporting scaffold exist.
- First-pass competition-shaped policy contract exists for native training and SITL wiring: `config/drone_race_competition.ini` trains the compiled `drone_race` backend with TS-002-shaped observations and body velocity/yaw-rate policy actions, while `scripts/drone_policy_contract.py` defines the matching Python decode path to MAVLink local-NED setpoints.
- First-pass TS-002 visual-servo geometry helper exists: detected square-gate corners can be converted into relative body/NED pose and conservative local-NED velocity/yaw commands.
- Local competition smoke entrypoint now exists (`scripts/drone_sitl_competition_smoke.py`): it wires MAVLink telemetry, TS-002 camera ingest, first-pass square-gate detection, visual-servo or policy-contract command emission, and deterministic JSON/CSV reporting in one run loop.
- First-pass square-gate detector now exists (`scripts/drone_gate_detector.py`) and is integrated into the camera/perception boundary used by the smoke entrypoint.
- Windows-local official simulator runtime is now validated on the RTX 3070 workstation with a lightweight `.venv-win` Python stack (`pymavlink`, `numpy`, `opencv-python` only for SITL). The observed simulator topology is MAVLink v2 telemetry from `127.0.0.1:14560` to client port `14550`, TS-002 camera packets on client port `5600`, and simulator-owned sockets on `14560` and `5601`.
- On 2026-06-04, Windows-local official simulator smoke runs validated telemetry+camera ingestion and controller-side visual gate-pass heuristics: `logs/sitl/competition_smoke_gate1_windows_local_drain.json` reports nonzero telemetry/camera, `command_rate_violations=0`, `telemetry_dropouts=0`, `completed_frames=5183`, and `detector_detections=2282`.
- On 2026-07-02, official simulator `v1.0.3379` replaced the unsupported `v1.0.3364` runtime on this workstation. The supported executable is `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3379\AIGP_3379\FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe`; the old `v1.0.3364` app now shows a service/version error and is not usable for promotion evidence.
- Official simulator `v1.0.3379` still uses the Windows-local topology: simulator-owned sockets on `14560` and `5601`, client MAVLink on `14550`, and client camera on `5600`. The current reliable operator flow is clean launch, submit saved login, select `AI-GP Virtual Qualifier R1`, click `RACE`, then start the controller locally from `.venv-win`.
- The `v1.0.3379` bundled `PyAIPilotExample-v2` and live traffic confirm that `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and track geometry should not be assumed for current runs; live official validation receives `HEARTBEAT`, `HIGHRES_IMU`, `ACTUATOR_OUTPUT_STATUS`, and encapsulated race status. Yaw/local position must therefore remain optional inputs.
- Superseded for v3379 actuation: early local-NED velocity trials (`official_gate1_validation_summary_3379_local_ned_*`) reached controller-side pass heuristics but never advanced official race status; velocity setpoints are now confirmed dead. **Working control path:** `SET_ATTITUDE_TARGET` rate-demand + thrust (`visual-servo-angles`, `policy-attitude`).
- The SITL smoke report now emits `approach_diagnostics` with closest range, confidence, body-vector estimate, visual-servo command, and bounded pose/command samples. The first diagnostic run (`logs/sitl/official_gate1_validation_summary_3379_approach_diag_001.json`) showed the controller never got inside the vision pass threshold: closest estimated camera range was `2.611632m` with low confidence `0.111434`, and official race status remained gate `0`.
- Early command-semantics probes on `v1.0.3379` did not prove first-gate advancement. `--visual-servo-desired-standoff-m 0.0` still missed (`logs/sitl/official_gate1_validation_summary_3379_standoff0_001.json`, closest `2.681855m`). Constant forward local-NED policy mode also missed with both yaw-active and official-example yaw-ignored masks; the yaw-ignored run recorded collisions and actuator outputs near `0.05` (`logs/sitl/official_gate1_validation_summary_3379_constant_forward_ignore_yaw_001.json`).
- The SITL adapter now supports the simulator example's `SET_ATTITUDE_TARGET` body-rate command shape: attitude ignored, body roll/pitch/yaw rates active, and normalized thrust active. `scripts/drone_sitl_competition_smoke.py`, `scripts/run_official_gate1_validation.py`, and `scripts/watch_official_gate1_validation.py` expose this as `--control-mode attitude-rates`, plus a camera-steered `--control-mode visual-servo-attitude` that maps detected gate pose to body pitch/yaw rates and thrust. `scripts/run_sitl_control_calibration.py` runs a fixed velocity/attitude-rate/steered-attitude matrix and emits per-case JSON/CSV plus a summary ranking official gate progress, closest range, confidence, collisions, and command-rate compliance.
- First live `SET_ATTITUDE_TARGET` diagnostics on `v1.0.3379` prove the simulator obeys body-rate/thrust commands. `logs/sitl/control_calibration_live_attitude_t060_probe_01_attitude_rates_pitch_negative_t060.json` reached `closest_range_m=1.251098`, `highest_confidence=0.970796`, zero collisions, and no command-rate violations in an 8s run, but official `active_gate_index` stayed `0`. A longer non-reset run (`logs/sitl/official_gate1_validation_attitude_t060_long.json`) produced multiple controller-side gate-pass events and real actuator outputs near `0.59/0.61`, but official race status still did not advance. The follow-up steered-attitude artifact (`logs/sitl/official_gate1_validation_visual_servo_attitude_001.json`) started from the disturbed post-diagnostic state and saw no gate detections, so it is not promotion evidence; rerun it only after a clean manual race reset.
- Fresh-race arming now requires receiving an inbound simulator heartbeat before sending `MAV_CMD_COMPONENT_ARM_DISARM`. `scripts/drone_sitl_competition_smoke.py` now waits for a pre-arm heartbeat by default, sends configurable arm attempts, and records `arm_on_start`, `arm_commands_sent`, and `prearm_heartbeats_seen` in `control_inputs`. The first armed visual-servo-attitude run (`logs/sitl/official_gate1_validation_visual_servo_attitude_prearm_001.json`) confirmed the fix: `prearm_heartbeats_seen=1`, `arm_commands_sent=5`, `base_mode=193`, `system_status=4`, and actuator outputs around `0.71-0.79`.
- The armed visual-servo-attitude run produced controller-side pass events but still missed official progress: `ordered_gate_passes=2`, first controller-side pass at `6.485s`, closest range `1.598883m`, zero command-rate violations, zero telemetry dropouts, but official `active_gate_index=0` and `last_gate_race_time=-1`. The controller is now applying thrust; the remaining blocker is approach geometry/control, not transport or arming.
- The yaw-search `visual-servo-attitude` path can now reacquire an off-screen first gate. `logs/sitl/official_gate1_watch_yaw_search_after_relaunch_latest.json` waited through menu/no-camera states, then ran after manual R1/RACE start with nonzero streams, `prearm_heartbeats_seen=1`, `base_mode=193`, `command_rate_violations=0`, `telemetry_dropouts=0`, `detector_detections=2772`, and `ordered_gate_passes=6`. It still failed official promotion (`official_active_gate_index=0`, `last_gate_race_time=-1`) and recorded one collision, so the result proves search/reacquisition and actuation, not gate completion.
- The detector is locking onto the real first gate in the current v3379 scene. `logs/sitl/after_yaw_search_after_relaunch_snapshot_latest/first_detection_annotated.jpg` shows the annotated square on the gate, with the vehicle low and obstacle-bound. The remaining controller problem is to climb/yaw/center before pitching forward through the gate plane.
- `scripts/drone_sitl_competition_smoke.py` now records the actual attitude target used for each approach diagnostic sample (`actual_command`) and supports opt-in `visual-servo-attitude` yaw-search plus center-gated forward motion. Use `--attitude-servo-search-yaw-rate-rad-s`, `--attitude-servo-forward-yaw-tolerance-rad`, `--attitude-servo-forward-z-tolerance-m`, and `--attitude-servo-uncentered-forward-scale 0.0` to rotate/climb until the gate is centered before applying forward pitch.
- `scripts/run_official_gate1_validation.py` now performs a short pre-smoke race-start check. When the simulator is streaming camera but `race_start_boot_time_ms=-1`, it returns `blocked_race_not_started` without arming or sending controller commands. `logs/sitl/official_gate1_validation_race_start_check_latest.json` proves the fast path: MAVLink/camera were nonzero, `race_started=false`, `smoke=null`, and elapsed time was `2.260205s`.
- Windows race restart automation now has a safe wrapper: `scripts/start_windows_aigp_race.ps1` relaunches v3379, optionally clicks the saved login/R1/RACE screen coordinates, and then calls `scripts/wait_official_race_start.py`. The helper only reports success when `race_start_boot_time_ms>=0`; if the coordinate click path misses, it fails before controller commands are sent.
- Current lifecycle rule: `scripts/start_windows_aigp_race.ps1` now reuses a healthy simulator process and launches only when absent. `scripts/run_windows_course_fsm.ps1` is the one-command entrypoint. Normal race retries use MAVLink reset `31000`; process termination requires explicit `-ForceRelaunch`.
- The restart helper is live-validated. `logs/sitl/race_start_helper_full_relaunch_retry_test.json` shows a full process relaunch plus UI retry loop reaching `race_started=true`, `base_mode=193`, `system_status=4`, and `race_start_boot_time_ms=45563`. A failed coordinate pass correctly produced nonzero exit with `race_started=false`, so the safety gate works.
- The live detector now has a color-required aperture path for the v3379 red gate. `scripts/drone_gate_detector.py` first finds a red gate contour, then returns a substantial inner opening contour when visible instead of treating the red outer frame/panel as the scoring target. This removed the earlier gray-ceiling/grid false positives and eliminated the misleading local pass events caused by clipping a red gate edge.
- Aperture-detector strict runs prove the remaining first-gate issue is controller geometry, not connectivity or arming. `logs/sitl/competition_smoke_aperture_hybrid_latest.json` kept nonzero MAVLink/camera streams, `command_rate_violations=0`, `telemetry_dropouts=0`, `ordered_gate_passes=0`, and official `active_gate_index=0`; the closest aperture estimate was `3.595506m` with the aperture high-left in `logs/sitl/debug_gate1_aperture_hybrid_latest/closest_detection_annotated.jpg`.
- The best close-range aperture run so far is `logs/sitl/competition_smoke_aperture_final06125_latest.json`: final approach latched at `3.922s` with trigger pose `range_camera_m=3.490909`, `image_center_px=[89.5,123.0]`, command `body_pitch_rate=-0.3`, `body_yaw_rate=-0.3`, `thrust=0.6`, and no command-rate violations, no telemetry dropouts, no collisions, but official race status still stayed at `active_gate_index=0`, `last_gate_race_time=-1`.
- The first loosened center-gated crawl attempt (`logs/sitl/official_gate1_validation_center_gated_crawl_latest.json`) still failed official progress with `official_active_gate_index=0`, but improved evidence: nonzero streams, `command_rate_violations=0`, `telemetry_dropouts=0`, zero collisions, `detector_detections=4820`, and closest range `1.580692m`. The miss occurred with high closest yaw error (`0.467369 rad`) while crawl was enabled, so the next attempt should keep the wider vertical tolerance but remove forward crawl unless yaw is centered.
- The armed local-NED constant-forward retest (`logs/sitl/official_gate1_watch_velocity_forward_armed.json`) is not a promotion result. It did start/move the race view, but the vehicle did not approach the first gate (`closest_range_m=11.958532`) and official gate progress remained `0`; treat local-NED constant forward as a poor primitive for gate traversal unless future axis sweeps show a better sign/frame mapping.
- Live TS-002 camera ingestion required draining multiple UDP chunks per smoke-loop iteration; `scripts/drone_camera_receiver.py`, `scripts/drone_sitl_competition_smoke.py`, and `scripts/run_official_gate1_validation.py` now support bounded camera packet draining via `--camera-max-packets-per-loop` (default `512`).
- The simulator-bundled `PyAIPilotExample` confirmed the MAVLink/camera endpoints, camera header, custom reset command `31000`, and race-status/track-info encapsulated data formats. The SITL adapter now parses official race status, local position, odometry, actuator output, and collisions.
- Superseded 2026-07-03: MAVLink reset command `31000` is now the preferred restart path on `v1.0.3379`. Live-verified behavior: the sim reboots in place (`sim_boot_time_ms` returns to ~0) and the race auto-starts about `3.3s` after boot with `race_start_boot_time_ms>=0`, healthy camera frames, and gate-1 detections. The earlier blank-camera/implausible-altitude auto-reset evidence was from the unsupported pre-3379 runtime and no longer applies. Per-attempt cycle time drops from ~60s (process/UI restart) to ~5s; `scripts/start_windows_aigp_race.ps1` is only needed after a full app/process restart.
- Confirmed 2026-07-03 on `v1.0.3379`: TRACK_INFO is never broadcast. After in-place resets and explicit request probes (`MAV_CMD_REQUEST_MESSAGE`, adjacent custom command IDs, legacy `REQUEST_DATA_STREAM`), zero `DATA_TRANSMISSION_HANDSHAKE` messages arrived and the only encapsulated packets were race status (type `1`). The active controller therefore follows observable course guidance and gate pixels; it does not require global gate coordinates.
- Attitude-plant discovery 2026-07-03 (falsifies earlier open-loop assumption): the `SET_ATTITUDE_TARGET` quaternion on `v1.0.3379` is a rate demand, not an absolute attitude. Gyro-measured per-axis response to a constant commanded angle: roll rate ~`+2.47x` commanded angle, pitch rate ~`-2.47x` (sign-inverted), yaw rate ~`+2.2x`. "Level hover holds attitude" only held because a level command is a zero-rate demand. Commanded yaw is NOT a yaw estimate; all earlier open-loop angle-servo circling/drift traces to this.
- State estimator now exists (`scripts/drone_state_estimator.py`): gyro AHRS plus vision-derived position/velocity diagnostics. Live probes showed the accelerometer carries no useful clean-flight translation information, so its position/map output is not the active course-navigation source.
- Closed-loop attitude control now exists and is live-validated: `attitude_loop_command()` wraps the rate-demand plant with the AHRS estimate and holds commanded roll/pitch/yaw within ~`0.01 rad` in hover, including yaw steps and steady forward pitch. `scripts/drone_sitl_competition_smoke.py --control-mode visual-servo-angles` uses this closed loop by default (`--no-angle-servo-closed-loop` to disable) and logs estimator calibration, trajectory samples, and landmark map per run.
- The detector now tags each detection `source=aperture|frame` and pose estimation scales the assumed physical width accordingly (`1.5m` inner aperture vs ~`2.7m` outer frame), removing the ~2x range flip between frames. The red-contour scorer no longer lets image position dominate, which fixes lock-ons to small distant red blobs above the real gate.
- First closed-loop gate-1 attempt (2026-07-03, `logs/sitl/agent_closedloop_1_smoke.json`): closest approach `5.9m` at `t=10s` — prior best across all earlier attempts was ~`16-18m` — before a tumble on final approach ended the run. The estimator mapped gate 1 at roughly `(20, 0.4, -7)m` NED from takeoff. Official `active_gate_index` still `0`.
- Gyro frame correction 2026-07-03: live probes (`scripts/probe_pitch_sign.py`) proved the `HIGHRES_IMU` gyro reports in the command/body frame directly — a pure yaw maneuver produces a pure z-axis gyro reading — so only the accelerometer rest direction carries the ~`17.8 deg` mount tilt. The estimator no longer mount-corrects gyro rates (doing so injected phantom cross-axis coupling). Re-measured plant gains with the fixed gyro: roll `+2.5`, pitch `-2.35` (sign-inverted), yaw `+2.05`, linear across `0.10-0.30 rad` commands.
- Camera-tilt falsification 2026-07-03 (`scripts/probe_camera_tilt.py`): the v3379 rendered camera is LEVEL, not tilted `20 deg` up as TS-002 documents. Measured from the pad without arming: gate 1 straight ahead at ~`23m` appears at image center (`+0.98 deg` elevation). The 20 deg correction in `camera_vector_to_body_ned` was manufacturing a fictitious `+8.8m` gate height on every detection, which explains the persistent climb-out-of-view failure of every prior approach. `TS002_CAMERA_UPTILT_DEG` is now `0.0`.
- Hover-thrust falsification 2026-07-03 (`scripts/probe_thrust_step.py`, `scripts/probe_thrust_curve.py`): normalized thrust liftoff sits between `0.16` (stays on pad) and `0.20` (climbs away); `0.30` already climbs hard and `0.5-0.6` is a rocket. All earlier tuning assumed hover ~`0.5-0.55` and therefore commanded massive permanent climb. Working band: hover ~`0.17-0.20`, min `0.18`, max `0.42` for the vertical servo.
- FIRST-GATE PROOF 2026-07-03 (`logs/sitl/agent_sprint_13_smoke.json`): official `active_gate_index` advanced `0 -> 1` with `ordered_gate_passes=1`, `acceptance_passed=true`, zero blockers. Closest approach `1.875m` at `t=6.2s`; the estimator mapped gate 1 at `(21.6, 0.3, -0.3)m` NED (realistic, vs the fictitious `-7m` altitude before the camera-tilt fix). Config: `visual-servo-angles` closed loop, hover thrust `0.27`, thrust band `0.18-0.42`, search thrust `0.28`, max pitch `0.06 rad`, speed cap `2 m/s`. Milestone 1 of the ladder is complete.
- Gate-1 reproduction 2026-07-03 (`logs/sitl/agent_sprint_{14,15,16}_smoke.json`, identical config, in-place resets): 3 of 4 total attempts passed officially (runs 13/14/16 `active_gate_index=1`, closest `1.88/1.93/1.98m`; run 15 missed at `2.40m` closest with `active_gate_index=0`). Every run still records `>=1` collision and long post-gate recovery episodes (`~900-1350` recovery ticks), so the pass is real but the post-gate behavior always ends the run — grounded/limp detection and gate-2 acquisition are the active work.
- Strategy reference: `docs/course_completion_strategy.md` is the concise active strategy; `docs/course_fsm_execution_prompt.md` is the copyable execution prompt.
- Track B infrastructure (2026-07): `interface_mode=2` sitl-plant native env, custom gate layout, `scripts/track_b_runner.py`, `config/drone_race_sitl_plant_gate1_fast.ini` (~200–280K SPS on RTX 3070 WSL). Observation parity in `scripts/drone_policy_contract.py` (`build_ts002_observation`). Transfer orchestration: `scripts/run_track_b_transfer.ps1`.

Not complete / blocking:

- **Course completion:** gate 1 is proven; gate 2+ traversal is not. The active fix is straight post-pass braking followed by visible cyan-guidance following. Global landmark mapping is not required for this milestone.
- **Track B transfer (milestones 2–3):** native `drone_race_sitl_plant_gate1_fast` training completed 10×1M chunks without promotion (`success_rate=0`, `crash=1.0`, dominant `crash_high`). Latest checkpoint `checkpoints/drone_race_sitl_plant_gate1_fast/1783365920143/0000000000917504.bin` is not transfer-ready. Policy transfer with wrong-geometry checkpoint failed at closest `2.76m` (`logs/track_b/state.json`, `track_b_race_live`).
- **Control-path truth on v3379:** `SET_ATTITUDE_TARGET` (rate-demand quaternion + thrust) is the only working actuation path. `SET_POSITION_TARGET_LOCAL_NED` produces no actuation (motors idle ~`0.05`). Submission baseline is gyro-closed `course-fsm`.
- R3/R4 native race promotion remains below thresholds (`crash=0.1225` best R3 eval). **Deferred** as primary path until Track A multi-gate stalls or Track B passes B1 transfer.
- Competition-shaped `drone_race_competition` (`interface_mode=1`, velocity/yaw actions) is implemented but not validated in live sim — velocity commands do not actuate v3379; prefer `policy-attitude` for transfer experiments.
- Native gate geometry still abstract vs TS-002 square opening (`1500×1500 mm` inner, `2700×2700 mm` frame).
- Q8/fake-quant not promoted.
- Offline cyan-guidance preflight and official gate-1 brake proof have not yet been run.

Historical blockers (resolved — do not re-open):

- Official first-gate race progress: **resolved 2026-07-03** (`agent_sprint_13`, 3/4 repro on sprints 13/14/16).
- Camera uptilt (`20 deg` documented vs level v3379 renderer) and hover thrust (`~0.5` vs `~0.17–0.20`): falsified and fixed.
- TRACK_INFO never broadcasts; the active controller follows visible guidance and gate pixels instead of building a global map.
- Connectivity, arming, heartbeat, command-rate, camera ingest, aperture detector: proven on v3379.

Historical blocking-work snapshot (superseded by the full-policy decisions and
work-continuity rule above; retain for provenance, do not execute its old next
actions):

- **Multi-gate SITL (Track A — primary):**
  - Current state: gate 1 official pass proven. `course-fsm` implementation and offline detector harness exist; live brake/gate-2 proof is pending.
  - Next action: validate cyan guidance offline, then prove gate-1 pass + straight brake + settle before attempting gate 2. Use in-place reset `31000` between attempts.
  - Acceptance signal: `official_active_gate_index>=2` on v3379 with `acceptance_passed=true`, then progress toward full course.
- **Visual-servo competition baseline:**
  - Current state: gate-1 config validated (hover `0.27`, thrust `0.18–0.42`, max pitch `0.06 rad`, speed cap `2 m/s`). Velocity setpoints are a dead end on v3379.
  - Next action: use cyan guidance for broad route tracking and the red aperture only for precision alignment/approach.
  - Acceptance signal: ordered official gate advancement through at least two gates without human input.
- **Track B native transfer (support):**
  - Current state: `drone_race_sitl_plant` `interface_mode=2` with identified rate plant; gate-1 at official pose `(21.6, 0.3, 0.0)m`; fast config ~200–280K SPS on 3070. Training failed to learn (`crash_high` dominant).
  - Next action: paused until official gate 2. When resumed, fail fast after two catastrophic evals and re-transfer only after native `success_rate>=0.85`.
  - Acceptance signal: B1 — `official_active_gate_index>=1` with `--control-mode policy-attitude` and gate1-aligned checkpoint.
- **R3/R4 native race reliability (deferred support):**
  - Current state: best R3 `success_rate=0.8775`, `crash=0.1225` — not promotion-ready. R4 blocked.
  - Next action: resume only on Vast 4090 when Track A or Track B stalls; do not block multi-gate SITL work.
  - Acceptance signal: R3 `success_rate>=0.85`, `crash<=0.10` on 4k deterministic eval.
- Spec-compliant SITL baseline:
  - Current state: **gate-1 complete.** Windows-local v3379 topology validated; MAVLink reset `31000` is fast-restart path; race-status judge wired.
  - Next action: multi-gate loop (see above).
  - Acceptance signal: milestone 4 progress (`official_active_gate_index` advances through course).
- MAVLink telemetry and controller contract:
  - Current state: heartbeat, setpoint, and inbound telemetry parsing validated on v3379 (`HEARTBEAT`, `HIGHRES_IMU`, `ACTUATOR_OUTPUT_STATUS`, race status). Working control path is `SET_ATTITUDE_TARGET` rate-demand + thrust via `visual-servo-angles` or `policy-attitude`. `scripts/drone_policy_contract.py` provides aligned TS002 observations for policy transfer.
  - Next action: live-validate the implemented `course-fsm` transition on official `active_gate_index`.
  - Acceptance signal: deterministic run reports with command-rate compliance and official gate progression through multi-gate runs.
- Local SITL eval runner:
  - Current state: `scripts/drone_sitl_competition_smoke.py` can connect to the official Windows simulator endpoint, run visual-servo or policy-contract command loops, and emit deterministic JSON/CSV artifacts with telemetry/camera health metrics.
  - Next action: document or automate fixed course/start-condition reset control, then add ordered multi-gate/completion extraction so it can gate promotion directly.
  - Acceptance signal: one-command SITL smoke/eval produces success, ordered gate passes, completion time, crash/invalid-run status, command rates, and telemetry dropout fields.
- Vision ingestion:
  - Current state: TS-002 defines a `30 Hz`, `640x360` JPEG-over-UDP stream on default/client port `5600`, chunked as a 24-byte little-endian metadata header plus JPEG payload. The live simulator sends high-rate chunk traffic; the receiver now supports bounded packet draining so smoke runs can reconstruct current frames instead of falling behind stale chunks.
  - Next action: keep packet-drain defaults tuned for live traffic, sync `sim_time_ns` with MAVLink `TIMESYNC` if/when the simulator emits it, and add regression coverage for high-rate chunk draining.
  - Acceptance signal: camera adapter publishes reconstructed frames with `sim_time_ns`, applies TS-002 pinhole intrinsics/extrinsics, and supports separate telemetry-only vs telemetry+vision eval tracks.
- Native geometry and gate fidelity:
  - Current state: native course logic is useful for curriculum training, but it has not been audited against TS-002 chassis, gate frame, inner opening, obstacle, and boundary dimensions.
  - Next action: compare native gate crossing/collision helpers, obstacle definitions, and course geometry against the TS-002 drone chassis `280x280x160 mm`, gate outer `2700x2700x260 mm`, and gate inner `1500x1500x260 mm` dimensions. Replace point/circular gate success checks with a spec-faithful clearance model before treating native success as transferable.
  - Acceptance signal: native and SITL geometry tests cover gate opening, frame depth, drone chassis clearance, obstacle/boundary collisions, and ordered crossing validity.
- Official simulator runtime topology:
  - Current state: native training and Vast.ai validation remain Linux-oriented, while the official DCL simulator runs on Windows 11. The validated local topology is Windows simulator plus Windows-local Python `.venv-win` controller on the RTX 3070 workstation; earlier WSL relay attempts were blocked by firewall/VPN routing and are not the preferred path.
  - Next action: document the Windows-local runbook (`.venv-win`, MAVLink client port `14550`, camera port `5600`, simulator reset requirement) and keep Linux/Vast only for training/checkpoint generation.
  - Acceptance signal: documented Windows-local topology can run the controller against the official simulator without assuming a Linux-hosted DCL simulator.
- Q8/fake-quant promotion:
  - Current state: Q8 export/runtime and benchmarks exist, but action drift and closed-loop race parity are not promotion-ready.
  - Next action: defer fake-quant RL until FP32 closed-loop behavior is reliable through the SITL/camera/telemetry path; then add QAT or post-training calibration against closed-loop hover/race metrics.
  - Acceptance signal: Q8 policy matches FP32 closed-loop SITL success/crash metrics within agreed tolerance and meets worst-case latency requirements.

## Competition-Aligned Requirements

- Gate recognition from provided sensor/visual streams.
- Precision drone control (thrust/orientation/rates) with reliability-first safety behavior.
- Efficient planning/navigation under realistic vehicle limits.
- No manual control and no hardware advantages.

## Official Virtual Qualifier Spec Alignment

Reference: `docs/260508_Technical_Spec_0002.pdf` (`VADR-TS-002`, issue `00.02`, cover date `2026-05-08`; revision history `2026-05-04` note: `camera`).

Decision:

- Native PufferLib v4 remains the high-throughput training and regression backend, not the primary definition of competition readiness.
- Native `_C` execution is the required path for drone training and evaluation; `--slowly`/PyTorch backend runs are not acceptance criteria.
- Final competition integration must communicate through MAVLink v2 over UDP via a MAVSDK-compatible SITL bridge.
- Training interfaces must converge toward the official observable interface instead of relying permanently on privileged absolute state.
- Native `drone_race` success is a training milestone only; it is not a qualifier-equivalent result until the policy/controller runs through the SITL/MAVLink interface.
- The first qualifier-facing implementation should be a conservative telemetry+camera visual-servo controller. RL/native checkpoints should be used to improve or replace pieces only after the observable pipeline is running.
- Linux/Vast remains valid for training throughput, but official simulator integration must account for the TS-002 Windows 11 DCL simulator runtime.

Spec constraints:

- Physics target rate: `120 Hz`.
- Qualifier-facing command rate: engineering target `>=50 Hz`, with strict TS-002 cap `<100 Hz`.
- Minimum heartbeat rate: `2 Hz`.
- Round 1 maximum run duration: `8 minutes` (`480s`).
- Simulator uses a local Cartesian internal frame; no GPS simulation or absolute global position is exposed.
- Supported inbound control messages include `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`.
- Supported simulator-to-client MAVLink messages include `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, and `TIMESYNC`.
- Relevant telemetry bullets include vehicle attitude, orientation, linear velocities, and system status flags. TS-002 no longer lists the TS-001 simulator navigation-reference/`ODOMETRY` row, so any dependency on nav-reference/ODOMETRY data needs organizer clarification.
- Vision is a forward-facing first-person camera stream over UDP: default port `5600`, `30 Hz`, `640x360`, little-endian packet metadata, 24-byte header, and JPEG payload chunks keyed by `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
- Camera model: pinhole, no lens distortion, `fx=fy=320`, `cx=320`, `cy=180`, `VFoV=90 deg`, camera/body same origin, camera tilted `20 deg` upward relative to body.
- Drone chassis: `280x280x160 mm`.
- Gate geometry: outer frame `2700x2700x260 mm`; inner square opening `1500x1500x260 mm`.
- Official DCL simulator runtime: Windows 11, standard PC with a decent GPU around `8GB` VRAM; Linux OS is currently not supported for the simulator.
- Course geometry, physics parameters, and environmental conditions are deterministic.
- Human interaction during a submitted timed run remains grounds for immediate disqualification.

Current alignment assessment:

- Status: **gate-1 qualifier-representative runs proven**; full-course completion not yet achieved. Competition path is closed-loop visual-servo on v3379, not native privileged-state racing.
- Aligned:
  - Native v4 `_C` high-throughput training backend; WSL ext4 + RTX 3070 for Track B (~200–280K SPS sitl-plant).
  - `drone_race` motor/RK4 + sitl-plant `interface_mode=2` for transfer experiments.
  - MAVLink v2/UDP bridge with `<100 Hz` command cap; heartbeat `>=2 Hz`.
  - TS-002 camera ingest, aperture detector, state estimator, closed-loop attitude control.
  - Official first-gate advancement via race status (`agent_sprint_13`).
  - In-place MAVLink reset `31000` for fast iteration.
  - Deterministic JSON/CSV smoke artifacts and replay regression (`N=30` visual-servo).
- Not yet compliant:
  - Multi-gate / full-course official completion (milestone 4).
  - Learned policy transfer to live sim (milestone 3).
  - Native gate geometry vs TS-002 square opening dimensions.
  - `drone_race_competition` velocity-action contract on live v3379 (velocity setpoints inert).
  - Q8 closed-loop promotion.

Qualifier acceptance boundary:

- A run is qualifier-representative only if it:
  - communicates with the simulator over MAVLink v2/UDP through the SITL bridge;
  - maintains heartbeat at `>=2 Hz`;
  - emits accepted setpoint/control messages with an engineering target of `>=50 Hz` and a hard cap `<100 Hz`;
  - consumes official telemetry/observable streams rather than native privileged state;
  - consumes the TS-002 forward-camera stream through a JPEG reassembler when using vision;
  - respects the `480s` Round 1 maximum duration;
  - has no human interaction during the submitted timed run;
  - records deterministic report artifacts with success, ordered gate passes, completion time, crash/invalid-run status, command rates, telemetry dropouts, and vision-stream health.
- Native `_C` race metrics remain necessary for training throughput and regression, but are not sufficient for competition acceptance.

## Program-Level Success Metrics

- `valid_run_rate`: percentage of runs that complete all gates in order.
- `median_valid_time`: median completion time over valid runs.
- `ordered_gate_pass_rate`: percentage of accepted in-sequence gate crossings.
- `crash_rate`: fraction of runs that terminate via crash/unsafe state.
- Future SITL/vision metrics: camera FPS, frame jitter/stalls, JPEG reconstruction failures, chunk loss, `sim_time_ns`/MAVLink time-sync error, and command-rate enforcement violations above the TS-002 `<100 Hz` cap.

## Architecture Decision (Active)

Use a spec-first competition stack with native v4 as the training/regression backend:

1. SITL bridge: maintain heartbeat, parse confirmed TS-002 MAVLink telemetry, enforce command rate `<100 Hz`, and emit deterministic run reports.
2. Camera/perception interface: reconstruct TS-002 JPEG frames, synchronize `sim_time_ns`, apply camera intrinsics/extrinsics, and expose frames to perception.
3. Baseline perception/planning: detect the visually distinctive square gates, estimate relative gate pose/scale from image geometry, and sequence gates conservatively.
4. Controller interface: emit local setpoints or attitude targets compatible with MAVLink command messages, with bounded velocity/attitude/thrust behavior before speed tuning.
5. Native training envs: keep `ocean/drone` for hover/control prior and `ocean/drone_race` for race progression, teacher signals, and regression under increasingly spec-faithful observations/geometry.
6. Deterministic evaluation: keep fixed-seed native smoke/eval harnesses for regression, but require SITL/camera/telemetry reports for qualifier readiness.
7. Edge inference: keep policies small enough for deterministic inference and add quantized PufferNet runtimes only after FP32 closed-loop SITL behavior is reliable.

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
- MAVLink/SITL adapter code, TS-002 camera ingestion, visual gate perception, controller integration, and submission/eval harness.

Out of scope for active race implementation:

- Rebuilding the old v3 Python `pufferlib.environments.drone_race` runtime.
- Hardware flight implementation.

## Training Backbone and Governance Model

Decision:

- Use native v4 `drone` hover/control training as a stability prior and teacher source.
- Use native v4 `drone_race` as a support track for dynamics, robustness, and reward diagnostics, not as the competition acceptance boundary.
- Use SITL/camera/telemetry outputs as the primary promotion surface once the baseline controller exists.
- Add L1-L9 as promotion governance gates over SITL outputs, with native evals retained as preflight/regression gates.

### Tier Backbone

- H0: native `drone` hover/control prior and controller sanity checks.
- R0: native `drone_race` support-track diagnostics under increasingly spec-faithful geometry.
- S0: SITL/MAVLink adapter smoke with telemetry parsing, command-rate enforcement, and deterministic reports.
- S1: SITL/MAVLink + TS-002 camera ingestion with visual gate detection and first-gate visual servoing.
- S2: SITL/MAVLink + vision/perception multi-gate sequencing.
- S3: speed optimization after reliable valid completion.

### L1-L9 Governance

- L-levels do not replace tier configs.
- L-levels decide promotion, remediation, checkpoint/controller acceptance, and whether a native result is allowed to influence the competition-facing path.
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

1. Build the observable SITL/camera/telemetry pipeline before treating native race metrics as competition progress.
2. Establish a conservative visual-servo controller that can pass gates using TS-002 camera frames and confirmed MAVLink telemetry.
3. Keep native hover/race training as a support track for control priors, robustness, diagnostics, and teacher signals.
4. Tighten native race timing and geometry to match TS-002 before using native scores as transfer signals.
5. Optimize speed only after sustained valid completion through the SITL interface.

Current priority order (2026-07-06):

1. **Track A — gate-1 brake proof:** 3/3 official passes, zero collision, completed `BRAKE -> FOLLOW_GUIDANCE` transition.
2. **Track A — gate 2:** **complete** via accepted, collision-free
   `full_policy_rl_stage_c_v6c_gate2_001`; reproduce before extension.
3. **Track A — Gate 3/full course:** extend the recovered recurrent-policy
   prefix without regressing official Gates 1-2; do not add privileged mapping.
4. **Reliability:** reach 8/10 valid full-course runs.
5. **Speed:** tune only the guidance/approach/commit pitch values while preserving reliability.
6. Track B, native geometry work, R3/R4, and Q8 — **paused** until official gate 2.

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
- Audit native gate/chassis geometry against TS-002 and add spec-faithful clearance tests.
- Improve 3-gate race reliability to promotion threshold as a support track.
- Promote to 4-gate/full-course curriculum only after R3 passes deterministic eval or the SITL baseline demonstrates reliable multi-gate behavior.

### Phase C: MAVLink/SITL Adapter

- Add MAVLink client scaffold for UDP MAVLink v2.
- Maintain heartbeat at `>=2 Hz` in scaffold/dry-run.
- Send `SET_POSITION_TARGET_LOCAL_NED` and/or `SET_ATTITUDE_TARGET` with an engineering target of `>=50 Hz` and strict `<100 Hz` cap in scaffold/dry-run.
- Parse attitude, orientation, linear velocity, status flags, and `TIMESYNC`.
- Do not rely on simulator navigation-reference or `ODOMETRY` data until organizer clarification confirms availability under TS-002.
- Use the selected normalized body velocity/yaw-rate model contract and decode it to MAVLink local-NED velocity/yaw-rate setpoints.
- Add telemetry dropout, command-rate, heartbeat, and time-sync metrics to every SITL run report.

### Phase D: Vision/Perception Path

- Add TS-002 forward-camera UDP ingestion on default port `5600`.
- Implement 24-byte metadata header parsing, JPEG chunk reassembly, chunk-loss detection, and reconstruction-failure reporting.
- Sync frame `sim_time_ns` to telemetry/control logs.
- Integrate pinhole intrinsics (`fx=fy=320`, `cx=320`, `cy=180`, `VFoV=90 deg`, no distortion). TS-002 documents a `20 deg` upward camera tilt, but the v3379 renderer is measured LEVEL (`scripts/probe_camera_tilt.py`); keep the tilt configurable and set to `0 deg` for v3379.
- Implement a first-pass square-gate detector and relative pose estimator from image geometry.
- Implement a conservative visual-servo route: center the gate in image space, control closure rate, pass through slowly, and only then increase speed.
- Keep privileged/native state available only for training diagnostics and teacher signals.
- Train/evaluate telemetry-only and telemetry+vision variants separately.

### Phase E: Submission Harness

- Add deterministic submission entrypoint for SITL.
- Add local SITL eval runner with fixed course/start conditions.
- Record and enforce no-human-interaction compliance assumptions for submitted timed runs.
- Add SITL report fields for vision FPS, frame jitter/stalls, JPEG reconstruction failures, chunk loss, `sim_time_ns` sync error, and command-rate cap enforcement.

### Phase F: Edge Inference

- Benchmark native float `forward_puffernet` latency on local CPU.
- Add Q8/mixed-precision PufferNet runtime.
- Add float-vs-Q8 latency and action-drift benchmark.
- Add exporter for trained float weights to packed quantized weights.
- Add closed-loop hover comparison harness for trained `.bin` checkpoints.
- Add fake quantization during native RL fine-tuning.
- Promote Q8 only on closed-loop hover/race metrics, not action drift alone.

## Immediate Sprint (Historical — superseded 2026-07-16)

This section is retained for provenance. Do not use it as current direction;
execute `docs/competitive_lap_execution_prompt.md` and the July 16 decision.

**Historical milestone:** Gate 1 is proven; brake and settle before turning,
then target gate 2.

### Track A (100% active effort)

- Validate cyan guidance offline with `scripts/eval_course_guidance.py` before live flight.
- Run `course-fsm`: guidance, align, approach, commit, straight brake, settle.
- Reproduce gate-1 pass 3× with zero collision and a completed brake transition.
- Then require official gate index 2 using `config/sitl_multigate_acceptance.json`.
- Judge: only `official_active_gate_index` counts; `ordered_gate_passes` is diagnostic.
- Closed questions (do not revisit): velocity setpoints (dead), TRACK_INFO (never broadcast), camera uptilt `0 deg`, hover thrust `~0.17–0.20`.

### Track B (paused; explicit opt-in only)

- Native training **finished without promotion** (`drone_race_sitl_plant_gate1_fast`, 10 chunks). Do not rerun same config.
- Do not train while Track A is below official gate 2.
- When resumed, fail fast after two evaluations at `success<=0.001`, `crash>=0.99`.
- Transfer retest only after native `success_rate>=0.85` with `policy-attitude` + aligned checkpoint.
- Orchestrator: `scripts/track_b_runner.py`, state `logs/track_b/state.json`.

### Housekeeping

- Strategy doc: `docs/course_completion_strategy.md` (progress snapshot 2026-07-06).
- PRD loop: `python scripts/prd_loop_runner.py` — state `logs/prd/state.json`.
- Windows runbook: `.venv-win`, ports `14550`/`5600`, reset `31000`.
- R3/R4/Q8: support track only — do not block Track A.

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
- 2026-05-11 Vast check: the previous direct SSH endpoint `root@89.221.67.131 -p 16604` reset during key exchange, so remote repo/checkpoint sync and deterministic R3 re-eval could not be run from this session.
- 2026-05-14 local R3 continuation attempt:
  - Restored `checkpoints/drone_race/1777433724949/0000000026279936.bin` locally from `/Users/anon/code/rl/puffer_ai/vast_backup_20260428/checkpoints/drone_race/1777433724949/0000000026279936.bin`.
  - Rebuilt `drone_race` with `bash build.sh drone_race --cpu` using Homebrew LLVM/libomp and confirmed `scripts/test_drone_race_native_regressions.sh` passes.
  - Deterministic checkpoint eval did not produce JSON/CSV artifacts: the preexisting local `_C` lacked `create_pufferl`, the CPU rebuild still only exposes the VecEnv surface, and the required default-native PufferL backend cannot be built locally because this macOS workstation has no `nvcc`/CUDA.
  - No R3 checkpoint was promoted and no training continuation was run. Resume on Vast/Linux CUDA with a default `bash build.sh drone_race`, not `--cpu` or `--float`.
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
- R3 low-crash diagnostics added after the above candidate:
  - Metrics now include `crash_low`, `crash_high`, `crash_xy`, `crash_low_z`, and `crash_low_vz`.
  - Re-eval artifact: `logs/drone_race/eval_r3_best_low_crash_detail_2000.json`.
  - Finding: R3 failures are low-altitude dives (`crash_low` accounts for essentially all crashes), averaging just below the floor with high downward velocity.
  - Mild altitude-floor shaping produced one promising 1k checkpoint, `checkpoints/drone_race/1777416809505/0000000026279936.bin` (`success_rate=0.8782`, `crash=0.1218`), but its 2k re-eval regressed to `success_rate=0.8458`, `crash=0.1542`; do not promote it.
- R3 route-position diagnostics added after low-crash diagnostics:
  - Metrics now also include `crash_low_progress`, `crash_low_time`, and low-crash next-gate buckets: `crash_low_next_gate0`, `crash_low_next_gate1`, `crash_low_next_gate2`, `crash_low_next_gate3plus`.
  - Re-eval artifact: `logs/drone_race/eval_r3_best_route_detail_4096.json`.
  - Best-candidate 4k metrics: `success_rate=0.8029`, `crash=0.1971`, `gates_passed=2.6283`; all crashes were low-altitude crashes.
  - Low-crash route split: about `34.5%` while targeting gate 1, `19.6%` while targeting gate 2, and `45.9%` while targeting gate 3; average low-crash progress was `0.4218` and average low-crash time was `4.485s`.
  - Control-penalty bracket probes from the best candidate were not robust:
    - `w_ctrl=0.016` produced a 1k false positive at `checkpoints/drone_race/1777432567340/0000000026279936.bin` (`success_rate=0.9019`, `crash=0.0981`), but 4k re-eval failed at `success_rate=0.7465`, `crash=0.2535`; do not promote it.
    - `w_ctrl=0.014` produced a 1k false positive at `checkpoints/drone_race/1777432719880/0000000039387136.bin` (`success_rate=0.9040`, `crash=0.0960`), but 4k re-eval failed at `success_rate=0.7808`, `crash=0.2192`; do not promote it.
  - Lower time pressure also regressed: `w_time=0.1`, `w_ctrl=0.012` from the best candidate collapsed late, and its best 1k scan was only `success_rate=0.8475`, `crash=0.1525`; do not promote it.
- R3 raised-gate altitude-margin curriculum probes:
  - Goal: train R3 with higher gate altitude as a curriculum split, then evaluate checkpoints back on the original R3 course (`gate_altitude=1.0`) to test whether altitude margin transfers.
  - `gate_altitude=1.35` improved the best-candidate 4k re-eval but did not pass: `checkpoints/drone_race/1777433609185/0000000026279936.bin`, artifact `logs/drone_race/eval_r3_alt135_transfer_best_4096.json`, `success_rate=0.8381`, `crash=0.1619`, `gates_passed=2.7309`.
  - `gate_altitude=1.6` is the best current non-promoted R3 lead: `checkpoints/drone_race/1777433724949/0000000026279936.bin`, artifact `logs/drone_race/eval_r3_alt160_transfer_best_4096.json`, `success_rate=0.8775`, `crash=0.1225`, `gates_passed=2.7847`.
  - `gate_altitude=1.8` regressed and should not be used; its best 1k transfer scan was only `success_rate=0.8514`, `crash=0.1486`.
  - Combining `gate_altitude=1.6` with `w_ctrl=0.014` also regressed; its best 1k transfer scan was `success_rate=0.8511`, `crash=0.1489`.
  - Do not promote any raised-gate candidate yet. The useful signal is that altitude-margin curriculum helps, but all larger evals still fail `crash <= 0.10`.
- R3 vertical-energy/time-to-floor diagnostics:
  - Goal: explain low-altitude dives as time-to-impact and stopping-distance failures, not just altitude-threshold misses.
  - Code now emits `crash_low_floor_margin_pre`, `crash_low_ttf_pre`, `crash_low_stop_margin_pre`, `crash_low_max_up_accel_pre`, `floor_impact_risk`, `floor_stop_violation`, `floor_risk_steps`, `floor_stop_violation_steps`, `floor_risk_sampled`, `min_floor_ttf`, and `min_floor_stop_margin`; `scripts/eval_drone_race_checkpoint.py` also writes conditional averages such as `env/avg_crash_low_ttf_pre`.
  - Local verification: `SDKROOT="$(xcrun --show-sdk-path)" CFLAGS="-O2 -DNDEBUG -isysroot $(xcrun --show-sdk-path)" bash scripts/test_drone_race_native_regressions.sh` passed, including a forced low-floor crash test with negative stopping margin.
  - Deterministic checkpoint eval remains blocked locally: the target checkpoint is now restored under `PufferLib/checkpoints`, but the local CPU `_C` build does not provide the PufferL `create_pufferl`/rollout/load-weight API, and default-native rebuild requires CUDA/NVCC. No candidate is promotable from this diagnostic-only step.
  - Next action: on Vast/Linux CUDA, sync these code/docs, rebuild `drone_race` with `bash build.sh drone_race`, run a 1k diagnostic eval of `checkpoints/drone_race/1777433724949/0000000026279936.bin` with the established R3 config, then run 4k only if it still passes the scan.
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
  - `checkpoints/drone_race/1777416711275/0000000069992448.bin`: strong altitude-floor/descent shaping collapsed to `success_rate=0.0`, `crash=1.0`.
  - `checkpoints/drone_race/1777416809505/0000000069992448.bin`: mild altitude-floor/descent shaping also collapsed late, despite one promising early checkpoint.

## Risks and Controls

- Risk: speed tuning before reliability leads to invalid-run collapse.
  - Control: enforce L6/L8 gates before aggressive speed optimization.
- Risk: native training overfits privileged state and fails the official observable interface.
  - Control: keep MAVLink telemetry/vision adapter as an explicit promotion gate.
- Risk: v4 native CUDA build cannot be validated on local macOS.
  - Control: use native CPU build/eval only for local correctness; run CUDA/NCCL native training on Linux GPU before trusting throughput claims.
- Risk: timing mismatch versus qualifier simulator.
  - Control: use `120 Hz` / `480s` defaults in native race config and SITL tests, and enforce the TS-002 command cap `<100 Hz`.
- Risk: native geometry or obstacle simplifications produce a policy that clears native gates but fails official gate/chassis clearance.
  - Control: audit native gate, frame-depth, chassis, obstacle, and boundary geometry against TS-002 dimensions before treating native results as transferable.
- Risk: telemetry contract assumes TS-001 navigation-reference/`ODOMETRY` data that TS-002 no longer lists.
  - Control: build telemetry-only paths on confirmed TS-002 telemetry fields and request organizer clarification before adding nav-reference dependencies.
- Risk: camera packet loss or timestamp drift destabilizes perception/control in SITL.
  - Control: gate telemetry+vision promotion on JPEG reconstruction, chunk-loss, frame-jitter, and `sim_time_ns` sync metrics.
- Risk: official simulator runtime differs from the Linux training environment.
  - Control: keep Linux/Vast for training, but validate the competition bridge against a Windows 11 DCL simulator host with the required GPU class.
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
- MAVLink/SITL scaffold dry-run passes locally, with TS-002 `<100 Hz` command validation.
- TS-002 camera packet parser/reassembler scaffold passes synthetic unit tests locally.
- TS-002 corner-based gate pose and visual-servo command helper passes synthetic geometry tests locally.
- TS-002-shaped native policy contract now exists in `drone_race` via `interface_mode = 1`, with local C regressions covering observation layout and velocity/yaw-rate action mapping.
- `drone_race_competition` config loads as a profile over the compiled `drone_race` backend through `backend_env_name = drone_race`.
- Python SITL-side policy action decoding now exists in `scripts/drone_policy_contract.py` and passes unit coverage.
- `scripts/drone_sitl_competition_smoke.py` now emits deterministic artifacts with robust ordered-gate pass/completion fields (confidence-gated, hysteresis/rearm, debounce over missed detections, cooldown, rejection counters); local run artifact paths remain `logs/sitl/competition_smoke_gate1.json` and `logs/sitl/competition_smoke_gate1.csv`.
- Latest canonical replay-backed smoke run (2026-05-30) sent compliant heartbeats/commands, received telemetry/camera traffic, and passed first gate with acceptance enabled (`ordered_gate_passes=1`) in `logs/sitl/competition_smoke_gate1.{json,csv}`.
- The smoke runner now supports strict acceptance gating (`--require-telemetry --require-camera --min-gate-passes 1`) and returns explicit blocker reasons when first-gate conditions are unmet.
- A fast stream preflight tool now exists (`scripts/sitl_stream_probe.py`) to verify inbound MAVLink/camera packet presence before running longer smoke evaluations.
- Windows-local official simulator validation works on the RTX 3070 workstation (ports `14550`/`5600`). **First-gate official progress proven 2026-07-03** (`agent_sprint_13`). Stale pre-3379 reset snapshots are not promotion evidence.
- Manual reset snapshot `logs/sitl/reset_snapshot_manual_001/snapshot_summary.json` confirms the current simulator can start in a valid first-gate view: local position near origin, non-black frames, `188` detections in `5s`, and `race_status.active_gate_index=0`. Auto-reset snapshot `logs/sitl/reset_snapshot_auto_001/snapshot_summary.json` is invalid for promotion evidence because it produced black frames and implausible altitude.
- Live camera traffic required bounded packet draining because each frame is chunked into many UDP packets; the smoke runner now drains up to `--camera-max-packets-per-loop` packets per loop (default `512`) and processes the newest completed frame.
- A local mock stream harness now exists (`scripts/mock_ts002_stream.py`) and was used to produce a strict passing artifact run on 2026-05-30 (`ordered_gate_passes=1`, nonzero telemetry/camera, acceptance passed) in `logs/sitl/competition_smoke_gate1_mocked_required.{json,csv}`.
- UDP capture/replay now includes metadata sidecars (`capture_id`, SHA256, timing profile, generator params), with replay-time hash verification and deterministic impairment injection (loss/reorder/latency/jitter) for degraded-stream regression.
- Deterministic regression runner now exists (`scripts/sitl_replay_regression.py`) to run repeated smoke acceptance and paired visual-servo vs policy comparisons on identical replay streams.
- Checkpoint-backed policy callable now exists (`scripts/policy_callable_checkpoint.py`) for wiring trained `drone_race_competition` checkpoints into `--policy-callable`.
- Detector stress suite now exists (`scripts/detector_stress_suite.py`) with frozen threshold config for noise/blur/compression/occlusion/scale.
- `--slowly`/PyTorch backend runs are considered debug-only and are not part of the accepted training path.
- Native CUDA build remains unavailable on local macOS; use Linux GPU hardware for accepted throughput/training validation.
- Latest implementation result: gate 1 remains complete (`agent_sprint_13`, 3/4 repro). The new `course-fsm` uses cyan guidance, red-gate precision phases, official-status transitions, and straight post-pass braking. Offline guidance and live brake/gate-2 proofs are pending. Track B is paused and fail-fast guarded.

## Current Competitive-Lap Decision — N146, 2026-07-17

- Product objective is unchanged: a valid, autonomous, competitive VQ1/R1
  time through all six ordered gates, followed by optimization. There is still
  no official finish, so the goal is not complete.
- Candidate 007 is rejected at Gate 1. Its `457` inference ticks were sampled
  into only `180` trace records, which prevents honest exact recurrent replay.
  Exact observable projection shows the N145 severe-direct branch was eligible
  `0/180` times and changed `0/180` actions; this isolates the failure to the
  unchanged N112 prefix. Evidence SHA-256 is
  `33b7c5621b5610739a094271299ca2cf20160f30d5f17ea38ad2d6b5b2f4ec3b`.
- The selected minimal correction is N146: the proven legacy v6c prefix owns
  Gates 1--3, and the existing independently reset 32-input models own Gates
  4--6. Candidate-129 history supplies ten official resets with Gates 1--2 at
  `10/10` and Gate 3 at `8/10`. Exact camera confidence is carried in reserved
  slot 30 for the prefix and removed before tail inference; measured float32
  bridge error is `0` across `767` historical samples. The evidence artifact
  is `logs/sitl/n146_hybrid_prefix_exact_bridge_20260717.json`, SHA-256
  `bab4185386812a48c85f27f4a521abe35de20fef3ca0dec56187fe12d841b039`.
- Deployment is fail-closed and hash-pinned in
  `scripts/run_windows_six_gate_hybrid.ps1` (SHA-256
  `3398e62437d18933630c912feac1f2da51c4fa7d7dfd1317da3ab607cd996d66`).
  Actual Windows Python passes `75/75` focused tests, both PowerShell scripts
  parse, and real Windows model loading/inference passes across all phases.
- Simulator lifecycle remains explicit: normal reset is MAVLink `COMMAND_LONG`
  `31000`, and success requires detected boot/race epoch change. The only next
  simulator work is one reset-only setup followed by passive zero-command
  `n146_six_gate_hybrid_shadow_010`. No Candidate 008 flight or FullLap is
  authorized before that shadow passes and the bounded flight is separately
  preregistered.

### N146 live promotion and next bounded proof

- One telemetry-confirmed reset-only setup used MAVLink command `31000` and
  changed simulator boot time from `83605 ms` to `3248 ms`; Gate 1 was visible.
- Passive `n146_six_gate_hybrid_shadow_010` passed with zero lifecycle/control
  commands, `59/59` complete visible samples, maximum replay error
  `1.8423500058872833e-7`, all 11 hashes matching, and no-op cadence
  `550/10.031 = 54.730 Hz` with zero violations or probe setpoints. Official
  index stayed `0`, finish `-1`, and health was clean. Shadow/parity SHA-256:
  `f0ee13930e8b09908201daaf48bcaad5984c8d9fc3766977137d97093b59e7ab`,
  `5418c12a00c487eccee8c426c980e098a0229846bdb612b7b06ac9e3b304b09d`.
- Exactly one bounded Candidate 008 is now authorized under tag
  `n146_six_gate_hybrid_gate4_bounded_008`: one detected command-`31000` reset,
  `80 Hz`, `30 s`, repeats/min-valid `1/1`, target/min/stop index `4`. It must
  stop before Gate 5 with finish `-1`; no FullLap or unchanged retry is
  authorized.

### Candidate 008 rejection and N147 ABI correction

- Candidate 008 ran once and is rejected at Gate 1: detected command-`31000`
  reset, zero official passes, index `0`, finish `-1`, then collision ID `1002`
  (impact `0.25549981`) at a `19.016 s` stop. Closest accepted Gate-1 geometry
  was `3.168/1.851/0.203 m`. Wire transport passed at `56.488 Hz` with zero
  fast violation, dropout, or drain hit. Aggregate/smoke SHA-256 values are
  `8a6d1f3d55d5dcacfdeb009f27aba369e0dac8b8c9cd2b89749b3ae546ce573e`
  and `d71af1b1d89416e485f0eda83deb546bf6a5170e06d9e66ec207e293c48da961`.
- The N146 bridge was incomplete: it restored legacy confidence (field 17)
  but not legacy time (field 18, `elapsed/10.5` rather than `elapsed/30`) or
  legacy phase (field 22, `gate/3` rather than last-yaw command). N147 restores
  all three while zeroing both reserved values before tail inference. Exact
  model-input roundtrip error is `0` across `767` archived samples; diagnosis
  report SHA-256 is
  `c561b89e2209a744efc2a2dd73ad32fd5103ec532e2b21834477dd586eae4b56`.
- Actual Windows passes `78/78` focused tests, both launchers parse, and model
  loading/inference passes. N147 wrapper SHA-256 is
  `446c18048f6e18e3c1523b3282e460ba6989867485816cc5c0743b35af195022`.
  Next is reset/UI recovery as required, then passive
  `n147_six_gate_hybrid_shadow_011`; no flight is authorized yet.

### N147 live shadow passed; Candidate 009 bounded

- Setup command `31000` was accepted (`67590 -> 3330 ms`), with visible Gate
  1 and no process/UI recovery. Passive Shadow 011 then passed with zero
  lifecycle/control commands, complete `54/54` visible samples, max replay
  error `1.8943729400422438e-7`, all 11 hashes, cadence `56.824 Hz`, and clean
  health. Shadow/parity SHA-256:
  `bfe416b4841bf850167f0e0eae0222741b801884ef32c6ce0f6687e936b3878e`,
  `ac28cd83014c4753b8720dcae9d3c2ef7f20a98229fb13c60746b1df73b9148d`.
- One Candidate 009 is authorized under tag
  `n147_six_gate_hybrid_gate4_bounded_009`: detected command-`31000` reset,
  `80 Hz`, `30 s`, repeats/min-valid `1/1`, target/min/stop index `4`. No FullLap
  or unchanged retry is authorized.

### Candidate 009 rejection and corrected N148 recurrent input

- Candidate 009 ran exactly once and is rejected. Its command-`31000` reset
  was telemetry-detected (`134628 -> 3375 ms`; race start `3278 ms`), but it
  made zero official passes and collided at Gate 1 (ID `1002`, impact
  `10.26780033`, threat `2`). Wire transport was valid at `55.243 Hz` with
  zero fast violation/dropout/drain hit, so transport is not the cause.
- The prior N147 conclusion about field 22 was wrong. The historical callable
  used outer field 22 to select phase, then overwrote recurrent model input 22
  with `_LAST_YAW_ACTION` and updated it from action index 3. N147 instead fed
  `gate/3`, causing the observed recurrent divergence and saturated roll.
- N148 is the minimal correction: preserve current field 22
  `last_cmd_yaw_rate_norm`; restore legacy confidence and `elapsed/10.5` only
  through reserved slots 30/31. Gate one-hot selects the phase outside the
  prefix checkpoint. Evidence
  `logs/sitl/n148_hybrid_prefix_correct_recurrent_bridge_20260717.json` passes
  with zero blockers and exact bridge error `0`; SHA-256 is
  `9c9be7edceac7951f982d58bb93eb5959da7b3bdc1c41b5650e3261dfdb99d12`.
- `75/75` focused tests pass under actual Windows Python and the N148 launcher
  parses. Next authorization is reset-only setup plus zero-command
  `n148_six_gate_hybrid_shadow_012`; no flight or FullLap is authorized before
  that shadow passes.

### N148 live shadow passed; Candidate 010 bounded proof

- One reset-only MAVLink command `31000` changed simulator boot
  `244531 -> 3307 ms`, set race start `3298 ms`, and restored visible Gate 1.
  Reset snapshot SHA-256:
  `f517a08edd61e2c8d7a989569ccf1670bbc66af1d6eb2635908990c89a41a04c`.
- Passive Shadow 012 passed with zero lifecycle/control commands, `44/44`
  visible samples, max replay error `2.5586318969095245e-7`, all 11 hashes,
  cadence `55.738 Hz`, and clean health. Shadow/parity SHA-256 values are
  `9ed223ffbb23a21e2c0c86582588501f06b13c275e37717e7f3ea183eda08c53`
  and `8536d25564d660d6cd6f7f319c2c94f6c190b84f6fd76ceae9e04c7ba696e409`.
- Exactly one Candidate 010 is authorized under tag
  `n148_six_gate_hybrid_gate4_bounded_010`: detected command-`31000` reset,
  `80 Hz`, `30 s`, repeats/min-valid `1/1`, target/min/stop official index `4`.
  No FullLap or unchanged retry is authorized.

### Candidate 010 rejection; N149 restores historical update cadence

- Candidate 010 ran once and is rejected at Gate 1. Its command-`31000` reset
  was detected (`171210 -> 3415 ms`), but it made zero official passes and
  collided (ID `1001`, impact `11.93652916`, threat `2`) at `5.562 s`.
  Transport remained valid at `56.779 Hz` with zero fast violation/dropout/
  drain hit.
- The fresh recurrent output matches historical attempt 010 within
  `0.0003507`; the remaining mismatch is runtime cadence. Historical successes
  processed observations at minimum/median `10.857/11.524 Hz`; Candidate 010
  processed motion at `4.675 Hz` and inferred at `5.753 Hz` because the
  high-priority wire publisher busy-spun and starved the main loop.
- N149 changes only publisher waiting: sleep/release the GIL until the last
  `1 ms` before each Windows deadline. Evidence
  `logs/sitl/n149_legacy_prefix_cadence_correction_20260717.json` passes with
  zero blockers, SHA-256
  `5623d409ce320835d4ebb64d9fc273ef6394003c01b9daf5f71ed98556b833d0`.
  A new shadow gate requires at least `10.0` inference Hz. Only reset setup and
  passive `n149_six_gate_hybrid_shadow_013` are authorized next.

### N149 live cadence proof passed; Candidate 011 bounded

- Command `31000` setup reset changed boot `47901 -> 3159 ms`, race start
  `3299 ms`, and restored visible Gate 1. Passive Shadow 013 then passed with
  zero lifecycle/control commands, `306/306` visible replay samples, max error
  `1.9685e-7`, all 11 hashes, inference `30.6 Hz`, no-op wire cadence
  `59.990 Hz`, and clean health. Shadow/parity SHA-256 values are
  `b9e93071556688fdbf92057d1706ba7b8c365c5f4e58edc99a49daafeae2d855`
  and `820b2b2cdae5a271f03aa6b213e8689cd8ca53c0b58ededcee9f72aa13f5cc9d`.
- Exactly one Candidate 011 is authorized under tag
  `n149_six_gate_hybrid_gate4_bounded_011`, detected command-`31000` reset,
  `80 Hz`, `30 s`, repeats/min-valid `1/1`, target/min/stop index `4`. FullLap
  and unchanged retry remain unauthorized.

### Candidate 011 Gate-3 rejection; N150 close-severe completion

- Candidate 011 passed official Gates 1-2, then collided at Gate 3 (ID `1001`,
  impact `9.45933437`) at `10.687 s`. This validates the scheduler/prefix and
  isolates the next reliability issue to Gate 3. Wire cadence was `61.121 Hz`,
  inference `42.294 Hz`, with zero rate/dropout/drain issue.
- Its final observable Gate-3 prediction remained `-1.072 m` lateral at
  `2.693 m` forward while the promoted tapered roll was only `0.9101`. N150
  raises roll to `1.0` only when promoted adaptive mode still indicates a
  severe miss inside `4 m`; every non-roll action and all other states/gates
  are untouched.
- Trace evidence affects Candidate 011, two samples from historical lateral
  failure 001, and only one sample among all eight historical Gate-3 passes.
  Non-roll error is zero. Report SHA-256:
  `3328ee70d9f20d763c5e5a1dd9bad4f59f5ffe89c5585f63ecaa3b20cb74d21d`.
  `81/81` focused Windows tests pass. Only reset setup plus passive Shadow 014
  is authorized next.

### N150 reset and live Shadow 014 passed; Candidate 012 bounded

- One reset-only MAVLink `COMMAND_LONG` command `31000` restored a fresh VQ1
  race epoch: prior boot `115371 ms` changed to `3053 ms`, race start was
  `3283 ms`, and visible non-black Gate 1 returned without relaunching the
  existing simulator. Setup SHA-256:
  `b2b9a5d02d1a2d4ad39739ad515c7bdd6a367ca10a93149ee15a5f7337f3432b`.
- Passive Shadow 014 sent zero reset/arm/setpoint/disarm and passed. All
  `315/315` visible samples replayed within `2.675651550321234e-7`, inference
  was `31.5 Hz`, all 11 hashes matched, and no-op cadence was
  `591/9.938 = 59.368 Hz` with zero violations/setpoints. Official index `0`,
  finish `-1`; collision/dropout/drain counts were zero. Shadow/parity hashes:
  `718d9080fa13b9437d386e759412c21f72c0c100a0d718fcf7205461a5b6862a`,
  `2b386ba7129be7f4ba29a5f1b61dc53c95827ced8ba9b4f2757c4e5ada473e92`.
- Exactly one Candidate 012 is now authorized under tag
  `n150_six_gate_hybrid_gate4_bounded_012`: reuse FlightSim, one detected
  command-`31000` reset, requested `80 Hz`, maximum `30 s`, repeats/min-valid
  `1/1`, target/minimum/stop official index `4`. Accept only clean index `4`,
  finish `-1`, cadence `[50,100) Hz`, all hashes, zero collision/invalid/
  dropout/drain/rate issue, one exit disarm, and passive disarm proof. Reject
  on failure without unchanged retry. FullLap remains unauthorized.

### Candidate 012 reset-only rejection; N151 lifecycle correction

- Candidate 012 emitted no policy flight. Preflight saw index `0`, finish
  `-1`, boot `398002 ms`, base `193`, status `4`, but race age `364.9 s`
  triggered unnecessary UI recovery. The helper then accepted old start epoch
  `3283 ms`; command `31000` produced no boot rollback, and the reset guard
  failed closed before policy arm/setpoints. Summary/per-attempt hashes are
  `747b25daf555328f004bc711053d97b9bf8bd9fe65614137f57b3cdaf9a671e2`
  and `c2395081d0d75d3ae48a2ebe1b70e7168f59b009041192018cb9016297e15934`.
- Passive state confirms old boot continued to `490448 ms`, index `0`, finish
  `-1`, no collision, visible Gate 1; the reset sequence's disarm left base
  `65`, status `3`. Snapshot SHA-256:
  `acf0e8f3eaf1ba7601b33ac93e679533e2c4d5c97743fdafd403fde461438fd5`.
- N151 removes race age as a recovery signal. Live base `193`/status `4` with
  an active race is command-31000-reset-ready and stays untouched; UI recovery
  is used only when that live readiness test fails. Startup/wrapper SHA-256:
  `198b356d613d238077577b2595b25466c59bc4832a03369c95820b032e8775af`,
  `2d6f7a85e1845e7a8228a61b534678fea8fd88d4ae3f0ba9b5e2583c9020149b`.
  Focused Windows tests pass `67/67`; PowerShell parsing passes.
- Authorize only legitimate UI recovery from the current base `65`/status `3`,
  one reset-only command `31000` setup with detected epoch/camera recovery,
  then passive `n151_six_gate_hybrid_shadow_015`. No flight or FullLap yet.

### N152 window-relative recovery and live Shadow 015 passed

- N151 was incomplete: the simulator physical window starts at `(949,465)`,
  but recovery coordinates were sent as absolute screen positions, opening
  Graphics instead of Back to Main. The helper also accepted old active-race
  telemetry at base `65`/status `3`. N152 offsets clicks by the live window
  rect and requires base `193`/status `4` after UI recovery; an already-ready
  race is still reused regardless of age. Startup/wrapper SHA-256 values are
  `1f75d88d8ff6839e921af01872b699fe8beedfad7eca0332fbc7df013d79474d`
  and `b6c6650d0c127d05e1975d581153dc4f49c5085955556dd30a20d1beefed9158`.
  Focused Windows tests pass `68/68`; PowerShell parsing passes.
- Correct recovery reused PID `24476` and changed race start
  `3283 -> 1337380 ms`. A subsequent reset-only command `31000` changed boot
  `1354001 -> 2989 ms`, race start `3288 ms`, and restored visible Gate 1,
  base `193`, status `4`. Setup SHA-256:
  `757acbf536304d768cbe60c9757785546c45b5022fc25d682f8243140f949a27`.
- Passive Shadow 015 sent zero lifecycle/control commands and passed:
  `279/279` visible replay samples, maximum error `2.609935760289339e-7`,
  inference `27.855 Hz`, all 11 hashes, no-op `60.4 Hz`, zero health/rate
  events, official index `0`, finish `-1`. Shadow/parity hashes:
  `135d4911b2741fc3810ea5d50ea626e1d076f02b5472bc4f4ca9572e690c69d7`,
  `971752ebce398f09fa0b7096c13b61e2ebe68059e1159fd8dafcb0a7e8c3136f`.
- Exactly one Candidate 013 is authorized under tag
  `n152_six_gate_hybrid_gate4_bounded_013`: reused PID, current pinned wrapper,
  one detected command-31000 reset, requested `80 Hz`, maximum `30 s`,
  repeats/min-valid `1/1`, target/minimum/stop index `4`. Accept only clean
  index `4`, finish `-1`, cadence `[50,100) Hz`, all hashes, zero collision/
  invalid/dropout/drain/rate issue, exit disarm plus passive proof. Reject on
  failure without unchanged retry. FullLap remains unauthorized.

### Candidate 013 Gate-3 miss; N153 invalidates cached gate epoch

- Candidate 013 had a valid command-31000 reset (`298904 -> 3402 ms`), passed
  official Gates 1-2, and stayed collision-free at index `2`, finish `-1`,
  through the `30 s` cap. Transport/inference were `61.61/55.067 Hz`, zero
  rate/dropout/drain issue. Smoke/post-snapshot SHA-256 values are
  `7c142e1bfe1c1519b37fecbffe708e9636ff9e3f73004e0206f65bf67c3aad63`
  and `a046df830216fae4adacf96e0b9e68d393fb965980c9297319958ce523734ea4`.
- At the Gate-3 transition the visible body vector was
  `[24.615,-3.192,5.538] m`, while a cached Gate-2 pose made the policy observe
  `[3.730,-2.324,-0.734] m` (`21.824 m` error). The filter then accepted zero
  and rejected 21 visible Gate-3 samples. This asynchronous cache poisoning,
  not the close-severe roll value, caused the miss.
- N153 clears both raw/control gate caches and timestamps and resets gate
  motion whenever official identity changes. Evidence SHA-256:
  `b99b451cb22ca7a3f9ccff5ce82f0a2b424e434a6930870b68ce3d9a252c0d15`.
  Runner/wrapper SHA-256:
  `26933542fa3d6dec17fa8859adef9c88e145b7e4072e2fee04a9e2a1b7be09d3`,
  `20d53f406484dfc7124e8b7f0c050a2e00a444c3679259bcf5cd5bc5b2213e70`.
  Focused Windows tests pass `82/82`; parser passes; live cadence remains
  `60.4--61.61 Hz`.
- Authorize only reset-only command `31000` setup plus passive
  `n153_six_gate_hybrid_shadow_016`. No N153 flight or FullLap before a passing
  shadow and separate preregistration.

### N153 reset and Shadow 016 passed; Candidate 014 bounded

- Reset-only command `31000` restored boot `3128 ms`, race start `3268 ms`,
  visible Gate 1, base `193`, status `4`, index `0`, finish `-1`; setup hash
  `be222e85a1a43200a8c899ba2cba8a0d71780fb26cbc8cf6a3a688f745191169`.
- Passive Shadow 016 sent zero lifecycle/control commands. `309/309` samples
  replayed within `2.2564048768325407e-7`; inference `30.567 Hz`, all 11
  hashes, no-op cadence `60.936 Hz`, zero health/rate issue. Shadow/parity:
  `6e7fdb132ae2180653e79baab1671120f3019eecc937053992777de124086e44`,
  `17902c975f59c3f5640d0a49884161c3dfb801a637d5639f9e287fd15425d64c`.
- Exactly one Candidate 014 is authorized under tag
  `n153_six_gate_hybrid_gate4_bounded_014`: reused PID, detected command-31000
  reset, requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/
  minimum/stop index `4`. Accept only clean index `4`, finish `-1`, cadence
  `[50,100) Hz`, hashes, zero health/rate issue, exit disarm plus passive proof.
  Reject failure without unchanged retry; FullLap unauthorized.

### Candidate 014 Gate-2 collision; N154 scopes invalidation to Gate 3+

- Candidate 014 reused PID `24476` and had a valid MAVLink command-`31000`
  reset (`198589 -> 3331 ms`, race start `3303 ms`, detection `0.484 s`,
  calibration `60/60`). It passed Gate 1 but collided at official index `1`,
  finish `-1`: ID `1001`, threat `2`, impact `9.970706939697266 m/s`.
  Publication was `377/7.0 = 53.714 Hz` with zero rate/dropout/drain issue.
  Aggregate/per-attempt/smoke/CSV SHA-256:
  `836d116272e91bd5636c200ea0cc4f01658f56cbc2b70d1a8d378bd7e4fee4de`,
  `32c35a08a8f868af63c2f19e01ff5a0588eaa841dc320b8e06e4c3f926a0dcf4`,
  `bf6d724b2a4dc4b08ba4e5d41ea121cab0c612ba617efe3e6bf6550bb04fc0ed`,
  `9fa39f626770407dfe08312539a77011d9ff817dfd9d897a3013bc34d7194197`.
- N153's every-transition clear was over-broad. N154 preserves the proven
  Gate-1-to-Gate-2 handoff and invalidates cached poses/motion only when the
  new official index is `>=2` (Gate 3 onward), where Candidate 013 proved the
  asynchronous stale-pose alias. No policy/action/scheduler value changes.
  Evidence/runner/verifier/wrapper SHA-256:
  `437c54fba6812468207f8a5713f71d6b72ec18f489df7c59c9b6756b24894b35`,
  `bb10b470e017c05c57fe2c5b9d5763708a9cd9b66133cdee8170d78496817f65`,
  `66a02b41c8011206fee4b2c337da21cbee3214ae013f16a3c622b754c96159f9`,
  `1965de670b84578c2184aa4e11f1de3bd7277cc93193b671d8ca8441b9a4cf6b`.
  Direct tests `54/54`; PowerShell parses. Passive live cadence is the final
  promotion authority for the known short scheduler-test fluctuation.
- Authorize reset-only command `31000` setup plus passive
  `n154_six_gate_hybrid_shadow_017` only. Reuse the running simulator; no
  N154 flight or FullLap before a passing shadow and separate preregistration.

### N154 reset and Shadow 017 passed; Candidate 015 bounded

- Command `31000` reset boot `58715 -> 3274 ms`, new race start `3288 ms`,
  visible Gate 1, base `193`, status `4`, index `0`, finish `-1`; setup hash
  `defad1af9403be583528022b293e70a41f70f16682c3fde610b5d8b2709952be`.
- Passive Shadow 017 sent zero lifecycle/control commands. `298/298` samples
  replayed within `1.6721377373019042e-7`; inference `29.752 Hz`, all hashes,
  no-op cadence `59.405 Hz`, zero health/rate issue. Shadow/parity:
  `4e082189f6015abcf1ef95546c1b9ee9c7c1a8818ca7509601e2a30c4b6a451a`,
  `ac0b9fb2d4d8374004f3c5cd9440b3a95b113920a70c4e9e00471e6b5fcc7b20`.
- Exactly one Candidate 015 is authorized under tag
  `n154_six_gate_hybrid_gate4_bounded_015`: reused PID, detected command-31000
  reset, requested `80 Hz`, maximum `30 s`, repeats/min-valid `1/1`, target/
  minimum/stop index `4`. Accept only clean index `4`, finish `-1`, cadence
  `[50,100) Hz`, hashes, zero health/rate issue, exit disarm plus passive proof.
  Reject failure without unchanged retry; FullLap remains unauthorized.

### Candidate 015 repeats Gate-2 edge strike; N155 projected governor

- Candidate 015 had a valid reset (`188703 -> 3403 ms`), passed Gate 1, then
  collided at index `1`, finish `-1`, impact `9.644123077392578 m/s`.
  Publication `60.276 Hz`, zero health/rate issue. Smoke hash
  `89338315b99eae47d5d5722dce145c0adafb803dc89a9ce11823fa8c678c2833`;
  passive snapshot hash
  `24d3064cddf3d0b1603e3ed6e3760998c454aa88365603c5334cce42a3702b77`.
- The final pose/rate projected a `+0.706 m` Gate-2 edge miss; Candidate 014
  projected `-0.658 m` on the other side. N155 acts only at Gate 2 inside
  `6 m` when `abs(projected miss)>0.5 m`: pitch action cap `-0.2`, lateral PD
  roll `kp=0.6`, `kd=0.35`, bound `0.7`. All other gates, thrust, and yaw are
  frozen. Evidence/policy/wrapper hashes:
  `336ff1e1b2b5cb9449ef592ef481c4a442208fb8c31c84393439cdde3c431faa`,
  `440a3bee51c3ceb4d039dec6757e970e348b989cd2d03c91d75f5d71100aaa21`,
  `8b0a55a49c9d68deb543ea19cf3113158522f962300d1b31493a8e6fc3f5bdb5`.
  Focused tests `28/28`; parser passes.
- Authorize command-`31000` reset-only setup plus passive Shadow 018. No N155
  flight or FullLap before shadow pass and separate preregistration.

### N155 reset and Shadow 018 passed; Candidate 016 bounded

- Command `31000` reset boot `38048 -> 3124 ms`, race start `3306 ms`, Gate 1,
  base `193`, status `4`, index `0`, finish `-1`; setup hash
  `937039ba64beddd9a7a39c3912a15acfe8dbdd981a84d7446492c6589a1643d0`.
- Passive Shadow 018: zero lifecycle/control commands, `238/238` replay,
  max error `1.8356218338400065e-7`, inference `23.762 Hz`, all hashes, no-op
  `56.609 Hz`, zero health/rate issue. Shadow/parity:
  `0ec4680f3863024bd7a82c348695b6b8c48f62f45a47fe8a415532ca438c75f6`,
  `73b5f94168771a87c6c2403c2c8e4057976830219d9aa4fdcc30bd539c3f567e`.
- Exactly one Candidate 016 is authorized under tag
  `n155_six_gate_hybrid_gate4_bounded_016`, reused PID, detected reset,
  `80 Hz`, `30 s`, `1/1`, target/minimum/stop index `4`. Reject failure without
  retry; FullLap remains unauthorized.

### Candidate 016 passed Gates 1--3; rejected at Gate 4

- Candidate 016 reused PID `24476` and had a valid command-`31000` reset
  (`140703 -> 3288 ms`, race start `3300 ms`, detection `0.531 s`, calibration
  `60/60`). It passed official Gates 1--3; N155 activated once at Gate 2 and
  converted the previously measured edge approach into a pass. It then
  collided at official index `3` with ID `1002`, threat `2`, impact
  `5.396681785583496 m/s`; finish stayed `-1`.
- Publication was `1195/19.812 = 60.267 Hz` over a `20.113 s` flight, with
  zero command-rate violation, telemetry dropout, or drain-limit hit; `5399`
  MAVLink messages, `2725` IMU samples, `252` detector frames, and `238`
  detections. Aggregate/per-attempt/smoke/CSV hashes are
  `839cb7e3bd5b83a716add57db441db0fb6881f955efef18d387210763c1ae0e1`,
  `549b46db4279b29f32ebbd0e9dd34503381246044b902c6270828254872044dc`,
  `556b05f935077baf16d69f5c0d7f6fa77067a4ef8f978d1c9e2ee178e9050af9`,
  and `9e7ae6c195708c01a3418f8c5b9b04aed162daae58a9c080b059be02bf61da13`;
  passive snapshot hash is
  `340542e8c4e2c0e9cd2f3553165743bae78094eb3179d1c5751f45f6fbbb524d`.
- The index-3 cache clear worked, but the existing Gate-4 tail accepted
  inconsistent red structures and produced aggressive lateral/descent control
  without a stable approach. Visual inspection corrects the initial diagnosis:
  the last raw edge detection is the cropped right upright/panel of the actual
  Gate-4 frame, not an unrelated sign; the closest frame shows the full scoring
  aperture. This was a lateral aperture miss under excessive tail authority.
  N156 will reuse the previously proven Candidate-143 Gate-4 safety semantics
  in the current 32-input hybrid:
  raw index-3 observation, one-second acquisition grace, `8 m` step/`12 m`
  anchor association bounds, bounded lateral roll, delayed and current-range-
  gated descent, and hover fail-safe. Preserve N155 Gates 1--3. Require offline
  evidence plus passive shadow before one bounded flight; FullLap remains
  unauthorized.

### N156 Gate-4 safe handoff passes offline

- Preserve N155 Gates 1--3 and the learned Gate-4 recurrent transition. At
  official index `3` only, use raw pose with one-second grace, `8 m` step/
  `12 m` anchor bounds, two `0.125 s` coherent close re-anchors inside
  `20.75 m` and `4.5 m`, roll `kp=0.02` capped at `0.10 rad`, yaw centering,
  bounded braking, three-second/current-pose descent authorization, thrust
  floor `0.22`, and level `0.27` hover/yaw freeze on rejected or missing pose.
  A new raw-observation end index confines this ABI change to Gate 4; Gates
  5--6 keep their trained filtered observations.
- Candidate-016 replay passes all `80` Gate-4 samples (`71` visible, `9`
  dropout), exercises two re-anchors, caps normalized roll at `0.2`, never
  commands below decoded thrust `0.22`, and restores level hover on every
  dropout. Evidence hash:
  `61eb7e8610920751b9eb6df0a4b0a351368efdc3fa99f9e064e3cb7a7b55f5e6`.
  Policy/runner/verifier/wrapper hashes:
  `4edec25979ecfab0a59b16ec05cec80673389ccf7cd622cbb10393ce2ca23302`,
  `0cb94723a2e6e266c690a0127c4c60450479a8aacc61d5594be51f33be4d0936`,
  `a72619c768f55b941398c0d93773601b9f79239c758ed7cb0e664dd3e7d1229f`,
  `f62757b3221cb35af02729c3091d8718aaf445f32a108d84029dcc206ab0fd00`.
  Linux tests `69/69`, Windows targeted tests `16/16`, Windows compile and
  PowerShell parse pass.
- Authorize only one separately logged command-`31000` reset setup followed by
  passive zero-command Shadow 019. Reuse PID `24476`; no N156 flight or FullLap
  before the shadow passes and a separate bounded preregistration.

### N156 lifecycle correction and reset setup

- Post-Candidate-016 passive state was inactive (`boot=1741494 ms`, race start
  `-1`, base `65`, status `3`), so reset was correctly withheld pending UI
  recovery. The preserved PID `24476` recovery exposed one coordinate defect:
  `(480,755)` selected Toggle HUD instead of Back To Main Menu. No flight or
  reset command occurred in the failed recovery.
- Window evidence fixed the v3385 relative coordinate to `(480,805)`. The sole
  VQ1/R1 race was restored on the same PID with base `193`, status `4`, index
  `0`, finish `-1`, race start `2050755 ms`, and visible Gate 1. Startup/wrapper
  hashes are
  `5122b892ebae263d5c28ecbf8694578fab353646a1cfd6363e44fb36bb807d40`
  and `f62757b3221cb35af02729c3091d8718aaf445f32a108d84029dcc206ab0fd00`;
  focused tests `62/62`, parser passes.
- One reset-only command `31000` then changed boot `2093597 -> 3085 ms`; new
  race start `3313 ms`, reset-ready status, visible Gate 1. Setup hash:
  `bccac933e615ce9738961deb16b087ff3a5b163218363d6cb6f19d7e059eb87d`.
  Authorize passive Shadow 019 only; no flight or FullLap.

### N156 passive Shadow 019 passed; Candidate 017 bounded

- Zero reset/arm/setpoint/disarm. All `311/311` visible samples replayed within
  `2.2075195310611306e-7`; inference `31.1 Hz`, all 11 hashes, no-op cadence
  `59.484 Hz`, zero health/rate issue, official index `0`, finish `-1`.
  Shadow/parity hashes:
  `e029e12a27cfee7d7d32b0af0e0757d3a9682b2f81b7f08502fa928277c9ca98`,
  `6a42838d8aa0e2ab816a1af027c8fd3218ab88369dd483b9fbe113a6041ee548`.
- Authorize exactly one `n156_six_gate_hybrid_gate4_bounded_017`: reused PID
  `24476`, current wrapper/Windows runtime, one detected command-`31000` reset,
  `80 Hz`, at most `45 s`, `1/1`, target/minimum/stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, hashes, zero
  health/rate issue, exit disarm plus passive proof. Reject without unchanged
  retry; FullLap unauthorized.

### Candidate 017 rejected at Gate 2; N157 late-trigger correction

- Valid reset `342878 -> 3330 ms`, race start `3302 ms`, detection `0.546 s`,
  calibration `60/60`, reused PID `24476`. Passed official Gate 1 only, then
  collision ID `1001`, threat `2`, impact `10.29787540435791 m/s` at index
  `1`; finish `-1`. N156 Gate 4 was not exercised. The vision-only ordered
  count `2` does not override official index `1`.
- Publication `454/7.469 = 60.651 Hz`, duration `7.766 s`, zero health/rate
  issue. Aggregate/per-attempt/smoke/CSV hashes:
  `3521cae3eda65ed6c0f900babdb541c96c493764e11210d9d13db132da28aea6`,
  `c7910fbda07d57eee2fde9b847501507211d428e350e915f97504a041b3f9f8c`,
  `473c97741465c743de814f1c3d07e789c6aa3664a0de906913fc56b423278417`,
  `e6de6799e6010c0cd9fb2d1b9cf50c50b26b3227300c701904ea89aebed8a010`;
  passive snapshot
  `a8e496d22d600b603303919cbc78399d10a25a12978d7a5e632f4139c23caf65`.
- At `4.916 m`, projected crossing was `+0.331 m`, so the `0.5 m` governor
  stayed off; the next sample jumped to `+0.757 m`, and the final projection
  remained `+0.708 m` with `0.400 rad` bearing. Candidate 016's clean pass ended
  around `+0.418 m`. N157 changes only Gate-2 admission to a symmetric `0.3 m`
  threshold latched until official transition; braking/PD values and all other
  gates remain exact. Require offline proof and passive shadow before flight;
  FullLap remains unauthorized.


### N157 Gate-2 latched aperture governor offline passed

- Change only Gate-2 admission from projected miss `0.5 m` to `0.3 m` and
  latch the existing brake/roll-PD correction until official transition.
  Preserve its gains/bounds, thrust, yaw, Gates 1 and 3--6, and N156 Gate 4.
- Archived replay has zero blockers: activation leads are `0.125 s` on failed
  Candidate 017 and `0.265 s` on clean Candidate 016; the latter also proves
  retention through a following below-threshold frame. Thrust/yaw error is
  zero and governed roll stays inside `0.7`.
- Policy/verifier/evidence/wrapper hashes:
  `743f1ae9fc656999ef1f7ef4450017850f724754924e4deec4913289d3c94e38`,
  `44903c877a6d2e658ef5be5316917f041083d6ad2362f7ad4da620016dd371fa`,
  `8a49fe26bcdb91d0056faeffbe261fbc960cc10742a50c391ac596ca5a19cdc6`,
  `6cc7977c62e002514ec229250916139627d2cf4b371a9bd9b3481e2f5d7b0f8f`.
  Linux `67/67`; actual Windows `20/20`; compilation and both parsers pass.
- Same-PID recovery required measured center Y values `277` for the VQ1 event
  and `872` for Race; startup hash is
  `d9e97ff6ac38c7a476eebc5ddfa5fd6f47d705c11f613922c17e9690f9031f7d`.
  It restored base `193`, status `4`, index `0`, finish `-1` without reset or
  flight.
- Authorize one logged reset-only MAVLink command `31000` on reused PID
  `24476`, requiring detected rollback/new epoch/index `0`/visible Gate 1,
  followed only by passive `n157_six_gate_hybrid_shadow_020`. No flight or
  FullLap before that shadow is accepted.

### N157 reset-31000 setup passed

- Reused responsive PID `24476`; sent exactly one reset-only MAVLink command
  `31000`, with no relaunch or flight setpoint.
- Boot rolled `1112445 -> 25848 ms` at verification; new race start `3292 ms`,
  base/status `193/4`, official index `0`, finish `-1`, non-black camera, and
  visible Gate 1 (`48` detections, first confidence `0.85375`).
- Setup/status/image hashes:
  `c73c6f465aa8e35f405a1dc4da9a867e88483b619efd7fc9701cd4e42c1b25c5`,
  `15def8bc55a264bfef2d345fccb9de73dba62222c6bffc2ba9f868a77d515e77`,
  `ee72e40319600462dcb6f3b2a3e81722eaa7bd5287294a742051ebca25d95636`.
  Authorize only passive `n157_six_gate_hybrid_shadow_020` next; no additional
  lifecycle command, flight, or FullLap.

### N157 passive Shadow 020 passed; Candidate 018 bounded

- Zero reset/arm/setpoint/disarm. Replay `132/132`, maximum error
  `2.4103546142351107e-7`, inference `13.2 Hz`, all 11 hashes, no-op cadence
  `555/10 = 55.4 Hz`, zero health/rate issue, official index `0`, finish `-1`.
  Telemetry/camera `2435` messages, `1254` IMU, `100` frames/detections; raw
  scope `3..3`. Shadow/parity hashes:
  `9b97f8b03db72d4bd34444b5f19bdc1bc92f0af6b6c7131017a7123e9f0abc29`,
  `1f397060c618c6a1d6b5a010c384b42d7e3f5fbf9a75d849e0ef0198d36a17ed`.
- Authorize exactly one `n157_six_gate_hybrid_gate4_bounded_018`: reuse PID
  `24476`, pinned wrapper/Windows Python, one detected command-`31000` reset,
  `80 Hz`, maximum `45 s`, `1/1`, target/minimum/stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, hashes,
  zero health/rate issue, exit disarm plus passive proof. Reject without
  unchanged retry; FullLap unauthorized.

### Candidate 018 rejected at Gate 4; N158 anchor/descent timing

- One detected reset `280952 -> 3393 ms`, race start `3306 ms`, calibration
  `60/60`, reused PID `24476`. Passed official Gates 1--3, then collision ID
  `1002`, threat `2`, impact `1.7557381391525269 m/s` at index `3`, finish
  `-1`. This proves N157's Gate-2 latch live; Gate 4 remains unpassed.
- Publication `836/15.734 = 53.07 Hz`, duration `16.047 s`, zero health/rate
  issue, `4492` messages, `2270` IMU, `168` frames, `157` detections.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `1b433638ab9b25d74bfe3fb94d8ab7d27696b52098a0f8767a7df9bc9fe4b602`,
  `bbbb0a6adbb480374f7dce6523607d07a8c99231f046c1e4472e6c8a05c0fa84`,
  `ed860bba7bb4cca7c27f8abd58f83af9601f3f636145d6c62ae2b35965e8b3d9`,
  `ea8449c4197787881e6fa4e1bca985268f55e9b845a47b04aa47cd5d39c689e6`,
  `1b8848ff297685bfb66d1dae1e4632f5a251008753cfc5323986fdafc6e64f87`.
- Grace admitted false right `10.875--11.458 m` as the Gate-4 anchor. The real
  centered family near `[25.412,0.556,13.818] m` was initially rejected,
  descent began `1.141 s` late, and the final `[26.667,-0.458,8.125] m` sample
  failed only stale-anchor drift. Replay hash:
  `70e33d9064fa526d1fd1553a20b3d680df09a17a01fa08855fc36e12da39bacc`.
- N158: require the existing `4.5 m` lateral bound for initial Gate-4 anchor;
  start vertical correction after association grace `1.0 s`, while retaining
  the `3.0 s` reanchor delay and all other controller values. Require replay
  of Candidates 016/018, tests, reset proof, and passive shadow before flight;
  no FullLap.

### N158 Gate-4 initial anchor and vertical timing offline passed

- Gate-4-only change: require `abs(right)<=4.5 m` for initial anchor and begin
  vertical control after association grace `1.0 s`; retain reanchor delay
  `3.0 s`, all gains/bounds, N157 Gates 1--3, and Gates 5--6.
- Zero replay blockers. Candidate 016: `80/71/9` total/visible/dropout, `23`
  false initial fail-safe samples, two reanchors, roll/thrust/dropout bounds.
  Candidate 018: `8` initial fail-safe samples, first centered association and
  full descent at `12.656 s`, lead `1.141 s`, final visible association retained.
- Policy/verifier/evidence/wrapper hashes:
  `de3c3069476fdcf56789a8a822f1f4ee03edecfa3410b0caaa1bce9c27933fa6`,
  `0bf4380646da5d8a189426cc022626061336e8ab24d48270bf6860bf5c3cdc16`,
  `67f9190d2231fed172f08d97a96d27dadb9f20e03d06aa67123bc5b2d41d64ca`,
  `73ec596346e3f0b3b816f7a6bda2647068dee11cabf9a5e1d5f55607c8604cfa`.
  Linux `68/68`; actual Windows `21/21`; compilation and parsers pass.
- Authorize one logged reset-only command `31000` on reused PID `24476` after
  VQ1/R1 recovery, requiring detected rollback/new epoch/index `0`/visible
  Gate 1, followed only by passive `n158_six_gate_hybrid_shadow_021`. No flight
  or FullLap before shadow acceptance.

### N158 reset-31000 setup passed

- Reused PID `24476`; exactly one reset-only command `31000`; boot
  `1006213 -> 2443 ms`, race start `3317 ms`, later base/status `193/4`, index
  `0`, finish `-1`, visible non-black Gate 1 (`74` detections, confidence
  `0.84875`). No relaunch or flight setpoint.
- Setup/status/image hashes:
  `cf1dbc53f9b500a3b0dea6628c133aad737efe873973c7088a0c3c23a2a6051a`,
  `c25f1c485bf9d7d33702dd17854df7b0b1dc812864cdfb776d1469a2efffc6ee`,
  `b431d19d38eee722c47e4953e4ae106195fa210b47e0a31227635b3ead61dffa`.
  Next only passive Shadow 021; no lifecycle command, flight, or FullLap.

### N158 passive Shadow 021 passed; Candidate 019 bounded

- Zero reset/arm/setpoint/disarm; replay `308/308`, maximum error
  `1.7313575745303567e-7`, inference `30.8 Hz`, all 11 hashes, no-op
  `603/9.984 = 60.296 Hz`, zero health/rate issue; index `0`, finish `-1`.
  Telemetry/camera `2476` messages, `1275` IMU, `124` frames, `121` detections;
  raw scope `3..3`. Shadow/parity hashes:
  `ca90056ecc823434aebdd186e3832177f2202a0a66df2cf8c31fbfc96a69b387`,
  `756bb1dd40973c8218d39b59d1a5218c4d02dca27b4333aa3f78d3a0ec27c7c3`.
- Authorize one `n158_six_gate_hybrid_gate4_bounded_019`: reused PID `24476`,
  pinned wrapper/Windows Python, detected command-`31000` reset, `80 Hz`,
  max `45 s`, `1/1`, target/minimum/stop index `4`. Accept clean index `4`,
  finish `-1`, cadence/hashes/health/disarm proof. Reject without retry; no
  FullLap.
- Readiness waiter now requires race status and heartbeat base/status together,
  eliminating the observed status-first false UI recovery. Hash
  `9c23a70544d0216b66bb3c6bf285b179be27dc1864fe966805d75bd881ac9cec`;
  Linux `8/8`, Windows `3/3` targeted tests pass.

### Candidate 019 rejected at Gate 3; Gate-4 proof still outstanding

- Exactly one `n158_six_gate_hybrid_gate4_bounded_019` reused PID `24476`.
  The waiter correctly left reset-ready VQ1/R1 alone; MAVLink `COMMAND_LONG`
  command `31000` rolled boot `358025 -> 3278 ms` (race start `3305 ms`,
  detected in `0.515 s`). No simulator relaunch occurred.
- Official Gates 1--2 passed, then collision ID `1001`, threat `2`, impact
  `3.598033905029297 m/s` at active Gate 3/index `2`; finish `-1`. The
  vision-only count `3` is non-authoritative. Gate 4/N158 never activated.
- Publication `622/10.390 = 59.769 Hz`, duration `10.754 s`, zero transport
  health issue; `3322` messages, `1678` IMU, `132` frames, `131` detections,
  and one exit disarm.
- Aggregate/attempt/smoke/CSV/passive hashes:
  `d327e3dfbff14bb7c5a29b5a04744d7465a14e0aa69694b1e8001100282d5b45`,
  `9d024e4aef2590f9d172712bdaef0c9926adfe5b99abf9f283088d725f8a55ea`,
  `c5ea83374a6c119eed8c108398ab5edfa942b508781e200ee1e088e25a5fcb43`,
  `24e5f9dc7c66e2f481a5ddb90e9e39b1ce44682fbed92b39daf51da780da614e`,
  `23d68708ce0681ff2d34841cf58693ed136b4eeb1f9ae1f9381cf9f73a5a12e2`.
- Near-plane position projected safely (`right=-0.252 m`), but the existing
  Gate-3 law repeatedly switched between about `+0.45` and `-0.50 rad` roll
  targets and reached measured roll about `-0.10 rad`. Successful Candidates
  016/018 retained positive roll at crossing (`+0.12/+0.48 rad`). Next change
  only close-plane Gate-3 roll persistence, prove it on archived traces, then
  reset/shadow before one new bounded flight. Preserve N158 Gate 4 and all
  other phases; no unchanged retry or FullLap.

### N159 Gate-3 close-plane roll persistence offline passed

- Gate 3 only, inside `4.0 m`: retain the existing full positive roll for an
  `adaptive` severe miss; when the promoted law enters `counter` while the
  observed gate is still left, floor normalized roll at neutral `0.0` instead
  of `-1.0`. No pitch/thrust/yaw or other-phase change.
- Candidate-016/018/019 replay has zero blockers. Candidate 016 changes three
  close counter samples, Candidate 018 changes none, and Candidate 019 changes
  exactly its final three negative counter actions to neutral at
  `3.927/2.850/2.199 m`. All non-roll values remain exact; final Candidate-019
  projected right remains centered at `-0.252 m`.
- Policy/verifier/evidence/wrapper hashes:
  `98f08ae181d62621a4fd3c3adb451c9f017aa747c5fdb42883136e42a9a040b9`,
  `e9d49d59fb3d0e8c30b5913195f86fb09c30871449baa01294ad43cd88a1794a`,
  `68d7930b11cd9ed785b0cefcc6a45e109e7f6f716784bd8fbfe5274c31754e89`,
  `629053098cc940a2a033104f286aaadc98c0e735fc3abce432099de829534ecf`.
  Linux `85/85`; actual Windows `24/24`; byte-compilation and both PowerShell
  parses pass.
- Next authorize same-PID VQ1/R1 recovery if required, then exactly one logged
  reset-only MAVLink command `31000` with detected epoch rollback/index `0`/
  finish `-1`/visible Gate 1, followed only by passive zero-command
  `n159_six_gate_hybrid_shadow_022`. No flight or FullLap before acceptance.

### N159 reset setup and passive Shadow 022 passed; Candidate 020 bounded

- Same PID `24476` recovered to sole VQ1/R1 without relaunch. One reset-only
  MAVLink `COMMAND_LONG` command `31000` rolled boot `1004917 -> 3333 ms`,
  race start `3312 ms`, base/status `193/4`, index `0`, finish `-1`, visible
  non-black Gate 1. Recovery/setup hashes:
  `b68c238406338bba0f6296b597fcbb2827b4150edec0d834054132dfc6f5fcae`,
  `948e8b826b9b78eac4a4c751808f9dece2358d40e1eaa6133c4bb1199ae0f25b`.
- Shadow 022: zero reset/arm/setpoint/disarm; replay `304/304`, maximum error
  `2.473499298161208e-7`, inference `30.306 Hz`, all 11 hashes, no-op cadence
  `598/10.016 = 59.605 Hz`, zero health/rate issue, index `0`, finish `-1`.
  Telemetry/camera `2419` messages, `1246` IMU, `126` frames/detections; raw
  scope `3..3`. Shadow/parity hashes:
  `7fef15cd5f0552e86b9fc2a575d5ce4fe2f8822543af0a382524f38f9fc7e2d1`,
  `8904d09ab12c0d972e458e5c342048f996b0b6da304bc3e73c1db4457003908f`.
- Authorize one `n159_six_gate_hybrid_gate4_bounded_020`: reused PID `24476`,
  pinned wrapper/Windows Python, one detected command-`31000` reset, `80 Hz`,
  max `45 s`, `1/1`, target/minimum/stop index `4`. Accept only clean index
  `4`, finish `-1`, cadence/hashes/health/disarm proof. Reject without retry;
  no FullLap.

### Candidate 020 passed Gate 3; collision-free Gate-4 acquisition stall

- One `n159_six_gate_hybrid_gate4_bounded_020`, reused PID `24476`, detected
  command-`31000` reset `145647 -> 3406 ms`, race start `3314 ms`, detection
  `0.515 s`, calibration `60/60`.
- Passed official Gates 1--3 and ended collision-free at index `3`, finish
  `-1`, after the full `45.0 s`. N159 is live-proven at Gate 3; N158 is
  live-proven fail-safe but did not acquire/pass Gate 4.
- Publication `2641/44.906 = 58.789 Hz`; zero health/rate/collision issue;
  `11005` messages, `5556` IMU, `527` frames, `188` detections. Aggregate/
  attempt/smoke/CSV/passive hashes:
  `4000e4ee7dfe2d96c2fc09afba4240227b2cdaf0af3d9841c94c0ceb73c1d38e`,
  `10b9a744bf3e8e30f1bd2ef11e8243f6849820badc51803a9b1b411b0aee24dc`,
  `123326118595c95984337c9f16451b3fd6e24e9150c94583c02509c52b654380`,
  `9a7f5b90027fb21b27a048b377d0ad11fe51be993ce237bbcd0d310da58268f4`,
  `0c9ee53e19cda74535f31890be493eecab8e1f689ef66c763ecef29f4bbd41f1`.
- Gate-4 raw right remained far positive (`+9.469` to saturated `+19.001 m`)
  from `10.844` through `16.391 s`. N158 rejected it safely but held current
  yaw, so the target left view and the drone hovered. N160: only on rejected
  visible Gate-4 samples after grace, command bounded yaw centering while
  retaining brake/zero-roll/hover and every existing anchor/descent invariant.
  Replay Candidates 016/018/020 before reset/shadow; no FullLap.

### N160 Gate-4 rejected-pose yaw acquisition offline passed

- Gate 4 only after `1.0 s`: a visible pose rejected by existing association
  gets a current-pose yaw step toward target capped at `0.35 rad`; fail-safe
  brake pitch, zero roll, hover, dropout yaw hold, and all admission/descent
  rules remain exact.
- Candidate-016/018/020 replay: `29/4/37` yaw acquisitions, bounded at
  `0.35/0.340845/0.35 rad`, correct direction, zero non-yaw violation, and all
  `9/7/240` dropouts hold yaw. Candidate 018 still anchors at `12.656 s`,
  Candidate 016 retains two reanchors, and the N158 regression passes.
- Policy/verifier/evidence/regression/wrapper hashes:
  `587e4b57f7fc4b28fabeaea2e189a81200835ebb645bb3f06465e885f9572953`,
  `2c1eee4a15088354a1dd32f70f8b60641c6d1fe0c38cdffb0f710d3410ee89ba`,
  `7ad9998a8dabc8bfe96f231ce6842abf2ee00f4d1199728a3f6ae34f76df6c4d`,
  `1d099acec8c05f3cc3fbe7da7a849af734d15f805edc7b912ffda684d09c8eec`,
  `7c5e80a30ed7f78e5f4b22a0f806a972ebdf2d5161210213b3919dcfa2987901`.
  Linux `86/86`; actual Windows `25/25`; compile and parsers pass.
- Next: same-PID VQ1/R1 recovery, one reset-only command `31000` with detected
  rollback/index `0`/visible Gate 1, then only passive
  `n160_six_gate_hybrid_shadow_023`. No flight/FullLap before acceptance.

### N160 reset setup and passive Shadow 023 passed; Candidate 021 bounded

- Responsive PID `24476` was reused without relaunch. Sole VQ1/R1 recovery was
  followed by one reset-only MAVLink `COMMAND_LONG` command `31000`; simulator
  boot rolled `642916 -> 3096 ms`, race start was `3299 ms`, base/status
  `193/4`, official index `0`, finish `-1`, and Gate 1 was visible/non-black.
  Recovery/setup hashes:
  `a18fa1ccc6eeb2f916896d2223d5b60c55bbd98cb90424936cab0c5cc74ec339`,
  `25afeafd311b52c3248385c14b8fcf9eb514312dba1306dfaee4f500ad54d45f`.
- Shadow 023 issued zero reset/arm/setpoint/disarm. It replayed `285/285`
  visible samples, maximum error `2.2688121795177985e-7`, inference
  `28.409 Hz`, all 11 hashes, and no-op cadence `572/9.922 = 57.549 Hz` with
  zero collision/health/rate issue. State remained index `0`, finish `-1`;
  telemetry/camera were `2426` messages, `1250` IMU, `140` completed frames,
  `119` detections. Shadow/parity hashes:
  `d413d77bead96c6a4b42929a201d41c6cd64afc50c8490669c3791e48f2fca40`,
  `8f5c03dba529849e58a87a064a934c34dea966dfcb2d0327798e407e02174d7e`.
- Authorize one `n160_six_gate_hybrid_gate4_bounded_021`: reused PID `24476`,
  pinned wrapper/Windows Python, one detected command-`31000` reset, `80 Hz`,
  max `45 s`, `1/1`, target/minimum/stop official index `4`. Accept only clean
  index `4`, finish `-1`, cadence/hashes/health/exit-disarm/passive proof.
  Reject without unchanged retry; FullLap remains unauthorized.

### Candidate 021 low-speed Gate-4 contact; N161 coordinated acquisition

- Candidate 021 ran once without retry, reused PID `24476`, and detected reset
  `281466 -> 3402 ms` (race start `3274 ms`, detection `0.516 s`, calibration
  `60/60`). It passed official Gates 1--3, then contacted Gate-4 course geometry
  at index `3`, finish `-1`: ID `1002`, threat `1`, impact
  `0.036701612174510956 m/s`.
- Publication `1343/21.906 = 61.262 Hz`; zero health/rate issue; `5861`
  messages, `2956` IMU, `335` frames, `257` detections, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `81d4a2eaf669c9d22e348577d735f8853e16a6959281df4b5ae33a112f653f12`,
  `2b2929392e9fbd055a43f35b21206c121d1e6dba0aa2c37aeecbaffb005ab57e`,
  `5ea58acf47b4643dab648ec9b5f25a05f46f4904f2e33149fd13024a67d4183f`,
  `923cc7b3d53002769ea58370d6d98707c4808cebf231ce4e51d7a25378aee1a7`,
  `f9a5a9a663e49a4c0d5b9f1d3dec97d515abe3ac9b0b6aece4f95ab8e81ec4b6`.
  Passive state was base/status `65/3`, index `3`, finish `-1`, no reset.
- N160 first anchored Gate 4 at `14.422 s`, but zero roll during pre-anchor yaw
  acquisition allowed substantial inside lateral motion (`-2.767` to
  `-4.701 m/s`). N161 adds only the existing bounded lateral roll law during
  rejected-visible pre-anchor acquisition after `1.0 s`; brake/hover/yaw and
  all post-anchor/dropout/other-gate behavior remain exact.
- Four-trace replay (Candidates 016/018/020/021) passes, exercising
  `16/2/37/23` samples, maximum normalized roll `0.2`, zero pitch/thrust/roll/
  yaw/dropout violation, unchanged first anchors `13.629/12.656/none/14.422 s`
  and reanchors `2/0/0/1`. Policy/verifier/evidence/wrapper hashes:
  `0b8dbd20347972ae64e8753cdb79018e3def97278e7df25ac982ec04106ea098`,
  `dbc5f774a943e63e6b04b97dd43c6f10ec1f52ac98a50afa329ede8f5ef383f5`,
  `b5a69a3a0b55a77c4802b8e8b1fe527a2a912c619e30f78ffe9357d80712a920`,
  `8f6252c670e6a0096235e058f5038111b595c53a724af96ddb0b90f2a470dfb7`.
  Linux `130/130`; Windows `32/32`; compilation and parsers pass.
- Next: same-PID VQ1/R1 recovery, one reset-only command `31000` with detected
  rollback/index `0`/visible Gate 1, then only passive
  `n161_six_gate_hybrid_shadow_024`. No flight/FullLap before acceptance.

### N161 reset setup and passive Shadow 024 passed; Candidate 022 bounded

- PID `24476` recovered to sole VQ1/R1 without relaunch. One reset-only command
  `31000` rolled boot `1052102 -> 4381 ms`, race start `3298 ms`, base/status
  `193/4`, index `0`, finish `-1`, collision-free visible Gate 1. Recovery/
  setup hashes:
  `2d4dda8c379b929e9a1dacca8d0fe90d4bc8277a1d21e3080a3355b2e47a3f63`,
  `7aa038c3f2541f27ffdc66db7ca3e8d3d0672f0e3943110b85090bfdccde63ab`.
- Shadow 024 sent zero reset/arm/setpoint/disarm; replay `299/299`, max error
  `2.3133983612089182e-7`, inference `29.9 Hz`, all hashes, no-op cadence
  `597/9.985 = 59.69 Hz`, zero collision/health/rate issue, index `0`, finish
  `-1`; `2416` messages, `1245` IMU, `141` frames, `125` detections. Shadow/
  parity hashes:
  `14ee0ebb15482960aa8340406af0944d9f9dcb7724543e07b18d11a5befa9df0`,
  `8e2f9b1e81f0c43fe010cb545e7bcbf9c46600687a6be2146bf89f8d30380143`.
- Authorize one `n161_six_gate_hybrid_gate4_bounded_022`: reused PID `24476`,
  pinned wrapper/Windows Python, one detected `31000` reset, `80 Hz`, max
  `45 s`, `1/1`, target/minimum/stop index `4`. Accept clean index `4`, finish
  `-1`, cadence/hashes/health/exit-disarm/passive proof. Reject without retry;
  no FullLap.

### Candidate 022 rejects N161 roll; N162 bounded dropout yaw carry

- One Candidate 022, no retry: reused PID `24476`, reset
  `149965 -> 3415 ms`, race start `3341 ms`, detection `0.500 s`, calibration
  `60/60`; passed Gates 1--3, then Gate-4 collision at index `3`, finish `-1`,
  ID `1002`, threat `2`, impact `8.278605461120605 m/s`.
- Publication `1156/19.390 = 59.567 Hz`; zero health/rate issue; `5278`
  messages, `2664` IMU, `301` frames, `237` detections, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `37d3a8d817bb01ccd0d260d42ac0b279e24728a8d4e8cc3ec619f279a5e09986`,
  `8e4ddbbfea05a8f6c0893b670c7cce695aa134193603a519d616c78f96ac669d`,
  `dcea9e3bbd68c6bb8ba2a0bbea69d91e1f34c7610ea5c879d9b98b89fffd6763`,
  `5595b2e6311835e2662d303ee51f20b2c36915bcf53b3b1c80bd370bac966019`,
  `f3fd4e6a834c786df6a982d2d71333c79db05644f0cac25b0eae62db29da1be8`.
  Passive base/status `65/3`, index `3`, finish `-1`, no reset.
- N161 roll is removed: alternating pose families saturated it at `-0.10` then
  `+0.10 rad`, driving estimated lateral position `-7.504 -> +55.082 m`.
- N162 restores N160 zero-roll acquisition and only carries the last visible
  yaw target across dropout for `<=0.5 s`, clamped `<=0.35 rad`; brake/level/
  hover stay exact, then current-yaw hold resumes. Five-trace carry/expiry
  counts `4/5,4/3,0/240,3/10,0/0`; deltas
  `0.35/0.021800/0/0.094046/0`, zero law violation, unchanged anchors/reanchors.
  Policy/verifier/evidence/wrapper hashes:
  `052e1869da50d359af526ed90d02e2ad30fdf9fb523bd737311b497764776b68`,
  `c95a12c0de71b09385fa1f59d38bad4fc8061f3148882029475ed8dda82be16e`,
  `209cffdcce091262f4ce682a66979a146cc20defa6277ae291d563dbbf724d03`,
  `16d7c79be471fca0cf2f574e13147e614e02f76973a5d4c7950998386707c78a`.
  Linux `131/131`; Windows `33/33`; compile/parsers pass.
- Next same-PID VQ1/R1 recovery, one reset-only `31000`, then passive Shadow025
  only. No N162 flight or FullLap before acceptance.

### N162 reset setup and passive Shadow 025 passed; Candidate 023 bounded

- Same-PID recovery reused responsive simulator PID `24476` and restored sole
  VQ1/R1 at base/status `193/4`, official index `0`, finish `-1`, without
  relaunch or flight. Recovery SHA-256:
  `b299444ea7ef46aad8f28e9333e2f30d3380d51192cc91c644c915c37a8cb395`.
- Exactly one reset-only MAVLink `COMMAND_LONG` command `31000` rolled boot
  `583001 -> 4437 ms`; race start `3338 ms`, base/status `193/4`, index `0`,
  finish `-1`, zero collisions, and visible Gate 1 (`72` detections, first
  confidence `0.80257`). No flight setpoint was sent. Setup SHA-256:
  `7f8c8d2dc387704653125dc447fad9582aedda05f713fcdc04f1f14886002403`.
- Shadow 025 sent zero reset/arm/setpoint/disarm. Replay passed `313/313`, max
  error `1.9664711952555036e-7`, inference `31.25312031952072 Hz`, all 11
  hashes, and no-op cadence `611/10.015 = 60.909 Hz`; official index `0`,
  finish `-1`, zero collision/dropout/drain/rate issue, `2438` messages,
  `1255` IMU, `142` completed frames, `126` detections. Shadow/parity hashes:
  `e709c89653c897df160c438fbd09515c6cb04d7e974e08a53978321b5b088444`,
  `838fd78f37edb339db728d72ebb7e4bbd9504bc421bfb7ba3eacd7634bb7cfcc`.
- Authorize exactly one `n162_six_gate_hybrid_gate4_bounded_023`: current
  pinned wrapper/Windows Python, reused PID `24476`, one detected command-
  `31000` reset, `80 Hz`, max `45 s`, `1/1`, target/minimum/stop index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching
  hashes, zero collision/invalid/dropout/drain/rate issue, one exit disarm and
  passive proof. Reject without unchanged retry; FullLap remains unauthorized.

### Candidate 023 rejected at Gate 2; N162 Gate 4 not evaluated

- One run, no retry: reused PID `24476`; detected reset
  `283598 -> 3412 ms`, race start `3300 ms`, detection `0.500 s`, calibration
  `60/60`. Gate 1 passed, then collision ID `1001`, threat `2`, impact
  `4.348381042480469 m/s` at collision-time official index `1`, finish `-1`.
- Publication `416/7.000 = 59.286 Hz`, zero rate/dropout/drain issue; `2535`
  messages, `1280` IMU, `89` frames, `85` detections, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `8c050294f2a16191e51369032b0a1caef1003c8a91ca9476c4d4c6fe052d38b9`,
  `6487400add7765f58daf6aafb69799c510feaeb7dfdbbe5e1e05c3644ef87f68`,
  `c9fc184a8641272985a1949e890359b7078a3ca94bec104ccbb514a3f0ec864d`,
  `3fe1884952dd628f282dbd49e8ee9789da2e51cc8d46836eb6ca454c8d2366cf`,
  `91a9e7270ed27cd912535f3c3ae22b5823900bdfab090024de6aef8e3158be35`.
  Passive inactive state was base/status `65/3`, finish `-1`; its eventual
  index `2` followed the collision and is not a valid pass.
- Gate-2's unchanged `6 m` governor latched only at `4.055 m`, projected miss
  `+0.502 m`, `0.344 s` before impact; accepted vision already projected
  `+0.845 m` at `11.847 m`. Diagnostic hash:
  `b827e2d471e2e661d6c4c342bfb98bdf4527b09c8ed38662f62a724be86bef48`.
  Next prove only an earlier Gate-2 admission window offline. No flight or
  FullLap before evidence and a new passive shadow.

### N163 Gate-2 early-brake window offline passed

- Only Gate 2 changes. Unsafe projection can latch the existing normalized
  pitch brake `-0.2` inside `12 m`; in the new `6..12 m` band, learned roll,
  thrust, and yaw stay exact. Inside `6 m`, the current roll PD (`kp=0.6`,
  `kd=0.35`, cap `0.7`) is unchanged. Gates 1/3--6 and N162 Gate 4 are exact.
- Candidates 016--023 pass replay with early sample counts
  `4/6/4/5/3/5/5/6`, activation leads
  `0.468/0.687/0.750/0.578/0.344/0.656/0.641/0.687 s`, and zero early roll/
  thrust/yaw error. Candidate 023 moves brake admission from close PD at
  `4.055 m` to `11.847 m`, projection `+0.845 m`, retaining learned roll
  `-0.221`. Independent Gate-4 preservation replay also passes.
- Policy/verifier/evidence/Gate-4-preservation/wrapper hashes:
  `80cb465c4d5a2eea8bc9e557c121f1d3382761fc0666457d65d0e54c10f677f8`,
  `29ba34411310c888191351abdb4a969440debabf78aac4157664bf73f2f3bd96`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `2d469b512e83fd86ae63598ee615f8721a485e92f06d44e0e996b9a28a73b50f`,
  `38f3f400b7fc74c98b8b81430f34018bdb5fcd80a9f7514aebb355d5527e85ba`.
  Linux `132/132`; actual Windows `34/34`; compile/parsers pass.
- Next: same-PID VQ1/R1 recovery, exactly one reset-only command `31000`, then
  only passive `n163_six_gate_hybrid_shadow_026`. No N163 flight or FullLap
  before acceptance.

### N163 reset setup and passive Shadow 026 passed; Candidate 024 bounded

- Recovery reused PID `24476` and restored sole VQ1/R1 at base/status `193/4`,
  index `0`, finish `-1`, with no relaunch/reset/flight. Recovery hash:
  `625f20efec9c3d673da11edb8919d65373abf7ad8f7413fa3a96fb2cb1e06aba`.
- One reset-only command `31000` rolled boot `1119238 -> 2942 ms`; race start
  `3269 ms`, base/status `193/4`, index `0`, finish `-1`, zero collision,
  visible Gate 1 (`152` detections, first confidence `0.76460`), no setpoint.
  Setup hash:
  `51b4576dfb17a3be70a02630dbdcde57d4fb5e5ac5399c1011c297eeaba2180c`.
- Shadow 026 sent zero reset/arm/setpoint/disarm; replay `324/324`, max error
  `2.651107788298468e-7`, inference `32.4 Hz`, all hashes, no-op
  `623/10.0 = 62.2 Hz`, index `0`, finish `-1`, zero health/rate issue; `2423`
  messages, `1248` IMU, `143` completed frames, `125` detections. Hashes:
  `98f9925ba06fbd9bc732b0a6d36100c04565f8bb0a56792ddc7df26df98c259a`,
  `73525574036ba95b4e709bfa171958eeaf22b29ec3d2dfac1a8657c12e0f4113`.
- Authorize exactly one `n163_six_gate_hybrid_gate4_bounded_024`: current
  pinned wrapper/Windows Python, reused PID `24476`, detected command-`31000`
  reset, `80 Hz`, max `45 s`, `1/1`, target/minimum/stop index `4`. Accept only
  clean index `4`, finish `-1`, cadence/hashes/health/exit-disarm/passive proof.
  Reject without retry; FullLap remains unauthorized.

### Candidate 024 proves N163 Gate 2; rejected at Gate 3

- One run, no retry: PID `24476`, detected reset `175602 -> 3416 ms`, race
  start `3309 ms`, detection `0.500 s`, calibration `60/60`. Official Gates 1
  and 2 passed; Gate-3 collision ID `1001`, threat `2`, impact
  `10.180516242980957 m/s`, index `2`, finish `-1`; Gate 4 not reached.
- Publication `628/10.406 = 60.254 Hz`, zero rate/dropout/drain issue; `3289`
  messages, `1661` IMU, `133` frames, `132` detections, one exit disarm.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `950d02b452a70c9645c945d1e343ec9af73cd7615f75b5cbf980d8cc7f35477d`,
  `a5ccbf982fe988b9c9314dfebf1ab199fdf2852289d95f2b3acd515b51238bab`,
  `5c790b9cefb2caa58bd02a52964cd5a07e94923178b219dc2279590c9cf8bddf`,
  `ccfbae57ca1a7fe1f3f01a7f780c36e5cb50cc482d1d988853358d13f4f217e8`,
  `be6c4cc343fe08f6c34412e54d89e1ac9561983b18ca8ede979834d5499ea8ab`.
  Passive base/status `65/3`, index `2`, finish `-1`, no reset.
- At `5.38 m`, projected lateral/vertical error was already safe
  (`-0.37/-0.82 m`) while output remained pitch/roll `+0.24/-1.0`. The final
  safe projected lateral error was `+0.206 m`, but direct right `+0.19 m`
  released the close floor and restored full counter-roll. Diagnostic hash:
  `f1a65cb90d61e1f8ae74b9d036c8a2e518b72a39efe6f7fb1758b3574d3211e`.
  Next prove only a Gate-3 terminal safe-aperture brake/level latch inside
  `6 m`, preserving thrust/yaw. No flight or FullLap before evidence/shadow.

### N164 Gate-3 terminal safe-aperture coast offline passed

- Gate 3 only: inside `6 m`, projected lateral/vertical error within
  `0.5/1.0 m` latches pitch/roll `-0.2/0.0` until official transition;
  learned thrust/yaw stay exact. N163 remains exact before admission.
- Candidates 016/018/020/022: zero activation/change. Failures 019/024:
  `2` samples from `2.850 m` and `4` samples from `5.384 m`; Candidate 024's
  first projected error is `[-0.374,-0.822] m`, replacing `+0.240/-1.0` with
  `-0.2/0.0`. Clean Candidate 021 activates only at `1.771 m` for three
  already-level samples. Zero thrust/yaw or pitch/roll-law violation.
- Independent Gate-2 and Gate-4 preservation replays pass. Policy/verifier/
  evidence/Gate-2/Gate-4/wrapper hashes:
  `eaa37cc016ee3f78fbff511db949da270e8ab69a0232b43db0933a9bf0168da4`,
  `9ff6f1ac7201adf3a5d199012803f0b68e364d4b1a9d1801adaf890e1edeb91f`,
  `dbd26ae238af9f6cb76d51906abca480783ab63d3cfecffab9f659db1fef9364`,
  `0aa2c8280dfacd6c38a4f2bf1d2046b9e53bce27708b23d66051a8cba209164a`,
  `aca52f7ca35dd6e20a9677e2ad03f50d43bbbcdc680345e65a4f328c795e428b`,
  `1c6697bf5cc760d0ca640620aa717f991cb709d6679bbe81a79e00d734957c22`.
  Linux `136/136`; actual Windows `38/38`; compile/parsers pass.
- Next: same-PID VQ1/R1 recovery, one reset-only `31000`, then only passive
  `n164_six_gate_hybrid_shadow_027`. No flight/FullLap before acceptance.

### N164 reset setup and passive Shadow 027 passed; Candidate 025 bounded

- Recovery reused PID `24476`, restored sole VQ1/R1 at base/status `193/4`,
  index `0`, finish `-1`, and did not relaunch/reset/fly. Recovery hash:
  `e1861c289d4358f83a4b075e49c159329a642c2acf40f89145990f74b41a1e9b`.
- Exactly one reset-only command `31000` rolled boot `1056545 -> 3090 ms`;
  race start `3307 ms`, base/status `193/4`, index `0`, finish `-1`, zero
  collision, visible Gate 1 (`200` detections; first confidence `0.7901247`),
  no setpoint. Setup hash:
  `172cc787d5a71ab53c7f34a1448cafc80a004d36830598f962eee138c1a489d3`.
- Shadow 027 sent zero reset/arm/setpoint/disarm; replay `297/297`, max error
  `2.2150001527387886e-7`, inference `29.656 Hz`, all hashes, no-op
  `608/9.968 = 60.895 Hz`, index `0`, finish `-1`, zero health/rate issue;
  `2438` messages, `1256` IMU, `143` frames, `127` detections. Hashes:
  `2a12e5f15239904caf9f651a2b4da1fe2e4c0d6db4df1fa60511114b34ffda3d`,
  `6bb172f849ecd4d8b47d058f1b1cefe1d07815baf687d04be8680754936b6807`.
- Authorize exactly one `n164_six_gate_hybrid_gate4_bounded_025`: current
  pinned wrapper/Windows Python, reused PID `24476`, detected command-`31000`
  reset, `80 Hz`, max `45 s`, `1/1`, target/minimum/stop index `4`. Accept only
  clean index `4`, finish `-1`, cadence `[50,100) Hz`, matching hashes, zero
  collision/invalid/dropout/drain/rate issue, one exit disarm and passive
  proof. Reject without retry; FullLap remains unauthorized.

### Candidate 025 rejected at Gate 3; N165 early brake offline passed

- One run/no retry: PID `24476`, detected reset `291494 -> 3403 ms`, race
  start `3269 ms`, detection `0.515 s`, calibration `60/60`. Official Gates 1
  and 2 passed; Gate-3 collision ID `1001`, threat `2`, impact
  `8.014646530151367 m/s`, index `2`, finish `-1`. Gate 4 was not reached.
- Publication `601/10.203 = 58.806 Hz`, duration `10.328 s`, zero health/rate
  issue; `3246` messages, `1639` IMU, `131` frames, `130` detections, one exit
  disarm. Aggregate/attempt/smoke/CSV/passive hashes:
  `fc3c345b5e4b60555076b438fd7eb53c0101fb0cf03c7dd2e3e3e53872d1442a`,
  `c4ef5963712dfe7d890c7d2b7c05dfbae93541b5fb026b8d09f1df3467f0b53e`,
  `a60c4318a2112003911046001e1437e5f5a24030694fa87e977a87489d8d9ea1`,
  `89069e751e2860d543954cbda6dd915ba82fc8b0baedebea5e672788e5a7fac9`,
  `493f7011c1022dd6d50c52bffc8cc6d070083959f31f22718469ef2818b7631d`.
  Passive base/status `65/3`, index `2`, finish `-1`, no reset.
- N164 latched at `4.156 m`, projection `-0.490/-0.212 m`, just `0.359 s`
  before collision; four outputs correctly held pitch/roll `-0.2/0.0`.
  Diagnostic hash:
  `ef20e8a683f0b53f5591a348c0c6c458d3fe38dfe3db86a10e552384bd72d292`.
- N165 adds only a Gate-3 brake-only latch: visible/closing inside `12 m`,
  projected lateral/vertical unsafe outside `0.5/1.0 m`, pitch cap `-0.2`,
  exact learned roll/thrust/yaw until N164's terminal level latch. Eight
  archived traces activate at `10.438..11.992 m` for `6..13` early samples;
  Candidate 024/025 gain `0.719/0.735 s`; zero non-target/output-law error.
- Gate-2, N164 terminal, and Gate-4 preservation pass. Policy/verifier/
  evidence/Gate-2/terminal/Gate-4/wrapper hashes:
  `4cca1c0e82c947962f5b99bb4f70d9f3f352d113e104ca27299ba92b4f7979e0`,
  `0b87f7cdbbaaee360caf9761a2915c5a250de799db809b5b247a6919fbd25b16`,
  `88f64d1360e2061aea9a2ede67f43a27f0b05d30f8efcffe4e1671f29fc50277`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `3e5d4d8e64e4d36453cbfe7dac753a77cea5deff8d07073747a605c474aca564`,
  `8768d67b497b5096a14b56d69cc0f82bb1b4cd93d13b9cb13063ecf5712d3ee1`,
  `abba4d696d613bee00f086224d71629daeb68ad67bfb2a240786f1eee667cd2f`.
  Linux/Windows `62/62`; compilation/parsers pass.
- Next: same-PID recovery, one reset-only `31000`, then only passive
  `n165_six_gate_hybrid_shadow_028`. No flight/FullLap before acceptance.

### N165 setup and passive Shadow 028 passed; Candidate 026 bounded

- Recovery reused PID `24476`, restored VQ1/R1 at base/status `193/4`, index
  `0`, finish `-1`, with no relaunch/reset/flight. Recovery hash:
  `d414d98e2885a0cd6becde4046d1768684beaa7ed6124cb7a194d8d1a0af24a9`.
- One reset-only `31000` rolled boot `920822 -> 3218 ms`; race start `3301 ms`,
  zero collision, Gate 1 visible (`210` detections, first `0.85875`), no
  setpoint. Setup hash:
  `4207e9612c22cd853586d3f953b3bdc9a97bdab82767749d2c1e057b173ba74d`.
- Shadow 028: zero reset/arm/setpoint/disarm; `256/256`, max error
  `1.899723053033764e-7`, inference `25.6 Hz`, all hashes, no-op
  `601/9.985 = 60.09 Hz`, index `0`, finish `-1`, zero health/rate issue;
  `2419` messages, `1245` IMU, `142` frames, `123` detections. Hashes:
  `ca88b471ebee77a361de19b022459bab67e5a49c63174c3564e80558a25cd62f`,
  `76e3a4cb1703a25149da3ce848014ecce09a6cf9edeb94fa05e2e9cb9cd53dbe`.
- Authorize exactly one `n165_six_gate_hybrid_gate4_bounded_026`: pinned
  wrapper/Windows Python, reused PID, detected `31000`, `80 Hz`, max `45 s`,
  `1/1`, target/minimum/stop index `4`; strict clean health/hash/disarm proof.
  Reject without retry; FullLap remains unauthorized.

### Candidate 026 rejected at Gate 3; N166 vertical floor offline passed

- One run: reset `154304 -> 3422 ms`, race `3306 ms`; official Gates 1/2,
  then Gate-3 collision `1001`, threat `2`, impact `8.48799991607666 m/s`,
  index `2`, finish `-1`. `641/10.594 = 60.412 Hz`, duration `10.688 s`, zero
  health/rate issue; `3344` messages, `1688` IMU, `135` frames, `130`
  detections, exit disarm `1`. Aggregate/attempt/smoke/CSV/passive hashes:
  `7794f37b85547c7a57787c81ea6e4583e87f85d6e0907d0b42ad35e00d87a41f`,
  `f0a2d415e149744800814df77c5c792a2957ec600f0c3f7181193fb0ffec6112`,
  `a83113a633180c39916ec12d19d1892c8936d144c1625633cecf370e3242d506`,
  `286ddc27b4b839bcdbbd3ef570a14ca4593734a9b41f83b29b43bf18a58aefc0`,
  `11a6d2109ff4b3d45032d4788930005d64dad6a7d455d845d7ce6d08eaff9591`.
- Early activation at `9.751 m` had projected down `-1.835 m` but below-hover
  thrust `-0.210`; terminal at `4.320 m` projected down `-0.908 m`, then
  `-1.065 m`. Early/terminal diagnostics:
  `17f54f00d1a917fcb2c8db73b230e11e7e12040eeb8165356cb94de73cbdda4d`,
  `3b94636ea36320a7f742edf378300c08d6c42112356ba14400cb1603139f7bde`.
- N166 adds only a latched hover-thrust floor (`0.0` normalized/`0.27`
  physical) on Gate-3 projected down below `-1.0 m`. It preserves roll/yaw and
  positive thrust. Nine traces activate `9.751..11.992 m`, `9..13` samples;
  only below-hover samples change, Candidate 026 six; zero law violation.
- Policy/verifier/evidence/Gate-2/terminal/Gate-4/wrapper hashes:
  `f17479db4c2bfcadb06295c2344fc9565f0c5044b465cbb9630f25ab2c689115`,
  `978c1c4c9af6ac468f24b0e18279c95c655abd3ffdf7f38b8cf3596ac5647605`,
  `37b625628428e5fa1cb0ba051421bb052c6767a891172be8c7e0ca671492faa7`,
  `768a7d135670c159b177a257419fbfc1b768cd12ed958b77bec13e00467c2c15`,
  `d40089d2532bb0cd458c1a925332b63214bd0d0aeee13b40619bd7d2605080ba`,
  `7dbe6d6dff49091aee2070079580e6d99f4edc0af00342c8f9bbec4c197ba630`,
  `a5a429a6718b3940f61026c3c8e4ae61788cb0425ea563898b67fe447c867c44`.
  Linux/Windows `63/63`; compilation/parsers pass.
- Next same-PID recovery, one reset-only `31000`, then only passive
  `n166_six_gate_hybrid_shadow_029`. No flight/FullLap before acceptance.

### N166 setup and Shadow 029 passed; Candidate 027 bounded

- Recovery hash
  `2028b6bcf6e2f376b5430df995369d3f13d79ea41068188870805e36c1371bdc`;
  one reset `824284 -> 3009 ms`, race `3282 ms`, index `0`, finish `-1`, zero
  collision, `193` Gate-1 detections, no setpoint; setup hash
  `9946313de517ea51c82768c67c33c67224956f518a9ebac5dc8c30dc1b2db560`.
- Shadow zero control lifecycle; `238/238`, max error
  `1.574008941518379e-7`, inference `23.762 Hz`, all hashes, no-op `52.2 Hz`,
  index `0`, finish `-1`, zero health issue; `2414` messages, `1242` IMU,
  `124` frames, `107` detections. Hashes:
  `64fe8cdfdb4a71c22c4b4884bbd751deb4f473f7a6fefcc5571c2c207fb98c9a`,
  `2470e798abb9247e047b0dc7aaa5a06447b9b67dd26dd2d03347b447f0c0e25a`.
- Authorize exactly one `n166_six_gate_hybrid_gate4_bounded_027`, pinned
  wrapper/Windows runtime/reused PID/detected `31000`, `80 Hz`, `45 s`, `1/1`,
  target/minimum/stop index `4`; strict clean proof. Reject without retry;
  FullLap unauthorized.

### Candidate 027 rejected at Gate 3; N167 pitch-brake retirement passed offline

- Candidate 027 ran once and is rejected without retry. One normal MAVLink
  `COMMAND_LONG` command `31000` reset reused PID `24476` with measured boot
  rollback `126297 -> 3389 ms`, race epoch `3316 ms`, detection `0.516 s`, and
  calibration `60/60`. It passed official Gates 1 and 2, then collided at
  active Gate 3/index `2`: ID `1001`, threat `2`, impact
  `6.9826836585998535 m/s`, finish `-1`. Vision's third apparent pass is
  diagnostic only.
- The last Gate-3 detection was centered/level at about
  `2.248/+0.011/-0.105 m`, image center `(321.5,165.0)`, but pitch remained
  forced to normalized `-0.2`. Delivery was `638/10.281 = 61.959 Hz`, duration
  `10.390 s`, with zero rate/dropout/drain issue, `3283` messages, `1658` IMU,
  `133` frames, `130` detections, and exit disarm `1`. Aggregate/attempt/smoke/
  CSV/passive hashes are
  `ce8ec534a079cc4d3d79b7b898427bedcfa6e1c7e05760a1ef4f8045a9ad07be`,
  `2b7d5e25d035c11d39f8a6af66f1543a210ffc7cc317e188003270448bc0f6ed`,
  `d48fc9fa6ab2a306aca602010a521933ef86f720b8d2061713541869d9993f66`,
  `57ebceff8b4321ff8c4ecd712e51840c6b41ab41f322f826a1d97abc4fc2ebc2`,
  `9d64a4de0f5accda2cb2e372c4cb37ce5a7ba4623708b154484bae4f9522339a`.
  Passive state is base/status `65/3`, index `2`, finish `-1`, no reset.
- N167 removes only both Gate-3 forced pitch-brake assignments and restores
  recurrent pitch. It retains terminal roll leveling and the projected
  negative-down hover-thrust floor, plus every Gate-2/Gate-4 behavior. Five
  official Gate-3-passing archives contain `18/18` positive-pitch close
  samples; the N164--N166 failures contain `9/9` forced `-0.2` close samples.
  Candidate 024 still exercises two terminal counter-roll removals. Across all
  nine source-hashed traces, N167 changes no pitch/yaw; vertical/terminal/
  Gate-2/Gate-4 preservation reports pass.
- Policy/verifier/evidence/vertical/terminal/Gate-2/Gate-4/wrapper hashes:
  `91ceaf1a14f59610496f2116b69cb86c2f5c90ecc372bcc4fe1c4ea03d43cd2b`,
  `12786cca8b346893fd1edd042776c9fd0ed126fd89d842d9c808cdbd84a04a6a`,
  `6387e509ef3db61615702c7364ce75b4a66d584c3668f6fbd626647bf35b96ba`,
  `a58cb6bb852d238d6d1afb4905aeb083c82eeff6b03f06ea21cada4d575fce97`,
  `56e52245ae102a5dc360414516488a3d8814ca2740a33dfa74544b36ad6d8c88`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `d707f0cb30034c9f642e4e0d44f2e5316a4969a79b5b2c3a60982a2f0acc2df7`,
  `2c3805f9033c309d8b9971d37d6d6cb949f7ac868a45ec108ac06b0bd733b762`.
  Linux/Windows targeted suites pass `63/63`; compile/parsers pass.
- Next: same-PID reset-ready recovery, one reset-only command `31000` with
  measured rollback, then passive zero-command
  `n167_six_gate_hybrid_shadow_030` only. Candidate 028 and FullLap remain
  unauthorized until that shadow passes.

### N167 setup and Shadow 030 passed; Candidate 028 bounded

- Same PID `24476` recovered without relaunch/control to base/status `193/4`,
  index `0`, finish `-1`; recovery hash
  `643e1a8c377ea29071dcc20ed7766c51131bb25122d651402abf5db52d0ebf25`.
  One reset-only command `31000` rolled boot `1527982 -> 4338 ms`, race start
  `3309 ms`, index `0`, finish `-1`, zero collision, and `251/251` Gate-1
  detections. Setup hash:
  `51f981ea99371b392097fe2e73ce3ce2372c6023c4a7210c6a21524032fe88fb`.
- Shadow 030 sent zero reset/arm/setpoint/disarm; parity `307/307`, maximum
  error `4.644851684387774e-7`, inference `30.7 Hz`, all 11 hashes, no-op
  `605/9.969 = 60.588 Hz`, index `0`, finish `-1`, zero health/rate issue;
  `2435` messages, `1254` IMU, `122` frames/detections. Shadow/parity hashes:
  `ae86706ee7258effe10a4f1fff300b06d16165144d03b02972525fea3dfb966f`,
  `74cc002c8d0738c2f8f80b26b66ef0619c11a0a8361f3d8eca37df23528aebdf`.
- Authorize exactly one `n167_six_gate_hybrid_gate4_bounded_028`, pinned
  wrapper/Windows runtime/reused PID/detected `31000`, `80 Hz`, maximum
  `45 s`, `1/1`, target/minimum/stop index `4`; strict clean hash/health/
  disarm proof. Reject without retry. FullLap remains unauthorized.

### Candidate 028 collision after official Gate 3; N168 offline passed

- Candidate 028 ran once and is rejected without retry. Reset `31000` rolled
  boot `188257 -> 3422 ms`, race `3295 ms`, detection `0.500 s`, calibration
  `60/60`. Collision `1001`, threat `2`, impact `4.625654220581055` aborted on
  the runner's stale index `2`, but command-free passive telemetry later proved
  official index `3`, last-gate time `10.754043579 s`, finish `-1`, base/status
  `65/3`, no reset. N167 therefore passed Gate 3 while still striking the frame;
  Gate 4 control never activated.
- `604/10.297 = 58.561 Hz`, duration `10.625 s`, zero rate/dropout/drain issue;
  `3311` messages, `1673` IMU, `130` frames/detections, exit disarm `1`.
  Aggregate/attempt/smoke/CSV/passive hashes:
  `6eaf8c40deaf0f52981d07d338a375934e333296b01583269e13d1e49996eb2e`,
  `78f1be2c82d1292f9d8552f0a9f32a80c5b9bc8975cdd363c7af742d0c6a2cf3`,
  `a4d4d7dfd37209fa4a410c54c59c4793d074c8fba832cdd1e63fa7dcf8a70268`,
  `5c6cd4fcbce3e88adc2109f92f528050f37f6e45d2b09580391c05d90d2da90e`,
  `e5abeada7ff2a60ff44efc38d52e49857577b861e2de0180aa37f5dbd784cffc`.
- N168 extends only left-side negative counter-roll suppression from `4` to
  `7 m`; adaptive positive boost remains inside `4 m`, and every non-roll/
  other-gate law is preserved. It changes zero samples on five clean Gate-3
  pass paths, exactly two Candidate-028 samples at `6.957/6.447 m` from
  `-1 -> 0`, and one sample each in rejected `019/024/026`, with zero non-roll
  error. The passive official index-3 proof is embedded in evidence.
- Policy/verifier/evidence/pitch/vertical/terminal/Gate-2/Gate-4/wrapper hashes:
  `7a3783cdf65800c6ea4ae98f070c95a3c96c7c3f27b827f6142ad779862506b5`,
  `a6f5cf6d0c1e28a2419b2f48faa356e54305f2a5e9138bec737b3b9c27838cee`,
  `40ae0bd2e9a7f6de4302364c85f815bff0614eb4468c8d5d2636ccca42ce6e35`,
  `bdb97fa67ed879a8efcfe823012add794f30594a4c157d0ed9db13500aa4b5d5`,
  `77b21226b0f7b5a53b0f57da1c8687e79945fb63709381f56af1df9c21369a74`,
  `4804e7d0d7118ba00e41e05914f19e617bc48241d2c132b73c5b74822987beca`,
  `daa702f1e2b0cafc89c6c074c68f254d9841faea42a579d41ad6adc09ba667ca`,
  `4596e55191cb70998dc584005b5c3f970cc61b7ed61ce42796773d06b5e22838`,
  `c75d9eb1a1255f89a0bac19232e28cb108559da6c8a09ef15fe85226d12c972a`.
  Linux/Windows `63/63`; compile/parsers pass.
- Next only: same-PID recovery, one reset-only `31000` with proof, then
  zero-command `n168_six_gate_hybrid_shadow_031`. Candidate 029 and FullLap
  remain unauthorized before shadow acceptance.

### N168 setup and Shadow 031 passed; Candidate 029 bounded

- Recovery/setup reused PID `24476`: recovery hash
  `5e27125029535437aa15b2d628dd847ca82ec0e2a512a74e3637c9a55fe3e70f`;
  one reset-only `31000` rolled boot `873911 -> 4307 ms`, race `3300 ms`,
  index `0`, finish `-1`, zero collision, `264` Gate-1 detections; setup hash
  `f082f0b0d134a735db2114767dd45e403e9c1c12d8140d6b55146d015d17906c`.
- Shadow zero lifecycle/control; `281/281`, max error
  `2.5453186036639153e-7`, inference `28.1 Hz`, all hashes, no-op `57.086 Hz`,
  index `0`, finish `-1`, zero health issue; `2427` messages, `1250` IMU,
  `119` frames/detections. Hashes:
  `6a1957ade87b45e1a1827c8f79d6e2591f7679ceabb62630dd8026bfaca64cc1`,
  `22b0551ae2866e27d8899210d3ebbdc34013aedef5a86b40a5c9721d0e0d6d9c`.
- Authorize one `n168_six_gate_hybrid_gate4_bounded_029`, pinned Windows
  runtime/reused PID/detected `31000`, `80 Hz`, `45 s`, `1/1`, target/minimum/
  stop index `4`; strict clean proof. Reject without retry; FullLap disabled.


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


## N191 Candidate 051 — Gate-3 Terminal-Attitude Collision

- Candidate 051 ran once on reused responsive PID `24476`; no simulator process
  was launched. The tracked reset sent normal disarm, waited `100 ms`, then
  sent MAVLink `COMMAND_LONG 31000` with confirmation and params 1--7 zero.
  Reset was detected in `0.516 s`: boot rolled `247975 -> 3412 ms`, fresh
  race start was `3317 ms`, index `0`, finish `-1`.
- Gates 1 and 2 passed before official Gate-3 collision `1001` (threat `2`,
  impact `10.746137619018555`) at about `10.25 s`. The run stopped at index
  `2`, finish `-1`. Transport was healthy: `628/10.109 = 62.024 Hz`, zero
  rate/dropout/drain/malformed faults, `3246` telemetry messages, and
  `129/129` frames/detections. Command-free post-stop proof confirmed
  base/status `65/3`, index `2`, finish `-1`.
- N191 severe and moderate Gate-2 activations were both zero, so Gate 2 passed
  on the unchanged N189 path. Near Gate 3, N189 admitted at `11.549 m`,
  right `-2.927 m`, right rate `+1.630 m/s`, then commanded terminal roll
  `-0.067/-0.467/-0.800` at `3.664/3.266/2.712 m`. The projected offsets
  were already within the aperture, isolating continued terminal bank as the
  next hazard. Do not use the generic N189 replay branch counts: that replay
  applies N189 again to already-governed actions.
- Smoke/attempt/summary/post-stop/Gate-2/Gate-3/transition SHA-256 values are
  `620a71944b57d7278d6038294709f0596feb9f0b0c41002d519e5e79d6c6828e`,
  `328140427c05274c7df417684843e1a4e6056cdc5e89ea3e5c5fd58f51d16753`,
  `f8145cd38e6673c8059c1676961977787f9752895b9d88faa76519794d92d3e3`,
  `19670708617bdaa0e0e47463d29daf94025139b309fdde7f2d2b27979c52ac77`,
  `e3e08ac31657582d67e4f28fa776b985da4737690d2eb642df3199a934293483`,
  `b2e4b0bca5dc3ba2f2aaa60ab1b15e3a6ab0905f509d9604880a89835a25a629`,
  and `ce842d509ec7a7a6f20b8cb935f340125707f8c5b728b4311f4b62f4ebd3f38d`.
- Candidate 051 is rejected without unchanged retry. Candidate 052 and FullLap
  remain unauthorized pending an offline-separated, source-pinned Gate-3
  terminal-attitude mechanism, tests, exact reset proof, and passive shadow.


## N192 Gate-3 Terminal Safe-Level — Offline Freeze

- N192 evaluates exact source-pinned N191 once, then reuses N184's already
  tested terminal-aperture projection as a final Gate-3 roll-only governor.
  While visible, closing at least `1 m/s`, and within `6 m`, it projects
  right/down to the `2 m` terminal plane. If absolute projected right is at
  most `0.5 m` and absolute projected down at most `1.0 m`, it latches
  normalized roll to `0.0` until official gate transition. Pitch, thrust, and
  yaw are unchanged; N191's Gate-2 rules and N189's preceding Gate-3 control
  remain exact.
- Offline replay passes with no blockers. It changes zero actions on protected
  Gate-3-pass Candidates 016, 018, 020, 021, and 022, and zero actions on
  recent fixed-point Candidates 045 and 048. On Candidate 051 alone among
  those current paths it changes exactly three roll commands, first at
  `9.907 s` and `3.664 m`, where projection is right `-0.255 m` and down
  `-0.072 m`; it replaces `-0.067/-0.467/-0.800` with level roll. It also
  activates on older failed terminal-bank traces 019, 024, 040, and 041. That
  supports mechanism separation but is not counterfactual pass proof.
- Actual Windows focused controller/verifier/composition/runner tests pass
  `28/28`, and the PowerShell runner parses. Policy/verifier/offline-report/
  runner SHA-256 values are
  `3750e1620f4a7d244b93c370f31ec135a89572e64a60888d7ea032c244949fae`,
  `34c77c5357f6c33ca34298e4cb93dea0f3dfbd83f5b47c141088b22b9b893cc5`,
  `17c2d14cbae9bbb935030ceb881b4b29ce406f6da3b963e52a98afabc9ce0b08`,
  and `7e6f5faf7bdf72c87b9e9b9843065f5dd09b8df83feb60996b207d82c1cef2e0`.
- No simulator control command occurred during N192 development. Candidate 052
  and FullLap remain unauthorized. Next only: keep reused PID `24476`, recover
  sole VQ1/R1 with `-NoRelaunch -SkipLoginClick` only if inactive, then use
  the current tracked normal-disarm plus learned-target MAVLink
  `COMMAND_LONG 31000` reset. Require boot rollback over `1000 ms`, fresh
  nonnegative race start, index `0`, finish `-1`, and visible Gate 1. Then
  run passive zero-command Shadow 053 with `-Gate3TerminalLevelN191`; it must
  send zero reset, arm, setpoint, and disarm commands and pass exact hashes,
  parity, cadence, telemetry, camera, and fault checks before a separately
  preregistered bounded flight.


## N192 Live Reset Proof and Passive Shadow 053 — Passed

- The simulator remained responsive on the original PID `24476`; no process
  was launched. Candidate 051 had stopped inactive at base/status `65/3`, so
  the sole VQ1/R1 UI was recovered with `-NoRelaunch -SkipLoginClick` while
  preserving the process. The command-free ready capture then showed
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, and old
  boot/race epoch `963417/954276 ms`.
- The current tracked reset learned target IDs, sent normal disarm, waited
  `100 ms`, and sent exactly one MAVLink `COMMAND_LONG 31000` with
  confirmation and params 1--7 zero. Boot rolled `963417 -> 3069 ms`; the
  fresh race-start epoch was `3286 ms`. A subsequent command-free proof
  showed boot `18362 ms`, base/status `193/4`, index `0`, finish `-1`,
  visible Gate 1, healthy telemetry, and no collision.
- Passive `n192_gate3_terminal_level_n191_shadow_053` passed while sending
  zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence was
  `602/10.0 = 60.1 Hz` with zero violations. All `294/294` trace samples
  replayed at `29.353 Hz` with maximum action error
  `2.522184371911429e-7`; every deployment hash matched. It observed `2431`
  telemetry messages and `124/124` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `90af19972fa8580291e777997cf9aff12196983ccf3423fb8b7b0d0c2ff85239`,
  `25aa8ce73568901c808790802310a3f432fb74704060cd6b8d7e12dd78f065f2`,
  `f38664acca9b8980e1e841d3b1c3a631a9f9f2221f99094fa91ab5c7d16833ea`,
  `f90be3a8552cb58e09b098c6a35169a8af8bf093db839d94c3ea41cd97906d7b`,
  and `114e715ac0bb317e78b133f6caa8c54b71aea95db72e1d3f3fdc94ac0c7a9e5d`.
- Authorize exactly one Candidate 052 `BoundedGate3` under tag
  `n192_gate3_terminal_level_n191_bounded_052`: frozen N192 hashes, actual
  Windows Python, reused PID `24476`, one detected normal-disarm-then-command-
  `31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `3`. Accept
  only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact hashes,
  zero rate/collision/invalid/dropout/drain fault, one exit disarm, and a
  command-free post-stop proof. Reject failure without unchanged retry.
  FullLap remains unauthorized.



## N192 Candidate 052 — Gate-2 Collision Before N192 Activation

- Candidate 052 ran exactly once on reused responsive PID `24476`; no process
  launch. The tracked runner sent normal disarm before exactly one learned-target
  MAVLink `COMMAND_LONG 31000` with confirmation/params 1--7 zero. Reset was
  detected in `0.5 s`: boot rolled `130379 -> 3388 ms`, fresh race start
  was `3305 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Only Gate 1 passed officially. Two camera-range pass detections were counted,
  but the authoritative index remained `1`; they are not two official gates.
  Gate 2 then produced collision `1001` (threat `2`, impact
  `8.721649169921875`) at `7.39 s`. The run stopped at index `1`, finish
  `-1`. Transport was healthy: `449/7.265 = 61.666 Hz`, zero rate,
  dropout, drain, or malformed-message faults, `2571` telemetry messages,
  `94` frames, and `91` detections. Command-free post-stop proof sent no
  reset/control command and confirmed base/status `65/3`, index `1`,
  finish `-1`.
- Neither N191 Gate-2 floor activated or changed a command, and there were zero
  Gate-3 samples, so N192 did not activate. This run therefore supplies no live
  evidence for or against terminal leveling. The last visible Gate-2 state was
  `2.183 m` forward, right `+0.481 m`, down `-0.100 m`, closing
  `7.924 m/s`, right rate `-0.696 m/s`, and down rate `-3.332 m/s`.
  Candidate 051 passed Gate 2 from a nearby lateral/yaw state but a materially
  smaller vertical rate (`-2.577 m/s`), isolating a fast terminal vertical
  deficit for offline study.
- Smoke/attempt/summary/post-stop/Gate-2-diagnostic/transition-comparison
  SHA-256 values are
  `12327cb380fd4b81b245a689432c2d192f9d9e0f345b164edb3db928ffb6a186`,
  `c761d22d3b2023092bd16548ec7d266121906cd3a8ad49d788a4d984ae7e0f6e`,
  `bd7262bc42d98e68df5f4fbb4f30f7b76a21cc67e99ade874e40bfa3287a0929`,
  `6389bc8aa428d16a6aa128f5d1a2a6dd0570d3a4a816da03b463bf270f474e6f`,
  `33327335f1be632b5952b4ddee26f72235c61de2f3494f09d68fd62223c70dce`,
  and `a91d97a92d4b7956ab3e5702469ab5587d6db5f20ad997752aff49665cac6cd0`.
- Candidate 052 is rejected without unchanged retry. Candidate 053 and FullLap
  are unauthorized pending an observable, offline-separated Gate-2 mechanism,
  hashes, focused tests, exact reset proof, and passive zero-command shadow.


## N193 Rapid Gate-2 Vertical-Deficit Latch — Offline Freeze

- N193 evaluates exact source-pinned N192 once and adds one Gate-2 thrust-only
  response to the failure state unique to Candidate 052. While visible,
  closing, and between `3--7 m`, it triggers only when down rate is below
  `-2.9 m/s`, plane-projected down is below `-1.0 m`, and inherited
  normalized thrust is still at or below `+0.05`. It then latches normalized
  thrust to `+1.0` until projected down recovers to `-0.4 m`, visibility is
  lost, or the official gate changes. Pitch, roll, and yaw are untouched; all
  N191/N192 safeguards remain exact.
- Replay across Candidates 037--052 passes with exact activation separation.
  Candidates 037--051 have zero activations and zero changes. Candidate 052
  triggers once at `6.859 s`, `5.385 m` forward, down rate
  `-2.941 m/s`, projected down `-1.070 m`, and inherited thrust near zero.
  The latch changes exactly five consecutive thrust commands before the
  recorded collision, with zero non-Gate2, non-thrust, or invalid changes.
  This gives about `0.53 s` of counterfactual correction authority but does
  not prove a crossing.
- Actual Windows focused N187--N193 controller/verifier/composition/runner
  tests pass `35/35`; the PowerShell runner parses. Policy/verifier/offline-
  report/runner SHA-256 values are
  `9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2`,
  `5f1ace160d4a752a34a959fa5bdb3bd1e99713ecf966875e4138fdf37f3e2913`,
  `f5df695093a59d2ffae5a766c2b92681ab97b5322de90bc0bbb0078118920fec`,
  and `4391572c67f7bc4b6c332c06a3daf4c14c0e682eee90b8858e4eb9b6f431ba8d`.
  The broader vertical-deficit study and its analyzer hash to
  `ebbbb36023b77566bf11953c6cac8bf2d761871362626851cfbebd893ea9ff8d`
  and `d5a2efd02ee752cb74e55f91983e23994088924c14ac869f05234f76b7c25934`.
- No simulator control command occurred during N193 development. Candidate 053
  and FullLap remain unauthorized. Next only: keep PID `24476`, recover sole
  VQ1/R1 with `-NoRelaunch -SkipLoginClick` because Candidate 052 stopped
  inactive, then run the current tracked normal-disarm plus learned-target
  MAVLink `COMMAND_LONG 31000` reset. Require boot rollback over `1000 ms`,
  fresh nonnegative race start, index `0`, finish `-1`, and visible Gate 1.
  Then run passive zero-command Shadow 054 with
  `-Gate2RapidVerticalDeficitN192`; it must pass exact hash/parity, cadence,
  telemetry, camera, and fault checks before a separately preregistered flight.


## N193 Live Reset Proof and Passive Shadow 054 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 052 had
  stopped inactive, so `-NoRelaunch -SkipLoginClick` recovered the sole
  VQ1/R1 session on the same process. Command-free readiness showed base/status
  `193/4`, index `0`, finish `-1`, visible Gate 1, and old boot/race epoch
  `588707/568695 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `588707 -> 2963 ms`, fresh race start was
  `3283 ms`, and a later command-free proof showed boot `21754 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n193_gate2_rapid_vertical_deficit_n192_shadow_054` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `603/9.984 = 60.296 Hz`, zero violations. All `277/277` samples
  replayed at `27.659 Hz` with maximum action error
  `2.1852722167925442e-7`; every deployment hash matched. It observed `2427`
  telemetry messages and `124/124` detections with zero collision, dropout,
  drain hit, malformed message, decode failure, or missing quad.
- Ready/reset/reset-ready/shadow/parity SHA-256 values are
  `51b173069af8ccd25e199cef22261385281988805ea31b36604117340dd0ed48`,
  `c5bcb4931e31a139349b8e130bb908635a1e0401686d8455c547c57f0955935e`,
  `15e4fc1e10c8b265c0611aece169ef0fc9801dee854d19e9b6587f19c6fe7897`,
  `661951f8a22f7a73db6694edf9fcc8d9e2f9bdaba4833cfc8a9c4d2bb48ea711`,
  and `656ffe5eb255670a7d9372b939799a98e153e3c6d79398bc2d128659e87ba9ef`.
- Authorize exactly one Candidate 053 `BoundedGate3` under tag
  `n193_gate2_rapid_vertical_deficit_n192_bounded_053`: frozen N193 hashes,
  actual Windows Python, reused PID `24476`, one detected disarm-then-command-
  `31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop index `3`. Accept only clean
  index `3`, finish `-1`, cadence `[50,100) Hz`, exact hashes, zero rate/
  collision/invalid/dropout/drain fault, one exit disarm, and command-free
  post-stop proof. Reject failure without unchanged retry. FullLap remains
  unauthorized.


## N193 Candidate 053 — Clean Gate-3 Stale-Observation Stall

- Candidate 053 ran once on reused responsive PID `24476`; no process launch.
  The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset was
  detected in `0.5 s`: boot rolled `114460 -> 3414 ms`, fresh race start
  was `3295 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially. Gate 3 did not pass and the run ended at the
  `45 s` bound with authoritative index `2`, finish `-1`, and no in-run
  collision. Transport was healthy: `2771/44.875 = 61.727 Hz`, zero rate,
  dropout, drain, or malformed-message faults, `11114` telemetry messages,
  `574` frames, and `183` detections. N193's rapid Gate-2 branch did not
  activate because this path never reproduced Candidate 052's rapid deficit;
  N192's terminal-level rule also did not activate.
- The camera pass detector counted three range events, including a Gate-3 event
  at `10.453 s`, but official index remained `2`; detector events are not
  authoritative gate passes. The last real detection aged to `30.703 s` by
  run end. Meanwhile the observable Gate-3 motion state froze for `271`
  consecutive inference samples from about `15.609 s`: forward `2.059 m`,
  right `+0.972 m`, down `-0.780 m`, closing `8.520 m/s`, right rate
  `+1.546 m/s`, down rate `-3.261 m/s`, while inherited roll stayed
  `-1.0`. The same five-sample frozen condition occurs in no protected
  Gate-3-pass trace or Candidate 037--052. This is a stale held-observation
  control loop after a missed crossing, not a live target approach.
- The command-free post-stop capture sent no reset or control command. After
  exit disarm, v3385 had already cleared the race epoch to start `-1`/index
  `0` and reported low-threat contact `1001` with impact `0.03224`; this
  is passive post-exit ground contact and is not substituted for the smoke
  report's zero in-run collisions.
- Smoke/attempt/summary/post-stop/transition/terminal-level/control-sequence/
  frozen-study SHA-256 values are
  `3e5ade2e0b9fb9d37d295673d0aefe43de021b00750bb3f06b065643f5a5539b`,
  `b8c39393b715ada7e014e16dfd0c48078dd46d2adc19c3717d75707f1f9a82a1`,
  `99b8eecd122a26570c9edba4bdd357b5f6a35fe49eeee54a34ad0a738899fc96`,
  `e54d0ff92f1417ada5d73c12604ec3298f2c5a5bb785d0658f35f1c15db36277`,
  `3ce1794e37d4b5623cf9f3b92228a26f431e7b3e617ac78b19f3b5e1d848b877`,
  `5de54f9a4eb942cd3466085c9cb3fda6b23d2d6a06d32b43212d8ee60d333c9f`,
  `c445747a70f76a5af4af408a010dea70d3f22b3e8b91b26a93f14c4d184e0cbc`,
  and `a0b6506edbb03e1817ef4affe70f8ba4700b8aa523817c21e2ab094243ce2ded`.
- Candidate 053 is rejected without unchanged retry. Candidate 054 and FullLap
  remain unauthorized pending an offline-separated stale-observation recovery,
  hashes, tests, exact reset proof, and passive zero-command shadow.


## N194 Gate-3 Frozen-Observation Recovery — Offline Freeze

- N194 evaluates exact source-pinned N193 once and adds one recovery for the
  Candidate-053 failure state. On official Gate 3 only, while visible,
  closing at least `1.0 m/s`, and within `3.0 m`, it watches features
  `0,1,2,11,12,13,14`. Five identical consecutive samples within
  `1e-7` activate recovery: normalized roll becomes `0.0` and yaw scans
  `0.25 rad` toward the last measured bearing. Inherited pitch and thrust
  remain exact. Any fresh feature motion, visibility loss, invalid approach
  state, or official gate transition releases the recovery immediately.
- Offline replay passed with exact separation. Protected Gate-3-pass Candidates
  016, 018, 020, 021, and 022 plus Candidates 037--052 have zero activations
  and zero changes. Candidate 053 activates once at `15.609 s`, on the fifth
  frozen sample, and changes exactly `267` roll/yaw commands through the end
  of its 271-sample frozen run. There are zero non-Gate3 changes, zero pitch or
  thrust differences, and zero invalid outputs. This is counterfactual recovery
  authority, not proof of Gate-3 reacquisition or passage.
- Actual Windows focused N189--N194 controller/verifier/runner tests pass
  `26/26`; the PowerShell runner parses. Policy/verifier/offline-report/runner
  SHA-256 values are
  `a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7`,
  `a92d2159698b9f19336a160d99d11f5eb865d4f32aeaebd5d3b0f4451a563178`,
  `42f0bed223539702b5109954451a523803d30051c25ec1ca34526ced77d8b620`,
  and `92f14881dfe55e57c61badeff2ffe752e77d6d30abc15d02d9f0079783e23737`.
- No simulator control command occurred during N194 development. Candidate 054
  and FullLap remain unauthorized. Next only: preserve reused PID `24476`,
  recover the sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` if inactive,
  then send normal disarm followed by exactly one learned-target MAVLink
  `COMMAND_LONG 31000` with confirmation and params 1--7 zero. Require boot
  rollback over `1000 ms`, a fresh nonnegative race epoch, index `0`, finish
  `-1`, and visible Gate 1. Then run passive zero-command Shadow 055 with
  `-Gate3FrozenObservationRecoveryN193`; only a clean hash/parity, cadence,
  telemetry, camera, and fault pass may authorize a separately preregistered
  bounded flight.


## N194 Live Reset Proof and Passive Shadow 055 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 053 had
  stopped inactive at base/status `65/3` with race start `-1`; the sole
  VQ1/R1 UI was recovered using `-NoRelaunch -SkipLoginClick` on the same
  process. Command-free readiness then showed base/status `193/4`, index
  `0`, finish `-1`, visible Gate 1, and old boot/race epoch
  `854041/840827 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `854041 -> 3163 ms`; fresh race start was
  `3303 ms`. A later command-free proof showed boot `21198 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n194_gate3_frozen_observation_recovery_n193_shadow_055` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `607/9.985 = 60.691 Hz`, with zero violations. All `307/307` trace
  samples replayed at `30.7 Hz` with maximum action error
  `2.627761840900966e-7`; every deployment hash matched. It observed `2424`
  telemetry messages and `125/125` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `2c960abeff7389128b3fe8a2ceff3ca95bbf38e2b80611d25d306adfba8a56bc`,
  `71f187a5f5916ff0e8f81e187d029c4c7473c8f683cb3be6516eb2660cf2ba07`,
  `704a30c02ca8d7ba34fec862762da0a13f71b6d25dfd796b8162ff7df700bd73`,
  `efb732f201d7fbd2da75effd48c938ced36bd748581499c2fc988af3c28d3b1a`,
  and `641710fdd64cbd683b3e47fcb6e313d59ec4fef8cca12da5635a3de0bbbbf06c`.
- Authorize exactly one Candidate 054 `BoundedGate3` under tag
  `n194_gate3_frozen_observation_recovery_n193_bounded_054`: frozen N194
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence
  `[50,100) Hz`, exact hashes, zero rate/collision/invalid/dropout/drain
  fault, one exit disarm, and a command-free post-stop proof. Reject failure
  without unchanged retry. FullLap remains unauthorized.



## N194 Candidate 054 — Clean Gate-2 Frozen-Observation Stall

- Candidate 054 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `232191 -> 3406 ms`, fresh race
  start was `3291 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gate 1 passed officially at race time `4.847601890 s`. Gate 2 did not pass;
  the attempt reached its `45 s` bound at authoritative index `1`, finish
  `-1`, with no in-run collision. Transport remained acceptable:
  `2698/44.906 = 60.059 Hz`, zero rate violations, `11089` telemetry
  messages, `560` frames, `106` detections, one allowed telemetry dropout,
  and zero drain-limit or malformed-message faults.
- The last real detection aged to `36.328 s` by run end. Despite that, the
  policy observation continued to claim Gate 2 visible and froze exactly for
  `312` consecutive logged inference samples from `9.891--44.860 s` at
  forward `4.183 m`, right `+0.821 m`, down `-1.351 m`, closing
  `4.296 m/s`, right rate `-0.690 m/s`, and down rate `-3.531 m/s`.
  The action also froze near pitch `-0.2`, roll `-0.251`, thrust `+1.0`,
  and yaw `-0.00095`. The five-sample Gate-2 frozen condition occurs only in
  Candidates 050 and 054 among Candidates 037--054; both are clean Gate-2
  stalls, while all protected Gate-2 passes remain inactive. N194's Gate-3-
  only recovery never activated and therefore received no live test.
- The command-free post-stop proof sent no reset or control command and showed
  base/status `65/3`, official index `1`, finish `-1`, and no collision.
  Smoke/attempt/summary/post-stop/vertical-study/frozen-study SHA-256 values are
  `7901d9eaaae4d8bb01a70b076aba5306845c0635313ee5f16092b1666e79513d`,
  `f74d95d1113f29ba531fa0e10607c4a8d6c148e35cd98262e284ea3be5637da9`,
  `555fef017744375a8121056b49dc715dba5cf1deef8be834c4057517bc78613e`,
  `54f74d1e9586268e1a4a87f8ece50a4785783ae43ff91c04e4570f60e3eaffcd`,
  `fbec238a75b79d066f6b5b8cc1e3d6ad9f1ec7562af50c5d987b8afefd403850`,
  and `819d5e1e0a58ad6709ed8eb82e021066990dfeb9c6b17e9d05f460cb03f402f3`.
  The Gate-2 frozen analyzer hashes to
  `578cad0bfb39eb1a7547f09c058666caa3daa9e96ecf848b3b72714191b84f17`.
- Candidate 054 is rejected without unchanged retry. Candidate 055 and FullLap
  remain unauthorized pending an observable, offline-separated Gate-2 stale-
  observation recovery, pinned hashes, focused tests, exact reset proof, and
  passive zero-command shadow.


## N195 Gate-2 Frozen-Observation Dropout — Offline Freeze

- N195 evaluates exact source-pinned N194 and changes no actuator channel
  directly. On official Gate 2 only, while the observation claims visibility,
  closes at least `1.0 m/s`, and is within `6.0 m`, it watches features
  `0,1,2,11,12,13,14`. After eight equal samples within `1e-7`, it changes
  only `gate_visible` from `1.0` to `0.0` before N194 inference. This
  releases N193's visibility-gated thrust latch and invokes the recurrent
  prefix's existing trained vision-dropout behavior. Fresh feature motion,
  raw visibility loss, an ineligible approach, or a gate transition releases
  the sanitizer immediately.
- Offline replay across Candidates 037--054 passes with exact separation.
  Candidates 037--049 and 051--053 have zero activations and zero changes,
  including every protected Gate-2 pass. Candidate 050 activates once at its
  eighth frozen logged sample (`11.594 s`) and sanitizes `303` samples;
  Candidate 054 activates once at `10.547 s` and sanitizes `306` samples.
  Every changed sample is Gate 2 and field 10 only; all other observation
  fields remain bit-identical and all outputs are finite. This demonstrates
  stale-input removal, not counterfactual Gate-2 passage.
- Actual Windows focused N189--N195 controller/verifier/runner tests pass
  `32/32`; the PowerShell runner parses. Policy/verifier/offline-report/runner
  SHA-256 values are
  `46a075eafd49a1e44a14e32af458a26e8d411e4cb507b510c6aaaa7ee6b97b4a`,
  `ef8a513c421870436f65d0c2e9f725cab3443610573b64a15c2470a68e1b45d8`,
  `d10fc3364865d2c66a45d12af028a71542a80b1f1daabe731d44fdce1d99d2b0`,
  and `4465f0e915a70510d63ec0c09f307b1cd0405b2f42431e811cca6b9b37259a30`.
- No simulator control command occurred during N195 development. Candidate 055
  and FullLap remain unauthorized. Next only: preserve PID `24476`, recover
  the sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` because Candidate 054
  stopped inactive, then send normal disarm followed by exactly one learned-
  target MAVLink `COMMAND_LONG 31000` with confirmation/params 1--7 zero.
  Require boot rollback over `1000 ms`, a fresh nonnegative race epoch, index
  `0`, finish `-1`, and visible Gate 1. Then run passive zero-command Shadow
  056 with `-Gate2FrozenObservationDropoutN194`; only clean hash/parity,
  cadence, telemetry, camera, and fault checks may authorize a separately
  preregistered bounded flight.


## N195 Live Reset Proof and Passive Shadow 056 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 054 had
  stopped inactive at base/status `65/3`; the sole VQ1/R1 UI was recovered
  using `-NoRelaunch -SkipLoginClick` on the same process. Command-free
  readiness then showed base/status `193/4`, index `0`, finish `-1`,
  visible Gate 1, and old boot/race epoch `954041/934020 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `954041 -> 3035 ms`; fresh race start was
  `3298 ms`. A later command-free proof showed boot `18571 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n195_gate2_frozen_observation_dropout_n194_shadow_056` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `614/10.0 = 61.3 Hz`, with zero violations. All `280/280` trace
  samples replayed at `27.955 Hz` with maximum action error
  `2.1868190765161888e-7`; every deployment hash matched. It observed `2427`
  telemetry messages and `125/125` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `8d8122f797cf4c42d411ee6d6d41d8ae82da0d3429430825f8d0a6b6f9fd948f`,
  `af73ed4f804e724737f6eadd6fde105f3fbd92293afb2cd2cc8527832b7f738e`,
  `4246c29885f38e863934e42e0e9326ed21d3ee024e75791cb4ce1fb7b3e1dda7`,
  `805626a5fc146b7060cd57713c6774be9589e4d4046dec228ce571bfcb5c379f`,
  and `94b19c2a1616347449a3e4dae898d12551a6ee626dacf1411ed38081add091a6`.
- Authorize exactly one Candidate 055 `BoundedGate3` under tag
  `n195_gate2_frozen_observation_dropout_n194_bounded_055`: frozen N195
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence
  `[50,100) Hz`, exact hashes, zero rate/collision/invalid/dropout/drain
  fault, one exit disarm, and a command-free post-stop proof. Reject failure
  without unchanged retry. FullLap remains unauthorized.


## N195 Candidate 055 — Clean Official Gates 1--3 Milestone

- Candidate 055 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `139093 -> 3336 ms`, fresh race
  start was `3284 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.537808418 s`; the
  configured index-3 stop triggered at controller elapsed `10.546 s` and
  finish remained `-1`. Camera-range pass events at `4.375/7.078/10.359 s`
  are diagnostic only. There was no in-run collision or invalid condition.
- Transport and perception were healthy: `616/10.421 = 59.015 Hz`, zero
  command-rate violations, `3267` telemetry messages, `131` frames and
  `131` detections, and zero telemetry dropout, drain hit, malformed message,
  decode failure, or missing quad. Offline replay of the exact attempt shows
  zero N195 Gate-2 dropout activations and zero N194 Gate-3 frozen-recovery
  activations; this successful path did not require either watchdog.
- The command-free post-stop proof sent no reset or control command. After exit
  disarm, v3385 had cleared the race epoch to start `-1`/index `0` and
  reported passive low-threat ground contact `1001`, impact `0.02107`; it
  is not substituted for the smoke report's zero in-run collision result.
  Smoke/attempt/summary/post-stop/Gate-2-diagnostic/Gate-3-diagnostic hashes are
  `b44b0b441e3382c4d2722a23f7ae00fae1a05d95e5816a8962595d9bdf212109`,
  `5d4514ace948067347852759dc7a4d92095c17131aadfc1894741a65d946f194`,
  `ce8408bba4da14fe76b04674de3a761da9d61efd77a552261093750550d33a67`,
  `4e8ff2468d7c96ab538497667e9e10e390fc260d88e8691a816d1da32d12935d`,
  `167b0a31efefde52826c4657627e02138f5962213f57e16333e875e1e470cb19`,
  and `d7d6abb6f99f883f0b79955c3c5eaf868073d6404dc498733d1915889239c4d5`.
- Candidate 055 is accepted only as the bounded Gates-1--3 milestone, not as a
  six-gate lap. Authorize exactly one Candidate 056 `BoundedGate4` under tag
  `n195_gate2_frozen_observation_dropout_n194_gate4_bounded_056`: no code,
  policy, checkpoint, or hash change from passed Shadow 056/Candidate 055;
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  FullLap remains unauthorized.


## N195 Candidate 056 — Gate-3 Collision Before Gate 4

- Candidate 056 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.484 s`: boot rolled `417871 -> 3390 ms`, fresh race
  start was `3307 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially. A third camera-range pass event at
  `10.109 s` was diagnostic only: official index remained `2`. Gate 3 then
  collided with frame `1001`, threat `2`, impact `9.248054504394531 m/s`;
  finish stayed `-1`, so Gate 4 was never active or evaluated. Candidate 055's
  clean official index-3 result remains the retained milestone.
- Transport was healthy: `601/10.250 = 58.537 Hz`, zero rate violations,
  `3212` telemetry messages, `128` frames/detections, and zero telemetry
  dropout, drain hit, malformed message, decode failure, or missing quad. N195
  Gate-2 dropout and N194 Gate-3 frozen recovery each had zero activations.
- The discriminating current branch is N192 terminal leveling. It was inactive
  throughout clean Candidate 055 but active for Candidate 056's last five
  Gate-3 samples. It latched at `9.781 s`, `5.367 m` forward, with plane-
  projected right/down `-0.342/-0.703 m`, then held normalized roll at
  `0.0` through collision while the lower promoted Gate-3 mode called for a
  negative counter-bank. Candidate 051's original three protected terminal-
  level samples begin much later at `3.664 m`; this separates an offline
  hypothesis that delays only the latch admission range without retiring it.
- The command-free post-stop proof sent no reset or control command. After exit
  disarm, v3385 cleared race start to `-1`/index `0` and reported passive
  low-threat ground contact `1001`, impact `0.02022`; this does not replace
  the in-run collision record. Smoke/attempt/summary/post-stop/control/
  transition/terminal-level/Gate-2/Gate-3-frozen hashes are
  `db797be5729a248b26841be1de81fd7ed34143a85ab369eb716cde5bbea0d9fc`,
  `970fbc2ee7ef90e6158a46e8c28c466c536379ab5c5280eb0eeff47f53d7505a`,
  `f9456a5425b19f9ed9ccd39929f7b8af0a03a714ccc88884d435d38fe6638ab0`,
  `bf1390342eb69e8a402d01797ec65969a47230d3a294ffcdfdf5dac6b92e0226`,
  `a53241039a24a18823d4a9507d02f8027fb9a81a2f9a767bde34051651a66174`,
  `adc0fe651c1d91824ba4afd4f87e263d8165cb8e650ed49b3237fa94ebe4511e`,
  `1b7250a551886dd50d35143a503e9b06016bdbf9181b0ac8ff29d7b20188fa28`,
  `42d7fe27c39b76ab7ec5eb2cc5208c39c6e20370490083fba376f910ea83fba0`,
  and `b553a445222dcdfb593278fb34736b0f237e2434b51de9b311191f66d10993e9`.
- Candidate 056 is rejected without unchanged retry. Candidate 057 and FullLap
  remain unauthorized pending an offline-separated later terminal-level
  admission, pinned hashes, focused tests, exact reset proof, and passive zero-
  command shadow.


## N196 Later Gate-3 Terminal-Level Admission — Offline Freeze

- N196 keeps the exact source-pinned N195 controller composition and changes
  only the Gate-3 terminal-safe-level admission range from `6.0 m` to
  `3.7 m`. The projection plane remains `2.0 m`; maximum absolute projected
  right/down errors remain `0.5/1.0 m`; minimum closing speed remains
  `1.0 m/s`; the branch still latches only on official Gate 3, changes only
  normalized roll to `0.0`, and releases on official transition. Gate-2 stale
  dropout, rapid vertical correction, Gate-3 frozen recovery, recurrent state,
  pitch, thrust, and yaw are unchanged.
- Source-hashed replay passes on protected clean Gate-3 paths 016/018/020/021/
  022 plus Candidates 045/048, with zero action changes. Candidate 051 retains
  its exact three proven level corrections beginning at `3.664 m`; clean
  Candidate 055 stays inactive. Candidate 056's old five-sample latch is reduced
  to two samples beginning at `2.447 m`, retiring exactly the premature
  `5.367/4.873/3.875 m` admissions. This is offline branch separation only,
  not counterfactual proof of a Gate-3 pass.
- Linux focused N189--N196 controller/verifier/runner tests pass `37/37`;
  actual Windows focused controller/runner tests pass `30/30`; Python
  compilation and the PowerShell deployment parser pass. Policy/verifier/
  offline-report/runner SHA-256 values are
  `3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df`,
  `f0978810ba51fcfdbd1a2f031049a73398648e2e5179027b8053499ee7e697e7`,
  `c53ddc068a47a1cc1f3b296ad4fd40c94916ab95c796a184cddc2713939bee5d`,
  and `82668d40b44f1991e199251f2b3109ebe37a6b38e076281ac4cc31f9b4043f79`.
- No simulator command occurred during N196 development. Candidate 057 and
  FullLap remain unauthorized. Next only: preserve responsive shipping PID
  `24476`; recover the sole VQ1/R1 UI on that same process with
  `start_windows_aigp_race.ps1 -NoRelaunch -SkipLoginClick` if the inactive
  base/status `65/3` state is present; then send normal disarm followed by
  exactly one learned-target MAVLink `COMMAND_LONG` command `31000`,
  confirmation `0`, parameters 1--7 all zero. Count reset only after boot
  rollback greater than `1000 ms`, a fresh nonnegative race-start epoch,
  official index `0`, finish `-1`, and visible Gate 1. Then run passive
  zero-command `n196_gate3_later_terminal_level_n195_shadow_057` with
  `-Gate3LaterTerminalLevelN195`; only a passing hash/parity/cadence/stream/
  fault report may authorize one separately preregistered bounded Candidate
  057.


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

## MAVLink documentation audit — 2026-07-18

- A site-wide relevance scan of the English MAVLink documentation tree is
  recorded in `docs/mavlink_site_context_2026-07-18.md`. It covers all
  documentation categories and deep-audits the subset applicable to v3385:
  framing/routing, heartbeat and IDs, commands, offboard attitude control,
  telemetry validity, packet loss, time synchronization, and signing.
- The deployed body-rate target mask `128`, learned target IDs (including valid
  component zero), one-shot command-`31000` confirmation `0`, seven zero
  parameters, independent setpoint publisher, and reset postcondition proof are
  protocol-consistent. A reset is still counted only after measured simulator
  boot rollback and a fresh race epoch; an ACK alone would not replace that
  proof, and blind reset retries remain forbidden.
- One real adapter defect was found: `TIMESYNC` request fields were reversed.
  It is corrected to `tc1=0, ts1=<local ns>` with a unit test. Current v3385
  runs expose no useful TIMESYNC stream, so this did not cause the failed laps
  and the recurrent policy must use a local monotonic fixed-rate clock.
- New high-value diagnostics are packet-sequence loss, fail-closed handling if
  `HIGHRES_IMU.fields_updated` stops marking gyro bits `8|16|32`, optional
  `COMMAND_ACK 31000` capture, and an opportunistic `ATTITUDE_TARGET` echo
  probe. None should delay N230 fixed-clock shadow/parity.

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

## VQ1 v3391 telemetry-enabled re-release supersedes N241 — 2026-07-26

- The user supplied a new authoritative simulator install at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391-VQ1\AIGP_VQ1_3391\FlightSim.exe`
  and its example SDK at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391-VQ1\PyAIPilotExample-v4`.
  The supplied official announcement says the legacy VQ1 simulator has been
  re-released with full telemetry interfaces, sensor outputs, downstream
  telemetry, and state estimation. VQ2 remains the no-telemetry environment.
- This changes the active dependency, not the objective: produce a competitive,
  autonomous, collision-free VQ1 lap through all six gates with official index
  `6` and nonnegative `race_finish_time_ns`. The v3385/N241 continuation is
  paused while v3391 is characterized.
- Static SDK inspection confirms reset command `31000` and documents revision-
  3390 type-mask bit `16` for correctly interpreted physical-rad/s body rates.
  However, the SDK receiver still contains stale-looking comments that
  attitude, local position, odometry, and gate fields are disabled/nulled.
  Resolve that contradiction by observing v3391 packets and non-null values.
- Required next sequence: migrate the simulator process to v3391; open VQ1;
  inventory message types/rates and track-transfer fields without flight
  commands; re-prove reset rollback/fresh index zero; then implement the
  simplest deterministic ordered-gate controller supported by those feeds.
  Prefer position/velocity trajectory following when the plant accepts it,
  otherwise use telemetry-closed cascaded attitude/rate control. Do not resume
  visual family-identification work unless live v3391 evidence requires it.

## N242--N248 measured v3391 telemetry controller — 2026-07-26

- Passive v3391 VQ1 measurement proves restored state telemetry: attitude
  approximately `102 Hz`, local NED position `83.5 Hz`, odometry `65.3 Hz`,
  HIGHRES_IMU `49.2 Hz`, actuator output `83.5 Hz`, and camera `30.2 Hz`.
- One exact no-flight reset proof rolled boot `334629 -> 374 ms`, provided a
  fresh index-zero/finish-minus-one epoch, and transferred all six ordered gate
  centers and `2.72 m` dimensions. Reset artifact SHA-256:
  `187895d524454ea2809218843670493bfda6eb945c07c7ec19a4311e7e92bc8a`.
- Local-NED velocity setpoints actuate. The admitted launch is N247: retain the
  spawn attitude for `0.5 s` with zero body rates/thrust `0.27`, then hand off
  to line-segment velocity plus bounded cross-track correction. Its two-second
  `1 m/s` probe reached `x=-1.757 m`, maximum cross-track error `0.0651 m`, and
  had no collision after the first-second/first-metre spawn-support envelope.
  All six low-impact threat-1 support contacts remain explicitly logged.
- Preregister N248 exactly once: `45 s`, `5 m/s`, cross-track gain `1 s^-1`,
  correction cap `2 m/s`, `60 Hz`, six gates, exact reset, N247 launch, and
  immediate abort on an out-of-envelope collision. Require official index six
  plus nonnegative finish time. Only after a valid finish may speed be raised.

### N248 rejection; N249 preregistration

- N248 followed the published Gate-1 center accurately but hit the physical
  gate frame at `4.37 m/s`: object `1001`, threat `2`, impact `4.4685`, position
  `(-22.927,-0.404,-0.030) m`, cross-track error `0.011 m`, official index `0`.
  This rejects unchanged high-speed crossing, not telemetry geometry.
- N249 retains `5 m/s` cruise and smoothly schedules speed to `1.5 m/s` over
  each final `5 m`, holding low speed through the gate until official retarget.
  Execute once for `45 s`, with every other N248 contract unchanged.

- N249 still hit Gate 1 at actual `2.65 m/s` despite a `1.60 m/s` command;
  impact fell to `2.7369`, and center error was `0.0131 m`. Reject the
  five-metre braking distance. N250 is one constant-`1 m/s`, 30-second Gate-1
  crossing test to separate clearance from deceleration lag.

## N250 result and N252 six-gate learned-policy promotion — 2026-07-26

- N250 also contacted Gate-1 object `1001` while centered and moving at only
  `1 m/s`: threat `1`, impact `0.065064`, official index `0`. Therefore speed
  alone does not explain the live gate failure; live geometry/collider or
  command-frame semantics must be reconciled before a full-lap attempt.
- The previously successful v3385 visual/attitude checkpoint remains the
  official three-gate baseline (SHA-256 `ea03965f...`). Its tensors are not a
  valid initialization for the new v3391 telemetry/local-velocity contract.
- A new PufferLib profile now models all six published gates and exposes only
  deployable v3391 local telemetry. Random-weight PPO failed, and a PPO
  checkpoint collected while the analytic teacher owned the plant also failed
  its teacher-off acceptance test at `0/6`; neither is promoted.
- Explicit full-network behavior cloning on `512` successful six-gate teacher
  episodes produced N252. With teacher fully disabled, deterministic exact
  evaluation passed `512/512` episodes at all six gates, `100%` validity,
  `23.065336 s` mean completion, and zero misses/timeouts. Retained checkpoint:
  `checkpoints/drone_race_vq1_v3391_telemetry/n252_bc_6gate.bin`, SHA-256
  `cec95192a04ebbe0c28f61226fcd76a6851404b6548be12ca6c44165397e9d76`.
  Evidence SHA-256 is
  `9d00c204ebc983974992e92495e9cb233aed1a31de6c3ecac8940abd2afe297a`.
- Acceptance remains two-stage: N252 is native-promoted but not official-live
  promoted. Required next action is a zero-command live telemetry/policy shadow,
  followed by a bounded Gate-1 validation only after resolving N250's collision.

## N253 shadow passed; N254 geometry audit; N255 bounded Gate-1 test — 2026-07-26

- N253 closes the command-free transfer gate: the N252 checkpoint ran `601`
  finite live inferences over `10 s` at `60.1 Hz` against a fresh v3391 origin,
  with exact checkpoint hash, no transport faults, and zero outbound MAVLink
  messages of every kind. Artifact SHA-256 is
  `75dbce6730b821f15c44dd2859ab379be35b88b858a48ab4b1fefae4f399f112`.
- N254's zero-message camera/telemetry alignment probe found the first aperture
  at `(326.0,176.5) px`, `20x21 px`, and estimated `23.4146 m` range while the
  published telemetry center was `23.3014 m` away. The visible opening is near
  the published center, so do not add a speculative half-gate-height offset.
  Artifact SHA-256 is
  `3e471fbf6981b122313a2428925105a3c8a8c65a947651194746b6d9b70c0d37`.
- N255 tests the remaining event-order hypothesis once. Repeat N250 at constant
  `1 m/s`, but continue through only a centered object-1001 threat-1 message
  within `0.5 m` of Gate 1 and stop/disarm immediately when the official index
  becomes `1`. Record the warning separately. Any threat-2/off-center/other
  object event still aborts. Success is bounded official Gate-1 progress, not a
  learned-policy promotion or a lap. No retry. Runner SHA-256 is
  `083192e1540946e27a4c7f0c6bfe316cfdf92911b1436b5fadb6ae08c13e062e`.

### N255 rejected; N256 queues a measured down-offset

- N255 did not advance Gate 1. It accumulated `88` low-threat gate collisions,
  stalled about `0.205 m` before the published center, and was forced from NED
  z about `-0.034 m` down to `+0.408 m` by the physical response before moving
  beyond the plane. Index remained `0`; reject the warning/event-order theory.
  Artifact SHA-256 is
  `0579c0792c54c25b590af5e6887d506d4551486200a982b97557d559d6508a1b`.
- N256 runs once at `1 m/s` through a target `+0.4 m` NED-down from the published
  Gate-1 z, holding published x/y. It has no collision exception and stops on
  index `1`. Any collision or failure to advance rejects the offset. This is a
  geometry probe, not learned-policy evidence. Runner SHA-256 is
  `4643df1590ece72075e2e87b5bddf9799c6c64cb4546723e7b1c95bca14fd56e`.

### N256 clean under-pass; N257 tests aperture center

- N256 crossed more than `5 m` beyond Gate 1 with target z `+0.4 m` NED-down,
  zero in-course collisions, and index still `0`. It cleanly went below the
  scoring aperture. Artifact SHA-256 is
  `e56f5ea238936a8f7eb5b652f957acf5ca8dbb0e64851330b4e8eaf8d9ef213e`.
- The paired result is now geometrically coherent: transmitted z is the bottom
  of the `2.72 m` gate; N255 hit that lower frame and was deflected down, while
  N256 went below it. Therefore the aperture-center target is transmitted z
  minus `1.36 m` NED-up. This supersedes N254's initial image-only center
  interpretation.
- N257 runs that centerline once at `1 m/s`, collision abort enabled, stopping
  at official index `1`. Any collision or no advance rejects the interpretation.
  Runner SHA-256 is
  `f37c8431e9924fd8855e48091d55a440693f9d0bc488d4d1edff49810396e20a`.

### N257 proves the aperture-center transform; N258 retrains by transfer

- N257 cleanly advanced official Gate 1 at transmitted z minus `1.36 m`:
  index `0 -> 1` at `24.469 s`, no in-course collision, final NED
  `(-23.464,-0.403,-1.397) m`, then immediate stop/disarm. Artifact SHA-256:
  `6676991f27ade521cc437d3a8957971f24b7022346bed753b3706a173d5d4c29`.
- Apply the proven half-height transform to all six transferred base positions.
  Corrected PufferLib config SHA-256 is
  `9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08`;
  focused environment/teacher/runner/shadow suite passes `15/15`.
- N258 transfers from N252 rather than random initialization. Regenerate `512`
  corrected full-course demonstrations, behavior-clone the whole two-layer
  network from N252, and require `512/512` deterministic teacher-off six-gate
  completion. Only that successor may proceed to a fresh command-free shadow.

### N258 rejected due legacy phase decoding; N259 fail-closed successor

- N258 demonstrations are valid: `512/512`, `718033` records, mean
  `23.373470 s`, maximum radial `0.050898 m`. But its BC invocation used the
  legacy three-phase decoder. Gate 2 decoded above the active loss range, so
  exactly `105472` records were silently excluded. The checkpoint passed one
  gate then timed out in exact teacher-off evaluation and is rejected.
- The trainer now detects six-gate phase columns and requires
  `--gate-progress-observation`; omission is a hard error. Focused tests pass
  `21/21`.
- Unmodified N252 also fails the corrected first aperture at radial `0.777577 m`.
  N259 therefore transfers from N252 using the same valid corrected dataset with
  the correct progress decoder and all `718033` records. Acceptance remains an
  exact deterministic `512/512` six-gate teacher-off screen.

### N259 native-promoted; N260 shadow-passed; N261 queued

- N259 used all `718033` corrected samples from N252 initialization. Its exact
  deterministic teacher-off screen passed `512/512`, all six gates, success and
  validity `1.0`, zero misses/timeouts, mean `23.088762 s`, and final radial
  `0.206347 m`. Checkpoint SHA-256 is
  `f13d4b8d571d70be16731068c88dd105e81fe7bd00a86eae6f13f81031f00fc0`.
- N260 command-free live shadow passed `601/601` finite inferences at `60.1 Hz`
  with no transport fault and zero outbound MAVLink messages. Artifact SHA-256:
  `2440e182a198b859e4cd29a08d3d0ca98ed02e542cf252c62801d923a31b824f`.
- Execute N261 once: N259 at `60 Hz`, `2 Hz` heartbeat, proven `0.5 s`/`0.27`
  launch, strict non-launch collision abort, and stop at official index `1`.
  Success is bounded learned-policy Gate-1 progress only. No retry. Runner SHA:
  `5e140d835719ff36d99d8b3fa85a2bd9c930568a894cfa5c2f17c202341b0481`.

### N261 learned Gate 1 passed; N262 full lap queued

- N261 advanced official Gate 1 in `4.922 s` with N259, no collision/fault,
  `60.138 Hz` control, and `6.13e-8 m` maximum gate-transform mismatch. Artifact
  SHA-256 is
  `e0215049227126f4ec60f907504ff18c19db1a8ef9d0d37747d7637df7dca3d8`.
- Execute N262 once with the identical frozen policy and controls for up to
  `45 s`. Require official index `6`, nonnegative finish time, and no in-course
  collision or health/rate fault. Stop/disarm at finish; do not retry.

### N262 result and N263 acceptance gate

- N262 passed the first two official gates at `4.891 s` and `9.157 s` without
  collision or invalidation, but missed Gate 3 off-center and timed out at
  official index `2`. The first Gate-3 plane crossing was `1.8731 m` from the
  corrected aperture center. Retained artifact SHA-256:
  `4100f48bdd658407788fee11173cc7e7bf13be68f4a93ea9b541d00fadbf3169`.
- Do not restart weights. N263 keeps the native-promoted/shadow-passed/live-
  two-gate N259 checkpoint as the learned speed policy and adds a public-
  telemetry gate-centering governor. Run once only to official index `3` with
  learned along speed capped at `8 m/s`, `4 m/s` crossing speed, final-`10 m`
  slowdown, and cross-track gain/cap `1.5 s^-1`/`4 m/s`. Collision, health
  fault, or an unchanged gate `1.5 m` beyond its plane rejects the successor.
  A clean Gate-3 pass is required before an identical six-gate run.

### N263 passed; N264 queued unchanged

- N263 passed Gates 1--3 at `5.156/9.406/14.421 s`, with null collision/fault
  and `0.06233 m` final sampled Gate-3 cross-track error. Artifact SHA-256:
  `a0bfdc2a84a697bc36904792bb01ea702aa7506fb51a300a0411bde63b3bc114`.
- Execute N264 once using the identical learned-policy governor contract. Only
  change the stop index from `3` to `6`; cap duration at `40 s`. Require index
  `6`, nonnegative finish time, and null collision/invalid reason. No tuning
  between N263 and N264.

### N264 milestone accepted: official six-gate lap

- N264 is the first accepted v3391 six-gate baseline: official finish
  `29.237468719 s`, official index `6`, null collision/invalid reason, zero
  telemetry dropout/drain-limit/malformed packets, and immediate stop/disarm.
  Control delivered `1767` commands at `60.024 Hz`. Artifact SHA-256:
  `3752249b0560a165f870fc5d222d111694f10b8290b2816de644c92ae3ce6d90`.
- Product status is no longer “obtain a valid lap.” Preserve N264 and optimize
  competitive time. Use offline sweeps first, number every successor, and keep
  the learned N259 checkpoint frozen unless new training evidence beats its
  exact `512/512`, `23.088762 s` native acceptance result. Live candidates must
  retain strict collision/reset/health and immediate stop/disarm contracts.

### N265 competitive-time analysis and N266 promotion gate

- N265 preserves the accepted N264 JSON and N259 checkpoint byte-for-byte. It
  reconstructs all six official segment times and measures a consistent live
  along-speed deficit: scheduled means `6.81--7.11 m/s` versus actual means
  `5.05--6.00 m/s`; the fitted primary x response is gain `0.8341`, time
  constant `0.8369 s`. No N264 sample reached the `4 m/s` cross-track cap.
- A source-hashed, deterministic `840`-candidate replay sweeps all requested
  governor dimensions and reports the raw `-0.3572 s` finish-model residual.
  Gate-wise correction anchors the counterfactual comparison to the official
  `29.237468719 s` baseline. Per-gate crossing schedules do not beat the best
  shared schedule in this grid.
- The minimal first-target recommendation changes only crossing speed from
  `4.0` to `7.5 m/s`. It predicts `26.792198 s` and maximum crossing error
  `0.30604 m`. Evidence SHA-256 is
  `8642d60f8192f494791b94bc85a37ec58dc457ee0c89bae536deeb554a2cfede`;
  analyzer SHA-256 is
  `aa9fc959406ac2cb4634e3b67940af88cf3410b2a575c8c22551a07cb177e81d`.
- Execute N266 exactly once as a `20 s`, official-index-3 bounded run. Keep all
  N264 lifecycle, rate, health, collision, gate-transform, plane-miss, stop,
  and disarm rules and keep N259 inference active. Use tag
  `n266_v3391_n259_cross7p5_gate3_bounded_001`; an unchanged retry is forbidden.
  A clean pass may promote the identical parameter-only controller to a new
  full-lap N-number; no N266 failure may be treated as a lap-time result.

### N266 bounded validation passed; N267 full lap

- N266 cleanly reached official index `3` at race time `13.016105651 s`, versus
  N264's `14.255555060 s`. Gates 1/2 occurred at `4.707434654/8.494734764 s`;
  final sampled cross-track errors were `0.04829/0.17932/0.23168 m`. Collision
  and invalid reason are null, transport is `60.062 Hz`, and reset/gate-transfer/
  telemetry/final-stop/disarm requirements all pass. Evidence SHA-256 is
  `f3dc321301e5a5cd4a770ae66f455642a4e2529fb2999e2773a7ad722a3e7bf8`.
- Execute N267 once with the identical N259-plus-governor controller for all six
  gates. Only stop index and duration become `6`/`40 s`; use tag
  `n267_v3391_n259_cross7p5_full_lap_001`. A valid faster finish is a candidate,
  not yet the repeatability deliverable: two more separately numbered identical
  valid laps are required before promotion.

### N267 beats N264; confirmation pending

- N267 is officially valid at `26.604381561 s`, `2.633087158 s`/`9.006%`
  faster than N264. It passed index `6` with finish timestamp equal to the last
  gate time, null collision/invalid reason, zero telemetry faults, `59.990 Hz`,
  and final stop/disarm. Evidence SHA-256:
  `212c5af1b10909b3a694c60203a3649de84ba116c3911fd7d587eff76a1c8993`.
- Run the identical N268 confirmation once under tag
  `n268_v3391_n259_cross7p5_full_lap_confirm_002`. No controller, parameter,
  lifecycle, or acceptance change is allowed. This is confirmation 2 of the
  required 3 valid laps, not a new optimization.

### N268 passes second optimized lap

- N268 is valid at `26.591306686 s`, with null collision/invalid reason, zero
  telemetry faults, `60.032 Hz`, and final stop/disarm. Evidence SHA-256 is
  `51d36c1154133606b8bdefdf95818a4972c53c17efc8922a87060a68e8b950db`.
- N267/N268 spread is `0.013075 s`. Execute final identical N269 once under tag
  `n269_v3391_n259_cross7p5_full_lap_confirm_003`; no change other than evidence
  path is permitted. A pass completes the three-valid-lap product requirement.

### N269 passes; N270 promotion complete

- N269 is valid at `27.412017822 s`, still `1.825450897 s` faster than N264,
  with null collision/invalid reason, zero telemetry faults, `59.833 Hz`, and
  final stop/disarm. Evidence SHA-256:
  `9be363aa9c96e1623fceea287314f1241ee34269facf9ba8548b3076f319cf73`.
- The promoted controller is N259 inference plus crossing speed `7.5 m/s`; all
  other N264 governor settings and guards are unchanged. Three official laps
  pass (`3/3`): best `26.591306686 s`, median `26.604381561 s`, mean
  `26.869235356 s`, worst `27.412017822 s`, spread `0.820711136 s`. Every result
  beats N264 and the best is `9.051%` faster.
- N270 aggregate artifact SHA-256 is
  `ff466a254fc829dd764484783b20bcf1fdca0c42f96fbba6045f937da2e0a247`.
  The aggregate source/hash/acceptance audit and focused `7/7` tests pass. The
  first sub-`27 s` target and repeatability deliverable are complete; near-`24 s`
  remains an optional stretch rather than evidence already achieved.

### N271 robust near-24 analysis; N272 bounded validation

- N271 uses all three promoted N267--N269 traces as independent live-response
  models and source-locks their evidence plus N270, N259, the unchanged runner,
  and corrected config. Every counterfactual is gate-wise anchored by the full
  official-minus-replay residual for its own trace; no modeled time is an
  official result.
- A deterministic `30`-candidate sweep finds the minimum-change setting meeting
  the robust `24.25 s` target and retained `0.45 m` crossing-error limit in all
  models: crossing speed `9.0 m/s`, maximum along speed `9.0 m/s`, and
  cross-track gain `2.0 s^-1`. Slowdown distance `10 m`, correction cap
  `4 m/s`, N259, and all reset/lifecycle/safety guards remain exact.
- Predicted N267/N268/N269 finishes are `23.273484/23.243388/24.080792 s`;
  worst modeled crossing error is `0.432355 m`. Evidence SHA-256 is
  `a5d3e09625ae8883f19b111d61c8c691310d361fffba593d6db6ee47ccbe1278`;
  analyzer SHA-256 is
  `687e823097a56415e61b3258d0283a8bd5d7ebd47298e550f3939cd522ea5a9b`;
  focused tests pass `10/10`.
- Execute N272 exactly once under tag
  `n272_v3391_n259_cross9_xgain2_gate3_bounded_001`, for at most `20 s`, at
  `60 Hz` commands and `2 Hz` heartbeat, stopping at official index `3`.
  Require the complete N270 reset/gate-transfer/health/rate/final-stop/disarm
  contract. Failure rejects the candidate without unchanged retry; a clean
  bounded pass authorizes only a separately numbered full-lap attempt.

### N272 bounded validation passed; N273 full lap

- N272 passed official Gates 1--3 at `4.435974121/7.801361083/11.806352615 s`,
  improving the accepted N266 Gate-3 prefix by `1.209753036 s`. Final sampled
  cross-track errors were `0.048645/0.195365/0.240791 m`; correction remained
  below `1.422 m/s` versus the retained `4 m/s` cap.
- The bounded result is accepted at official index `3`, finish `-1`, null
  collision/invalid reason, `59.964727 Hz`, zero telemetry faults, one detected
  reset, six transferred gates, and final stop/disarm. Evidence SHA-256 is
  `8563b545191a03b5354e5d334c3ecc647627a834a2b5c3fe70ff39f84f7637a5`.
- Execute N273 once under `n273_v3391_n259_cross9_xgain2_full_lap_001`, changing
  only N272's duration/stop target to `40 s`/official index `6` and its evidence
  path. Require a valid nonnegative official finish and every existing safety,
  health, rate, stop, and disarm proof. A failure forbids an unchanged retry; a
  pass is a candidate near-24 lap requiring separately numbered confirmations.

### N273 rejected at Gate 4; N274 fail-closed status acknowledgment

- N273 retained the fast Gates-1-through-3 prefix at
  `4.460481643/7.811909675/11.828386306 s`, but the legacy `1.5 m` plane-miss
  guard fired at Gate 4 before official index `3` advanced. The run is rejected
  without unchanged retry; collision is null, telemetry is clean, rate is
  `59.983943 Hz`, and final stop/disarm passed. Evidence SHA-256:
  `6852268b9804d0744658e13640e5fc80ca7d38d8c7e884173158d3d5caa579e1`.
- N274 reconstructs a Gate-4 aperture-plane error of only `0.267310 m` at
  `7.314596 m/s`. Race status measured `5.103796 Hz`, while the distance guard
  allowed `205.069 ms`; the latest official status timestamp remained
  `21.473 ms` before the crossing. This proves the runner had no post-crossing
  authoritative sample before declaring the miss.
- The correction keeps `1.5 m` exact and adds an opt-in `350 ms` acknowledgment
  deadline. A post-crossing same-gate status or deadline expiry still aborts;
  zero grace preserves legacy behavior. Corrected runner SHA-256 is
  `643cdec893c511d59a8a5a9b54bde2bbab1d543a46b7d977ad92867a01e09505`;
  diagnosis SHA-256 is
  `08a16334973d13ca10caa84732d3b60937d4bdf300cf69804fe6c5dfb4d01678`;
  focused tests pass `16/16` and Windows checks pass.
- Execute N275 exactly once under
  `n275_v3391_n259_cross9_xgain2_ack350_gate4_bounded_001`: N273 control exact,
  `0.35 s` acknowledgment grace, `25 s` bound, and official stop index `4`.
  Require all existing acceptance guards. Failure forbids an unchanged retry;
  a clean index-4 result authorizes only a separately numbered full lap.

### N275 bounded Gate-4 pass; N276 full-lap preregistration

- N275 passed official index `4` once at Gate times
  `4.444323539/7.826340675/11.826963424/17.182216644 s`. Finish remains `-1`;
  collision/invalid reason are null; `1049` commands delivered `60.049230 Hz`.
  Reset, six-gate transfer, telemetry health, final zero setpoint, and disarm
  all pass.
- The fail-closed acknowledgment was active and material: `4` plane-crossing
  deferrals, maximum race-status lag `245 ms`, and no abort diagnostic. N275
  evidence SHA-256 is
  `89fab65cee876300ea0a86614786bf4a540f11c5970d492050799592f44edde1`.
- Execute N276 once under
  `n276_v3391_n259_cross9_xgain2_ack350_full_lap_001`, changing only duration
  to `40 s` and authoritative stop index to `6` from the accepted N275
  contract. Require official index `6`, nonnegative finish, null collision and
  invalid reason, zero telemetry faults, `[50,100) Hz`, exact frozen artifacts,
  final stop, and disarm. Failure rejects N276 without unchanged retry; success
  requires separately numbered repeatability confirmation before promotion.

### N276 target lap passed; N277 repeatability confirmation

- N276 passed all six official gates in `23.848192214 s`, with gate times
  `4.422417163/7.812987327/11.818965911/17.180858612/20.544542312/
  23.848192214 s`. This improves N270's best by `2.743114472 s`.
- Index `6`, nonnegative finish, null collision/invalid reason, `59.992485 Hz`,
  zero telemetry faults, exact reset/transfer, and final stop/disarm all pass.
  Evidence SHA-256 is
  `6618847fcd8cbe959be045f89a43b623314cff22f9564984ffc614e626deef8a`.
- Execute N277 exactly once under
  `n277_v3391_n259_cross9_xgain2_ack350_full_lap_confirm_002`, keeping N276's
  controller and complete execution contract exact. In addition to all safety
  and validity requirements, require an official finish `<=24.5 s`. A failed
  N277 is not retried unchanged; a pass authorizes one final independently
  tagged confirmation before promotion.

### N277 equal-timestamp rejection; N278/N279 correction path

- N277 is rejected without unchanged retry. It passed Gates 1--3 at
  `4.432579994/7.811755180/11.823871612 s`, then the plane-miss guard stopped at
  index `3`; collision is null, rate is `59.981632 Hz`, telemetry is clean, and
  final stop/disarm passed. Evidence SHA-256 is
  `9e5e35f5142036c3aef1b58f208c36084650f75d11538e30e8e8d5857d52e690`.
- N278 proves the recorded crossing and status timestamps were equal at
  `20457 ms`; equality cannot establish a post-crossing sample. The abort was
  only `211 ms` after crossing, inside the `350 ms` grace, and the interpolated
  vehicle center was inside Gate 4's aperture.
- Change only `race_status_boot_time_ms >= crossing_boot_time_ms` to strict
  `>`; retain the distance/timeout guards. Corrected runner SHA-256 is
  `de4995a19c5f3c82f0de540b4b07ce810ffae49ad9fd4308befbcea3cfcb34ea`;
  report SHA-256 is
  `232c9d190def9ff9216229cd5c448816d34aebd230fe223875e8bd38c39edcce`.
  Linux passes `19/19`; Windows compile/direct and byte-parity checks pass.
- Execute N279 once under
  `n279_v3391_n259_cross9_xgain2_ack350_strict_gate4_bounded_001`, using the
  exact N276 controller/config and corrected runner, `25 s`, `60/2 Hz`, and
  stop index `4`. Require the complete bounded validity and safety contract.
  Failure forbids an unchanged retry; success authorizes only a newly numbered
  full-lap attempt.

### N279 bounded pass; N280 competitive full-lap attempt

- N279 passed official index `4` once at
  `4.423978328/7.786879062/11.807009696/17.171604156 s`. Finish remains `-1`;
  collision/invalid reason are null; rate is `60.036769 Hz`; reset, transfer,
  telemetry health, final zero setpoint, and disarm pass. Evidence SHA-256 is
  `e068a68501ee5fc9a73f9a86b43a222b8c3f3b14e30ae2b910976bb62256ea47`.
- Execute N280 exactly once under
  `n280_v3391_n259_cross9_xgain2_ack350_strict_full_lap_001`, changing only
  duration to `40 s` and stop index to `6` from N279. Require the full official
  validity/safety contract and finish `<=24.5 s`. Failure forbids an unchanged
  retry; success begins a separately tagged repeatability gate.

### N280 target lap passed; N281 confirmation

- N280 passed all six official gates in `23.832376480 s`, with gate times
  `4.415704250/7.803259372/11.801978111/17.162479400/20.518789291/
  23.832376480 s`. This is `2.758930206 s` faster than N270's best.
- Full validity, `60.041667 Hz`, zero telemetry faults, exact artifacts, final
  zero setpoint, and disarm pass. Evidence SHA-256 is
  `5b926383063ffe8cf2f881a634925853a73c5fbb81cce4c93418e6c95161abec`.
- Execute N281 once under
  `n281_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_002`, keeping
  N280 exact and requiring the complete contract plus finish `<=24.5 s`.
  Failure forbids an unchanged retry; success authorizes one final confirmation
  for a corrected-runner `3/3` promotion decision.

### N281 passed; N282 final repeatability confirmation

- N281 passed all six official gates in `23.879062652 s` with the full clean
  contract. Evidence SHA-256 is
  `0fd5a7d4b78ebdcc11562443c11a70b6593769bd15ecc5d9d9b4bc9a29225250`.
- Execute N282 once under
  `n282_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_003`, keeping
  N280/N281 exact and requiring every validity/safety invariant plus finish
  `<=24.5 s`. Failure forbids an unchanged retry; success completes a `3/3`
  corrected-runner promotion gate.

### N282 passed; N283 promoted near-24 controller

- N282 passed all six official gates in `23.867099761 s`, with null
  collision/invalid reason, `59.994159 Hz`, zero telemetry faults, exact reset/
  transfer, final zero setpoint, and disarm. Evidence SHA-256 is
  `27fbf3ba00c62553468f95d515a1144b0aeb0a31555cc5177fcce6c7e9e3fb36`.
- Promote N283 from N280/N281/N282: `3/3` official finishes, best/mean/worst
  `23.832376480/23.859512964/23.879062652 s`, range `0.046686172 s`, all below
  `24.5 s` and all faster than N270's best. N283 supersedes N270.
- Freeze N259, strict-ack runner SHA `de4995a1...`, config `9eec139d...`, and
  governor `9/9/10/2/4`, miss distance `1.5 m`, grace `0.35 s`. Promotion
  aggregate SHA-256 is
  `898c667184d9d69ea9367db356d963467c96546c8b3911b071929d5183c30818`;
  it regenerates exactly and focused tests pass `20/20`. No additional live
  confirmation is required or authorized.

### VQ2 package migration and command-free interface inventory

- The user supplied the VQ2-capable v1.0.3391 package at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`.
  Launcher/shipping/PAK SHA-256 values are `0d3217fa...`/`68dfd80d...`/
  `5d424b4e...`. The UI exposes VQ1, VQ2 Submission, and VQ2 Training; only
  Training was selected.
- The command-free raw probe saw `3215` MAVLink-v2 datagrams and `39822` valid
  camera chunks in `20 s`, with no v1 packet or camera-header failure. Artifact
  SHA-256 is `73cb40ea...`.
- The decoded `12 s` probe confirms VQ2 provides camera, finite HIGHRES_IMU
  accel/gyro, heartbeat, actuator outputs, and official race status. It provides
  no ATTITUDE, LOCAL_POSITION_NED, ODOMETRY, or track transfer. Message rates
  are heartbeat `9.917`, IMU `54.583`, actuator `93.917`, and race status
  `4.000 Hz`; telemetry faults are zero.
- Camera is `640x360` at approximately `30` unique FPS, with duplicate complete
  frames requiring `(frame_id, sim_time_ns)` deduplication. Inventory/frame
  SHA-256 values are `7e68308c...`/`8a0e9ea3...`; probe SHA-256 is `fe6943a4...`.
- N283 remains frozen VQ1 evidence and cannot deploy in VQ2. Before any VQ2
  control command, measure camera extrinsics and validate a pose-free camera/
  IMU/previous-action/raw-progress observation plus passive perception/state
  reconstruction in Training. VQ2 Submission remains untouched.

### VQ2 Gate-2 PufferLib requirement and N296--N313 outcome

- Runtime control is PufferLib-policy-only. Deterministic/native controllers
  may generate offline teacher labels or rewards but cannot arbitrate commands
  in evaluation, shadow, or FlightSim.
- N295 proves official Gate 1 and reconstructs a severe Gate-2 transition:
  about `14.74 m` forward, `8.70 m` right, `1.095 m` above, entered near
  `4.677 m/s`. The corrected training-only teacher solves `128/128` exact, but
  N296--N310 remain zero-success when assistance is disabled.
- N311 step `188416` is curriculum-only: `128/128` at blend `0.50`, `119/128`
  at `0.25`, and an unassisted crash. N312 is `0/33` unassisted checkpoints;
  N313 does not improve the lower rung. None is deployable or authorized.
- Keep the simulator frozen. The next promotion gate is a recurrent Puffer
  checkpoint that completes Gate 2 at teacher blend `0` on exact and perturbed
  native screens, followed by replay, Windows zero-command shadow, source lock,
  and a newly preregistered bounded VQ2 Training attempt.

### VQ2 Gate-2 Puffer-only continuation through N336

- A phase-switched deployment is allowed only when every phase is a recurrent
  PufferLib checkpoint consuming the legal 32-value observation. N294 may
  remain the frozen Gate-1 policy while a second Puffer policy is warmed from
  race start and selected at official index `1`. No classical command blend or
  state-feedback fallback is allowed at runtime.
- The N317 prefixed evaluator replays all `195` official N295 public
  observations/actions into candidate recurrent state, restores the final
  previous action at Gate-2 entry, and requires teacher blend `0` for admission.
  Training-only assisted collection is explicitly labeled non-admissible.
- N314--N327 do not yield a reliable unassisted policy. N323 SHA `dff5bc7c...`
  is a safe `0/128` local parent. N328 SHA `1fe9c426...` proves the first
  Puffer-only Gate-2 completions (`7/128`) but crashes `89/128`; its wider
  radius-`2.0 m` parent SHA `532f0dfa...` completes `32/128` and crashes
  `71/128`. Neither is eligible for shadow or FlightSim.
- N329--N335 do not improve the frontier. N336's corrected episodic PPO child
  also loses the fair same-seed comparison: `31/128` success and `72/128`
  crash, versus its unchanged parent's `32/128` and `71/128`. Reject N336.
- Continue command-free/offline PufferLib optimization. The live gate remains
  reliable zero-assistance exact and perturbed completion, source lock,
  composite replay, and Windows zero-command shadow. No command has been sent
  since N295, and VQ2 Submission remains unauthorized.

### VQ2 Gate-2 corrected evaluator and retained N380 Puffer frontier

- N337--N342 policy schedules, decoder surfaces, action-row PPO, and ES do not
  promote. The audit that follows discovers legacy custom-start reset noise in
  the nominal “exact” screen. N343+ explicitly disables that noise only for the
  measured-transition contract; default native behavior remains unchanged.
- Corrected exact evaluation first passes with the N328 pitch/thrust child.
  N348 centers its crossing, and N353 proves that tiny legal start variation
  bifurcates the recurrent trajectory after camera detection dropout. Phase
  reset and the legal motion-predictor branches fail.
- Default-off training-only camera-alignment reward then moves deterministic
  Puffer success from `55/128` through N363 `120/128`, N365 `125/128`, and
  N366 `128/128`, always at teacher blend zero in admission. N366 SHA
  `a53d767e...` is `509/512`, zero crash, at start scale `0.001`; separate
  gate, plant, and combined probes are `128/128`.
- A gradual `0.00125`/`0.0015` curriculum, lower exploration, and bounded full
  secondary-policy training retain N380 checkpoint SHA
  `6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a`.
  It passes the exact transition at `0.264729 m`. N386 measures `509/512` at
  `0.00125` and `498/512` at `0.0015`; the failures are rare missed approaches
  followed by low crashes several seconds after the normal pass. Per-dimension
  N381--N383 screens identify an attitude/body-rate interaction. N384--N387 do
  not improve the held-out result and are rejected.
- N388's two-policy callable advances N294 and N380 on every public
  observation, selects one complete action vector by quantized official
  progress, and contains no classical action path. Callable SHA is
  `1998a337...`; replay preserves all `195` N295 Gate-1 actions within
  `4.806e-7` and matches the standalone N380 Gate-2 action exactly. Actual
  Windows controller Python passes the three composite tests and produces the
  same parity result without a PyTorch dependency.
- N389/N390 prove that `0.10` and `0.01` classical blends each destroy the
  trajectory (`0/256`); their label datasets are rejected. N391--N396 learned
  checkpoint lines, per-row directions, Puffer schedules, and legal predictor
  settings do not improve N380.
- N397 shadow is a state-precondition failure from N295's inactive ground
  state, with no camera/inference but zero lifecycle/control commands. UI
  recovery preserves the process and explicitly selects VQ2 Training, after
  which receive-only status is reset-ready at index `0`.
- N398 active-Training shadow passes `527/527` replay within `4.052e-7`, 52.7
  Hz inference, 64.002 Hz no-op cadence, all nine hashes, healthy streams, and
  zero reset/arm/setpoint/disarm. Evidence hashes are `13120ab1...` and
  `66b059e4...`.
- Exactly one N399 bounded Training attempt is preregistered: N294 then N380,
  detected reset, 80 Hz request, 14-second maximum, target/min/stop index `2`,
  immediate safety abort, and one exit disarm. Wrapper SHA is `db2e603f...`;
  any failure rejects N399 without unchanged retry. Submission is untouched.
- N399 ran once and is rejected without unchanged retry. It passed official
  Gate 1 at `3.194960355 s`, then collided with ID `1002` at index `1` around
  `5.845 s`; finish remained `-1`. Its `372` Puffer commands arrived at
  `63.999 Hz` with zero rate violation, so transport is not the failure.
  Report SHA is `28a63a4...`. A separate command-free passive inventory proves
  `base_mode=65`, system status `3`, and zero outgoing packets/commands; SHA is
  `c84347fb...`.
- N400 confirms the previously optimistic exact native result (`1/1`,
  `0.264745 m` radial). N401 replays every one of N399's `340` whole-Puffer
  actions within `6.557e-7` and identifies the first contract divergence: a
  `7.278 m` camera-range jump in `31 ms` with only `0.00174 rad` bearing change.
  It also records the N295-12-second/native-20-second versus N399-14-second
  elapsed-normalization mismatch. Analysis/report SHA values are
  `a78c1e73...`/`59813343...`.
- No further live run is authorized. Continue offline with an N399-prefix,
  deployment-clock-matched recurrent Puffer policy and camera-range/association
  perturbations. Runtime actions must remain an unmodified whole Puffer output;
  native/classical state is training-only. VQ2 Submission remains untouched.


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

### VQ2 N483 rejection and corrected-bearing Puffer successor — N483--N520

- Runtime is Puffer-only by explicit user requirement. Every FlightSim action
  must be one complete recurrent Puffer checkpoint output. Deterministic legal
  preprocessing remains allowed; classical control is offline teacher/reward/
  diagnosis infrastructure only and cannot contribute a deployed action.
- N483 is rejected without unchanged retry. It passed Gate 1 at
  `3.251771926 s`, collided with ID `1002` at index `1` around `5.735 s`, and
  finished `-1`. Its `363` commands delivered at `63.822 Hz` with zero rate
  violations; reset/calibration and one exit disarm passed. Attempt/passive
  hashes are `8f167767...`/`9377a16f...`, and the passive proof is command-free
  at base mode `65`, status `3`.
- N519 replays all `333` recorded actions within `6.705523e-7`. The first
  unambiguous new Gate-2 aperture is sample `241` at `4.266 s`: yaw error moves
  from stale `0.49832` to centered `0.035891`. A public-bearing correction
  yields initial world vector
  `[14.518822083,0.418616153,0.472708623] m`; the registered training vector
  `[14.51880584,0.41894794,0.47249277] m` differs by at most `0.000332 m`.
  Source-lock SHA-256 is `ab6fcb57...`; focused coverage passes `42/42`.
- N495's corrected-geometry corpus is `320/320`, but its first exact decoder
  fails robustness (`58/128` at scale `0.05`). N505 expands training to a
  `512/512` scale-`0.10` corpus (`346,311` records, dataset SHA
  `1c27b0c9...`). Retain N506 checkpoint
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n506_corrected_geometry_stress_decoder/stress_only_ridge_1e-7.bin`,
  SHA-256
  `55b15e7347eb13769a55f2719509e34424914f9a4e69eb0d4c6eb588f975601f`.
  Teacher-free results are `128/128` exact, `125/128` scale-`0.05`, and
  `104/128` scale-`0.10`, with zero crash/timeout.
- N509--N518 recurrent, DAgger, assisted-distillation, and encoder refinements
  do not improve N506 and are rejected. N520 Linux/Windows composite replay
  passes: N294 prefix error is `4.805625e-7`, Gate-2 selection error is zero,
  callable SHA is `f54c885e...`, and report hashes are
  `19b96abd...`/`166b0685...`.
- No post-N483 simulator command has been sent. Before another bounded VQ2
  Training attempt, build and pass a source-hashed zero-command Windows shadow
  for N506 with all lifecycle/control counts zero. Submission is forbidden.

### VQ2 N521 shadow and N522 Gate-2 preregistration — 2026-07-27

- Recovery preserved the responsive v3391 process and visibly selected only
  the third VQ2 Training event. Receive-only status proved official index `0`,
  finish `-1`, base mode `193`, and system status `4`; readiness/screenshot
  SHA-256 values are `60e02e79...`/`9cbdd6fe...`.
- N521 `vq2_n521_corrected_geometry_shadow_005` passes without flight control:
  `529/529` Puffer actions replay at `52.821 Hz`, maximum error
  `4.392170e-7`, all nine deployment hashes match, and live inputs include
  `1,805` telemetry messages plus `114/114` frames/detections. No-op cadence is
  `64.004 Hz` with zero violations/MAVLink setpoints. Reset, arm, control, and
  disarm counts are zero. Report/parity hashes are
  `92053ff9...`/`88f34aa3...`.
- Exactly one bounded VQ2 Training attempt is preregistered as
  `vq2_n522_corrected_geometry_gate2_bounded_004`. Wrapper
  `scripts/run_windows_vq2_gate2_policy_n522_bounded.ps1`, SHA-256
  `274abd7fdd8010c422e27afcec3a3ef795c5758f501c644af6c416104d1d3cb8`,
  pins the N294/N506 whole-Puffer composite and all source-lock/shadow/runtime
  evidence. Use one detected command-`31000` reset, requested `80 Hz`, a
  `14 s` bound, and target/min/stop index `2`. Require finish `-1`, clean
  safety/streams/rate, one exit disarm, and passive disarm proof. Reject any
  failure without unchanged retry. Never select Submission.

### Emergent-policy redirection after N522 — 2026-07-27

- N522 is rejected without unchanged retry: official Gate 1 passed, Gate 2 did
  not, and collision ID `1002` stopped the run at index `1`. Report/passive
  hashes are `bbb5dd70...`/`233e226d...`; delivery was clean at `64.0 Hz` and
  passive disarm proof is mode/status `65/3` with every outgoing count zero.
- By explicit user direction, close the explicit detector/pose/vector/
  predictor and classical-teacher control lineage. The required successor is
  an emergent recurrent visual Puffer actor. Runtime inputs are pixel-level
  camera representations, body rates, reliable actuator feedback, previous
  actions, and learned recurrent state only. Official gate progress is judge
  evidence, not an actor input. Privileged state is confined to training-time
  reward, critic, auxiliary reconstruction, and world-model targets.
- No external controller, channel blend, override, state estimator, planner,
  or MPC may contribute a flown action. Use direct neural actuator commands
  only if the installed runtime's legality and actuation semantics are proved;
  otherwise use one complete neural body-rate/thrust output through the public
  interface. The detailed actor-legality contract is specified in
  `docs/vq2_emergent_puffer_goal_prompt.md`; the authoritative paper-guided
  execution prompt is `docs/vq2_paper_guided_puffer_goal_prompt.md`.
- No FlightSim actuation is authorized until a new pixel-input Puffer policy
  passes source-locked native/held-out screens, deployment parity, actual-
  Windows cadence, and a zero-control Training shadow. Submission is forbidden.

### N523 Informed-Dreamer implementation foundation — 2026-07-27

- Replace the earlier optional-Dreamer wording: the required optimizer is now
  Informed Dreamer with a Puffer-native rollout environment. No PPO/classical
  teacher may bootstrap the flown actor.
- Implemented a dedicated `drone_race_vision` ABI with `4,118` legal values and
  `34` decoder-only privileged values. Intervention tests prove that changing
  privileged dynamics targets leaves the legal slice and action exact.
- Implemented a continuous `64x64` projected inner-edge mask using the useful
  Geles et al. abstraction, three-action memory, asynchronous `30/64 Hz`
  timing, camera/edge/dropout/corruption randomization, and no explicit
  perception reward.
- Implemented the compact PyTorch RSSM, privileged/reward/continue heads,
  latent Gaussian actor/critic, deterministic deployment step, quantized
  replay, and native Puffer trainer. CPU/CUDA integration smokes complete with
  finite updates; course discovery and performance remain unproven.
- Detailed contract and verification evidence are in
  `docs/vq2_informed_dreamer_implementation_2026-07-27.md`. Live actuation
  stays frozen through held-out screening, Windows export/cadence, and a
  zero-command VQ2 Training shadow. Submission remains forbidden.

### Paper-guided Puffer goal and hardware adaptation — 2026-07-27

- `docs/vq2_paper_guided_puffer_goal_prompt.md` is the authoritative execution
  prompt for continuously re-auditing SkyDreamer and applying supported
  mechanisms to the VQ2 solve.
- The minimal target remains one learned `64x64` visual front end, one compact
  recurrent Informed-Dreamer Puffer policy, and one complete neural CTBR action
  vector. The paper's mapped flight plan and decoded-state runtime phase logic
  remain excluded.
- Training must separate burn-in, world-model sequence, and imagination
  horizons and scale through resumable local-hardware stages. Use mixed
  precision, microbatching, gradient accumulation, and bounded WSL-backed
  replay for the RTX 3070 8 GiB / 7.7 GiB RAM host rather than copying the
  paper's A100 memory schedule.
- N528's evaluation precedes the paper-aligned temporal/replay redesign and is
  closed by the N528-N531 frontier section below. Live VQ2 actuation and
  Submission remain frozen.

### N528-N531 paper-guided offline frontier — 2026-07-27

- N528 is closed and rejected: `0/64` deterministic Gate-1 success and
  `64/64` low crashes. Initial-mask blanking nevertheless changes every first
  action, proving a live but incompetent visual pathway.
- The trainer now separates replay context/world sequence/imagination as
  `16/64/16`, supports strict resumable WSL-backed NumPy memmaps, reports
  replay density and CUDA peaks, and fails closed on observation/action schema
  mismatch or non-finite action.
- Current deployment schemas are `vq2_visual_ctbr_v2` and
  `normalized_attitude_ctbr_v1`; the single recurrent actor supplies every
  pitch/roll/thrust/yaw target and the fixed decoder maps the full vector to
  the supported `SET_ATTITUDE_TARGET` contract.
- N530 passes the CTBR/memmap integration smoke with exactly two finite world/
  critic and zero actor updates. N531 measures a `0.7361111045` local-start
  fraction over 72 completed episodes for the paper-adapted 70/30 gate-local/
  full-start curriculum, with gate phase excluded from the legal actor slice.
- Next implement and benchmark AMP plus microbatch accumulation at world
  sequences 64 then 128 under a 7.2 GiB VRAM ceiling. Only then preregister one
  resumable `0.5M` offline discovery lineage. Live VQ2 commands and Submission
  remain forbidden.

### N592-N612 restricted actor/reward continuation — 2026-07-28

- The active deployed contract remains one recurrent 4,118-input visual Puffer
  actor emitting one complete four-channel CTBR vector. Output-row freezing is
  training-only; no runtime override, detector, phase input, or teacher exists.
- N592 rejects a non-semantics-preserving actor-distribution reinterpretation.
  N593-N595 isolate pitch/collective learning and improve offset-700 vertical
  miss to `2.17037 m`, but still score `0/64` with zero crash/timeout.
- N604 is the retained mixed current-policy world parent (`44ed81a3...`). Its
  learned reward head has weak correct collective ordering. One update is
  directionally correct; longer fixed-update branches reverse.
- N611 recovers N610's interrupted eight-update calculation under a new tag
  without modifying N610. Eight fails direction; four is selected by the
  original bracket rule. N612 then scores `0/64`, zero crash/timeout, improving
  vertical error by only `0.02110 m` to `2.14927 m`. Reject and close the branch.
- Next quantify collective gradient sign/SNR under one/four imagination samples
  and legacy/paper percentile normalization without any optimizer or native
  environment mutation. No FlightSim action is authorized; Submission remains
  forbidden.

### N613-N623 reward/event-support continuation — 2026-07-28

- Normalization and four-sample accumulation do not stabilize N604 actor
  gradients (`1/8` negative in every N613 condition). N615 finds a correctly
  signed but effectively zero `7.18e-9` local collective reward advantage and
  only eight current-policy gate events.
- N617 supplies a legal 128-event, all-agent, all-phase native corpus using the
  frozen recurrent Puffer actor only. This is training evidence, not a teacher
  or deployed phase signal.
- Whole-world event balancing is rejected through N623: it can improve decoded
  ordered phase but does not learn event reward/crossing, degrades dense state,
  reverses local collective credit, and leaves gradients unstable.
- Next implement phase-only event transition training with reward/actor frozen,
  followed only after admission by a separate frozen-latent reward calibration.
  Require dense preservation and a repeated stable negative gradient audit
  before another actor child. FlightSim and Submission remain frozen.

### N624-N632 transition/representation continuation — 2026-07-28

- Isolated phase regression and crossing BCE fail through N627: no child
  predicts a crossing before non-event dense phase drift exceeds the hard gate.
- Group-disjoint N628/N629 probes show that no frozen N604 legal encoder,
  posterior, or prior feature linearly separates a true event from four hard
  pre-event negatives. Privilege intervention remains exact.
- N631 replaces the overstrict global binary view with a continuous one. True
  current active-gate forward distance is ordered in every held-out event
  window, but N604's decoder has only `0.0547` correlation and `216.6` steps of
  MAE. A privileged pose/rate oracle is diagnostic only.
- N632 rejects frozen decoder/head capacity: best held-out continuous
  correlation is `0.1005`, with essentially no monotonic windows. The next
  bounded optimizer must update the legal visual/recurrent representation on
  training-only continuous plane labels while preserving fixed dense N604
  features/actions and keeping reward/actor tensors frozen.
- Retain N587 as `11/64`. Do not run a native screen, FlightSim command, or
  Submission action from any N624-N632 diagnostic.

### N633-N669 successful-prefix representation continuation — 2026-07-28

- Short 17-step representation updates and frozen probes fail through N646;
  sequence-only trust changes that preserve N604 semantics are orders of
  magnitude too small, and the five-sample near-plane target is confounded by
  frame holding. Broad N604/N652 replay proves masks encode coarse range but
  neither existing recurrence nor one second of failure-heavy history meets
  near-plane/direction gates.
- The corrected exact-history collector/filter now supports float32 windows,
  deterministic full-start N587 collection, arbitrary episode offsets, and
  per-prefix replay selection. N667 records 192 successful 321-step prefixes
  at offset 2000; N668 selects 128 stable histories from 189 eligible with zero
  stochastic mismatch.
- N669 uses seven 128-step crop offsets so recurrent age cannot identify
  progress. A mask-only 128-unit GRU generalizes on untouched event groups at
  `0.960515` correlation, `0.249514 m` overall MAE, `0.240912 m` near-plane
  MAE, and `80.23%` direction; tail-only stays near zero. Successful legal
  visual trajectories therefore contain the missing state signal.
- This is capacity evidence, not a deployable model. Next integrate the target
  into unchanged N587 with exact legal crop-boundary replay, actor frozen, and
  fixed successful/broad action anchors. Freeze prior, all task heads, actor,
  critic, and target critic; use validation only before touching N668 test
  labels. No native policy screen or FlightSim action is authorized.

### N670-N672 integrated representation continuation — 2026-07-28

- N670 closes microscopic one-step integration: exact legal replay and the
  soft gradient path pass, but N587's frozen latent has `-0.0629` validation
  correlation and a `1e-6` update is ineffective. No test label is evaluated.
- N671 supplies the first continuous-child joint integration. Correlation
  reaches `0.7392` while validation and broad hard-action RMSE remain below
  `0.00034`, proving useful signal can grow without changing actor tensors.
  Numerical latent preservation nevertheless selects an early, incompetent
  child; near-plane scale and direction fail throughout.
- N672 removes that inappropriate latent constraint. Its step-400 validation
  state reaches `0.942461` correlation, `0.302188 m` overall MAE, and
  `0.803922` direction, but near-plane MAE is `0.542766 m`. Validation/broad
  action RMSE `0.001016/0.001715` exceed the fixed `0.001` bounds. The saved
  checkpoint is preservation-selected step 150, not step 400, and is rejected.
- Do not relax the failed admission gates or screen N672. First source-lock an
  export-only deterministic replay that reproduces N672's full validation
  trace and captures step 400 as a quarantined representation donor. Freeze
  that donor representation; calibrate only the training plane probe and actor
  distillation on N668 training plus N652 broad histories. Validation may
  select; test/native/FlightSim remain forbidden until the original progress
  and action bounds all pass.
- N587 remains `11/64`; no post-N522 FlightSim action has occurred and
  Submission is forbidden.

### N673-N676 representation calibration and test continuation — 2026-07-28

- N673 rejects exact recovery of N672's unsaved state: maximum eight-checkpoint
  trace error is `0.0160443` against `1e-7`. The similar exported child is
  quarantined and cannot seed later work.
- Fresh N674 passes donor-only validation at step 350. N675 then holds that
  representation bit-exact, calibrates the probe, and distills actor behavior
  to N587 over N668 training plus disjoint N652 histories. Every validation
  progress/action gate passes.
- N676 performs the only authorized N668 test read. Correlation `0.931859`,
  overall MAE `0.328685 m`, direction `0.807359`, and test/broad action drift
  all pass. Near-plane MAE `0.501634717 m` fails the fixed `0.500000 m` gate by
  `0.001634717 m`; reject N675 and do not reread or tune to this test.
- Next collect a larger episode-disjoint deterministic N587 successful-prefix
  corpus and freeze new validation/test groups before any fit. The consumed
  N668 data may join training only; it can no longer substantiate held-out
  claims. Target genuine near-plane support and retain the exact progress and
  action gates. No native/FlightSim/Submission action is authorized.

### N677-N680 disjoint corpus and sealed partitions — 2026-07-28

- N677 passes a deterministic full-start collection at episode offset 4000:
  416 exact-length Gate-1 histories, 64 agents, 412 vector steps, zero skipped
  histories/crash/learning/non-policy actions/FlightSim packets. Corpus/report
  SHA are `0d5bf433...`/`23bb1d6c...`.
- N678 selects 400 of 411 independently replay-eligible histories. Maximum
  selected deterministic/logit/action errors are
  `0.00174588/0.00007132/7.39470e-7`, with zero stochastic mismatch. Filtered
  corpus/report SHA are `bd79043d...`/`daea39cd...`.
- N679 fails before reading input because of a direct-CLI import defect and is
  not retried. Corrected N680 uses only the frozen `vector_step` identifiers to
  assign 272/64/64 train/validation/test events across disjoint whole groups.
  Manifest/report SHA are `91b9acf1...`/`75b70103...`.
- The N680 test split is sealed. Isolate it from development loaders before
  fitting, audit and improve true near-plane support on train/validation only,
  and preserve the N674 representation-donor then N675 frozen-calibration
  separation. Only a complete fixed-gate validation pass admits one test read.
  N668 is development-only. Native/FlightSim/Submission remain unauthorized.

### N681-N687 isolated support and near-balanced donor — 2026-07-28

- N681 materializes exact physical 272/64/64 files and seals test at SHA
  `ef43fd77...`. N682 uses train/validation only and measures 43--49 near-plane
  samples per event; crop overlap underweights them from `15.9%` unique support
  to `8.86%` of training exposure.
- N683 rejects the four-row recurrent-reference batch at `0.00766128`
  deterministic error. N684's reporting bug writes no artifact. N685 proves a
  fixed padded 416-row batch passes exact replay at
  `0.00174588/0.00007123/0`; N686 passes a one-step integration smoke.
- N687 corrects plane-level exposure and doubles near-plane weight. Its step
  400 validation reaches `0.903538` correlation, `0.421591 m` overall MAE,
  `0.500221 m` donor near-plane MAE, and passing action/process gates. Direction
  `0.713914 < 0.75` rejects it; checkpoint `52048c9b...` is quarantined.
- Next correct temporal-delta pair exposure under the same data/model/settings,
  require a clean one-step smoke, then one fresh donor. Test, native,
  FlightSim, reward/prior, and Submission remain forbidden.

### N688-N695 pair-weighted and balanced-sampling continuation — 2026-07-28

- N688/N689 quantify and validate inverse crop-pair exposure with a `2.0`
  either-endpoint near-plane multiplier. N690's step 350 is the first state on
  the physical N681 validation partition to pass all donor progress and
  preservation gates, but the old correlation-first selector saves a regressed
  step 400. N691 audits the corrected complete-progress-first selector without
  loading a model or dataset.
- N692 is the required fresh-seed test and fails direction/near-plane at
  `0.676711/0.772668 m`. N693 localizes a real minibatch-support confound:
  N692 is in the bottom `11%/8.5%` for near/pair exposure, current schedules
  leave 58--67 crops unseen, and near/pair exposure correlates `0.989558`.
- The default-off balanced sampler passes N694's one-step contract. N695 gives
  every crop exactly three exposures and every event exactly 21, with exact
  source/action/process isolation. Its `0.880770` correlation, `0.464102 m`
  overall MAE, and `0.503379 m` near-plane MAE pass, but direction `0.646205`
  fails. Quarantine all N689/N690/N692/N694/N695 checkpoints.
- Next isolate the co-adaptation mechanism rather than run another seed: audit
  representation-gradient norm, pairwise cosine, and sign agreement across
  independent training-only probe initializations on one fixed balanced train
  batch, then repeat after a probe-only balanced warmup while N587 tensors stay
  exact. Only materially improved, finite, source-locked alignment may admit a
  warmup implementation/smoke and one fresh donor.
- Keep N681 `test_sealed.npz` unopened and all donor thresholds unchanged.
  Frozen-representation actor calibration, the one test read, native screening,
  reward/prior continuation, FlightSim, and Submission remain unadmitted.

### N696-N703 decoder-free channel diagnosis — 2026-07-28

- Probe-only warmup worsens cross-seed representation-gradient alignment in
  N696. A deterministic-only actor-insensitive channel then fails to reach the
  visual encoder/posterior in N698.
- N699 measures a valid actor-null/simplex-tangent channel, while N700 confines
  its legacy report correction to a report-only audit. N701 passes a one-step
  default-off implementation smoke.
- N702's paired three-epoch donor remains below direction admission at
  `0.656994` despite passing correlation/overall/near metrics and every process
  guard. N703 proves the fixed tangent readout is worse (`0.639137` direction,
  `0.661581 m` near-plane MAE). Quarantine all children and close a weight sweep.
- The next development gate is a train-only, decoder-free, rotation/sign-
  invariant covariance or pairwise geometry with no random probe gradient.
  Audit gradient reach and invariance first, smoke one balanced update second,
  then fit a deterministic readout on frozen train features before validation.
  Donor thresholds remain unchanged.
- The physical N681 sealed test remains unopened. N587 remains `11/64`; native,
  reward/prior/actor continuation, FlightSim, and Submission remain frozen.

### N704-N705 invariant progress implementation — 2026-07-28

- N704 validates a decoder-free, rotation/sign-invariant weighted covariance
  objective for plane level and temporal delta. It is finite, non-collapsed,
  reaches all three RSSM components, leaves N587 exact, and has no validation/
  test/model-output/native/live path.
- N705 passes one balanced update with probe features detached. The probe has
  no representation-gradient contribution; only allowed RSSM tensors change
  and all replay/action/resource/isolation gates pass. Its child is quarantined.
- Next source-lock a fixed-lambda affine ridge fit on frozen N681 train
  features only. After a contract smoke, train one fresh exact-three-epoch
  invariant representation from N587 and evaluate it with that unchanged
  frozen readout. Validation cannot tune lambda or the representation schedule.
- Sealed test, native, reward/prior/actor continuation, FlightSim, and
  Submission remain unadmitted.

### N706-N710 invariant representation donor admission — 2026-07-28

- N706 freezes the affine-readout procedure before donor training: one
  train-only weighted float64 ridge at lambda `0.001`, deterministic double
  solve, no model/checkpoint/test/native/live effect.
- N707 passes exactly three balanced invariant-representation epochs from
  N587 with detached random probe, complete crop/event exposure, action and
  parameter preservation, source/process/resource isolation, and only allowed
  RSSM changes. Checkpoint/report SHA are `8f415f85...`/`2bb3f31c...`.
- N708 fails before report output on the CUDA/DXG path. N709's CPU recovery
  identifies a report serialization bug. Both are rejected under unique tags.
  N710 repairs only the NumPy-boolean boundary and passes.
- N710's fixed ridge has coefficient SHA `bd2b1e32...`, all 512 features
  active, and validation correlation/overall/near/direction
  `0.957912/0.261334 m/0.443587 m/0.861979`. Model tensors are exact and all
  optimizer, checkpoint, test, native, and FlightSim counts are zero. Report
  SHA is `e7e3492f...`; N707 is the admitted representation donor.
- The next product gate is a physical-partition, mutation-free validation
  package. Adapt the N675 path to refit/store the exact ridge and verify N707's
  already passing strict N681-validation and N652 action bounds
  `<=0.001/0.01`, without a probe or actor update. Only a complete strict
  validation pass may authorize the single physical N681 sealed-test read.
- N681 test remains unopened. N587 remains the `11/64` native frontier;
  native/reward/prior/actor continuation, FlightSim, and Submission remain
  unauthorized.

### N711-N722 invariant posterior/prior promotion boundary — 2026-07-28

- N711 stores the fixed ridge without changing N707. N712 performs and passes
  the sole physical sealed-test read; that test is permanently consumed. N713
  improves paired native Gate 1 from `11/64` to `13/64`, zero crash/timeout.
- N716 repairs categorical prior KL to about `0.0002` and agreement to about
  `97.4%`, preserving posterior actor actions, but N717's fixed soft-progress
  gate remains above `0.10 m`.
- N719 selects level-only prior progress distillation at `3e-7`. N720 exactly
  reproduces that virtual step, and N721 validates all-horizon improvement to
  `0.1422..0.1840 m` MAE. All other prior gates pass; N720 is not yet an
  imagination parent.
- N722's resumed second minibatch fails its local MAE gate and emits no child.
  Replace minibatch continuation with a train-only aggregate-gradient virtual
  audit over all 1,904 pairs, then at most one aggregate prior step and immediate
  held-out admission. Reward-event RMSE `8.456` is next only after prior pass.
- Never reopen the sealed test. Native screening, actor training, FlightSim,
  and Submission remain forbidden; no post-N522 FlightSim packet was sent.

### N723--N735 closure and solve-first recurrent Puffer direction — 2026-07-28

- N731 is the retained prior diagnostic after the aggregate/direct open-loop
  audit. N732's held-out horizon `1/2/4/8/16` MAE is
  `0.135247/0.134998/0.131514/0.132079/0.157686 m`.
- N735 rejects N734 because held-out horizon 1 regresses `0.0001622075 m`
  versus N731. Its report SHA-256 is
  `46bfe16694965f514b6030f685cea37cb2c7bcd99268b7fb4ab9e327bbaab7f1`.
  Close the N523--N735 optimizer line and never reopen/hash/score the consumed
  N712 physical sealed test.
- The user superseded the prior Dreamer-only/no-classical-label training
  constraint. The new product path preserves the hard deployment boundary:
  every flown CTBR action is the complete deterministic output of one recurrent
  Puffer policy, with no runtime teacher, blend, planner, phase switch, pose,
  course coordinates, override, or fallback.
- Build the smallest actor first: causal all-red-pixel `64x64` soft mask,
  legal IMU/actuator/action/timing history, compact CNN, one GRU, and Gaussian
  actor. Use privileged native state only for offline oracle labels, curricula,
  reward, and a separate critic; establish multi-gate behavior with full-history
  BC/DAgger and finish with teacher-free recurrent PPO.
- Solve before optimizing time. Admission requires true `0.75 m` one/two-gate
  screens, reliable nominal and randomized six-gate completion with zero
  collisions, measured replay, export parity, and a zero-command Windows
  shadow. GateNet is a fallback only if official-frame evidence falsifies the
  simple red mask.
- FlightSim commands remain frozen after N522 and Submission remains forbidden.
  The canonical execution contract is
  `docs/vq2_paper_guided_puffer_goal_prompt.md`.

### VQ2-SF001 causal mask admission — 2026-07-28

- The first solve-first artifact maps each exact `640x360` JPEG independently
  to a flat `4096`-value `float32` soft-red mask. It scales the full frame to
  `64x36` and letterboxes 14 zero rows above and below. This preserves all
  horizontal evidence and maps the official camera intrinsics exactly to the
  native `64x64`, `fx=fy=32`, `cx=cy=32` projection.
- No contour, component, morphology, target selection, geometry, gate index,
  or action logic exists in the transformer. All red candidates survive in one
  dense channel. The implementation is therefore legal actor preprocessing,
  not an analytic controller or phase selector.
- Focused and adjacent camera/ABI tests pass `31/31`; eight retained official
  v3391 renderer frames yield finite, nonempty masks. Linux JPEG preprocessing
  is `9.262 ms/frame` or `107.968 Hz`. These fixtures do not claim VQ2 course
  coverage because the recorded passive VQ2 JPEG is absent from the current
  worktree.
- Implementation/report SHA-256 values are
  `1c220738ea9105bf35e50631f61df19ac29eb2ddefe338b6caf9530202d6dda9` /
  `accacceefc75ddc1cea157cf7f725b9dbe7d9791d6add6ec7829a5493ee0e32f`.
  VQ2-SF001 is admitted for offline preprocessing only.
- Next prove the native oracle on held-out uninterrupted six-gate episodes at
  the true `0.75 m` aperture before collecting any BC/DAgger labels. FlightSim
  remains frozen, Submission is forbidden, and N712's consumed sealed test
  remains untouched.

### VQ2-SF011 native oracle admission — 2026-07-28

- SF002--SF008 falsify spline, direct active-gate, velocity-damped, speed-only,
  and minimum-jerk alternatives without writing labels. The retained simplest
  controller is SF009's memoryless alignment governor: slow toward `0.2 m/s`
  while cross-track error exceeds `1 m`, center with coupled privileged PD,
  and rise smoothly to `2.0 m/s` once aligned.
- SF009 passes fixed `64/64`; SF010 passes randomized `512/512`. SF011 passes
  the complete disjoint admission at `4096/4096`, mean time `98.078308105 s`,
  with zero collision, miss, out-of-order, timeout, crossing-margin, action,
  wire-rate, or thrust-envelope violation.
- Every ordered gate is sampled in every episode. Mean radial errors are
  `0.042277/0.005679/0.002592/0.002648/0.002659/0.002579 m`. Report SHA-256 is
  `f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e`.
- This proves the training oracle only. The next product artifact is a
  source-locked full-history dataset containing exactly the `4118` legal actor
  values plus the complete executed CTBR label. The `34` privileged native
  values must never be saved into the actor dataset. FlightSim remains frozen
  and Submission forbidden.

### VQ2-SF012 legal recurrent-BC dataset — 2026-07-28

- The admitted oracle passes a new `64/64` randomized full-course collection
  screen and yields `403722` ordered causal transitions. Mean completion is
  `98.56494140625 s`; all safety and envelope diagnostics remain zero.
- The dataset stores only the leading legal ABI as a `4096`-value uint8 mask
  and `22` float32 sensor/history values, plus one complete float32 CTBR label.
  It stores none of the native `34`-value privileged tail.
- Every episode contains exactly one terminal on its last valid row. Both the
  online terminal-inclusive action-history audit and independent persisted
  nonterminal audit have maximum error `0.0`.
- Report SHA-256 is `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f`;
  metadata SHA-256 is
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`.
- This is training data, not a solved policy. Implement and source-lock the
  compact legal CNN/GRU behavioral-cloning baseline before any teacher-free
  screen. FlightSim remains frozen and Submission forbidden.

### VQ2 variable-gate strategy revision — 2026-07-28

- Ledger note: the SF013--SF068 and C001--C014 work after SF012 was recorded
  in standalone `docs/vq2_sf*.md` and `docs/vq2_c*.md` preregistration/result
  documents rather than appended here. Verified frontier from those records:
  SF063/SF064 pass Gate 1 `512/512` with zero crash but miss Gate 2 by a
  consistent `8.19 m` under-turn; SF068's exact two-gate PPO reached the
  Gate-2 plane only on a sampled trajectory (`0.054 m`) that the
  deterministic mean never consolidated, while whole-actor updates eroded
  Gate 1; C014 closed checkpoint interpolation and recommended phase-local
  recurrent PPO on the measured true/alias fixture.
- Two corrections supersede prior assumptions. First, the operator observed
  approximately `11` gates from the VQ2 start line; every SF screen so far
  trained and evaluated at `num_gates = 6`. The engine already supports up to
  `DRONE_RACE_MAX_GATES = 16` per instance via the `num_gates` kwarg, fixed
  per instance. Second, the corrected SF012 mask-gap diagnostic (after fixing
  `step_dt` normalization; true rate `64 Hz`) measures mask-empty gaps of
  p50 `62 ms`, p99 `250 ms`, max `359 ms`, killing the long-blind-gap
  hypothesis; the live failure mode is post-crossing re-targeting ambiguity.
- The active plan is `docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md`:
  variable-count course generation (`5..12` gates, count-agnostic phase
  scalar normalized by the fixed engine cap `16`), an oracle re-admission
  screen per count, a regenerated legal BC corpus (`>=1.5M` transitions),
  recurrent BC with `>=256`-step BPTT and `>=3x` transition oversampling,
  DAgger rounds on visited states, and phase-local PPO only if DAgger
  plateaus. Offline stages run on rented remote compute (Vast.ai class
  hardware); local RTX 3070/4-core hardware is the measured bottleneck.
- All prior constraints stand: one recurrent full-output Puffer actor on the
  legal mask/tail/phase ABI, teacher and privileged state offline-only, zero
  teacher blend in admission, unique tags with SHA-256 evidence, no unchanged
  retries. Repository checkpoint for the handoff is commit `508286e`.
  FlightSim remains frozen after N522 and Submission remains forbidden.

### VQ2 variable-gate environment — VG001/VG001B, 2026-07-29

- VG001 extends the existing native random-course machinery without changing
  its default path. Each vector environment may now select one fixed episode
  count in `5..12`; 512 instance indices assign exactly 64 of every count,
  and resets preserve that assignment. Optional bounded validation requires
  finite ordered planes, non-overlap, an `8 m` minimum forward gap, and a
  `40 m` maximum adjacent-center distance.
- The fixed public phase contract is now
  `clamp(active_gate_index, 0, 16) / 16`, held at the official `4 Hz` update
  cadence. Total gate count is never exposed. A count-agnostic native fixture
  proves the complete `4096` mask plus `22`-value legal tail is byte-identical
  for five- and twelve-gate courses when the visible state is identical.
- Both native regression suites pass, including `4096` randomized course
  constructions spanning every count `5..12`; all `32` focused Python tests
  pass. Existing SF012 collection now intentionally fails its frozen source
  lock rather than silently relabeling the old six-gate corpus.
- The first compiled-binding distribution report passed but used the default
  bf16 extension and is superseded; preserve report SHA-256 `0f67ff81...` as
  rejected ABI evidence. VG001B rebuilt `drone_race_vision` explicitly as
  float32 (extension SHA-256 `847d77ee...`) and passes 128 forced episodes:
  exactly `0.125` episode mass at every count `5..12`, unchanged across reset,
  with zero off-range mass. Accepted report SHA-256 is `0845ea46...`.
- VG002 is preregistered but not yet executed: four independent `512`-episode
  oracle screens at counts `{5,8,11,12}`, true `0.75 m` aperture, at least
  `99%` success for each count, and zero collision. No variable-gate corpus or
  policy training is admitted until it passes. FlightSim received no packet,
  N712's consumed sealed test was not accessed, and Submission remains
  forbidden.
