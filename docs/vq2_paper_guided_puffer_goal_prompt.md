# VQ2 solve-first, then competitive-lap continuation goal

Work autonomously in `/root/pufferlib-drone` toward a competition-legal,
repeatable, collision-free valid lap in **AI-GP Virtual Qualifier R2 -
Training**, then optimize the official elapsed time without weakening
reliability.

The user-supplied repository `AGENTS.md` on 2026-07-28 is authoritative and
supersedes older goal prompts, hidden conversation assumptions, and later
research branches that conflict with it. Read it before acting. Runtime packet
evidence supersedes stale SDK comments, native approximations, and paper
assumptions.

Never select or control VQ2 Submission without a promoted, repeatable Training
result and a separate explicit user decision.

## Outcome and ordering

The primary objective is an official six-gate valid Training lap:

- official `active_gate_index >= 6`;
- nonnegative `race_finish_time_ns`;
- null collision and invalid-state reasons;
- no human input during the timed flight;
- healthy camera, IMU, actuator-feedback, telemetry, and command streams;
- command rate in `[50,100) Hz` and heartbeat at `>=2 Hz`; and
- proved reset, arm, final stop, disarm, and passive disarm.

One finish proves possibility but is not promotion. First make the finish
repeatable. Only then minimize official elapsed time. Never trade a valid,
collision-free finish for a faster failure.

## Authoritative VQ2 boundary

Use only the supplied v3391 VQ2 build:

`C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`

Development uses the Training event only. The runtime exposes camera,
`HIGHRES_IMU`, actuator status, heartbeat, race status, and event-driven
collision. It does not expose `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, or
track transfer. Official race status is the sole gate-pass authority.

The controller may use deterministic camera/IMU preprocessing, but every
flown pitch, roll, thrust, and yaw action must come from a recurrent PufferLib
checkpoint. Native/classical control may generate offline labels, rewards,
critic inputs, and diagnostics. It may never emit, blend, clip, schedule,
override, arbitrate, or rescue a runtime action.

The current legal deployed observation remains the established **32-value
camera/IMU/action/progress ABI**:

- camera-derived gate pose and gate-rate values;
- gyro and gyro-integrated attitude values;
- one continuous deployment-matched elapsed fraction;
- the previous complete action;
- normalized official gate progress;
- six official active-gate flags; and
- the two established reserved values.

Do not replace this contract with the later 4,119-value dense-mask research
ABI. Do not add vehicle coordinates, gate coordinates, native state, future
course geometry, decoded privileged state, or simulator-only values.

The existing legal composite contract is retained: frozen N294 may emit every
Gate-1 action while one separately warmed recurrent suffix policy advances on
the same public observations and emits the complete four-action vector from
official index `1` onward. Both sources are whole-output Puffer checkpoints.
Official progress may select which complete Puffer output is used; there is no
action blend or classical fallback. Prefer one suffix policy for Gates 2--6
over a growing bank of per-gate policies unless measured evidence proves that
it lacks capacity.

## Current evidence and command freeze

- Frozen N294 Gate-1 checkpoint SHA-256:
  `a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6`.
  It is exact `512/512`, perturbed `422/512`, and passed the full-rate Windows
  shadow.
- N295 passed official Gate 1 once in `3.228605508 s`, with clean transport,
  stop, and disarm evidence. Do not retry N295 unchanged.
- Retained Gate-2 native frontier N380:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n380_gate2_full_policy_reward_bracket/teacher100_align20/update_0001.bin`,
  SHA-256
  `6389d30a6c03eb680d0cb205e91b52690c05cf74bf9871b6eaee1897779e838a`.
- N388 proved the two Puffer checkpoints can be advanced continuously and
  selected without blending. Its source-locked N295-prefix replay matched all
  `195` samples within `4.805624485015869e-7`, and its Windows runtime tests
  passed.
- N398 passed an active-Training zero-command composite shadow: `527/527`
  inference samples, maximum replay error `4.0513782501028217e-7`, healthy
  streams, zero setpoints, and zero lifecycle commands.
- N399 ran exactly once and is rejected without unchanged retry. It passed
  Gate 1 at `3.194960355 s`, then collided at official index `1` around
  `5.845 s`. Transport and Puffer replay were correct.
- N401 replayed all `340` N399 decisions within
  `6.556510925292969e-7`. The failure was not inference or transport drift.
  At the Gate-1 transition, camera-derived pose jumped from approximately
  `[14.736, 8.608, -1.233]` to `[8.488, 4.978, -0.364] m` in `31 ms` while
  bearing barely moved. The old native screen also warmed on a `12 s` clock
  and began Gate 2 on a separate `20 s` clock; N399 used one continuous
  `14 s` deployment clock.
- No FlightSim command has been sent after N399. Keep that freeze until the
  complete offline admission below passes. VQ2 Submission remains forbidden.
- The SF001--SF068 4,119-input dense-mask branch is a superseded offline
  research detour under this prompt. SF068 completed `48` PPO updates but no
  deterministic Gate-2 finish; it sent zero FlightSim packets. Do not continue
  that ABI or treat any SF checkpoint as deployable evidence.
- Frozen VQ1 N283 remains evidence only and is not a VQ2 candidate.

## Paper-context decision

The locally reviewed SkyDreamer paper is sufficient for **design guidance**:

- separate legal actor inputs from training-only privileged information;
- use recurrent state for partial observability;
- train with asynchronous sensing and persistent visual/dynamics variation;
- standardize camera geometry;
- use an asymmetric training signal while deploying a deterministic policy;
  and
- expose an explicit legal progress/flight-plan signal rather than forcing
  recurrence to rediscover task phase.

It is not sufficient for an exact paper reproduction. The public material does
not include the exact training repository, checkpoint, replay ordering, or
complete experiment configuration. Those missing artifacts are nonblocking:
the official task does not require SkyDreamer or end-to-end learning, and the
repository already has a legal observation ABI, a proven Gate-1 policy, a
corrected native teacher, recurrent Puffer training code, and exact live
failure evidence.

Therefore the answer to “is all required paper context present?” is:
**yes for this VQ2 solution plan; no for a bit-exact SkyDreamer reproduction**.
Do not invent unpublished details or make exact-reproduction claims.

## Simplest credible solution

Do not restart perception, replace the 32-value ABI, enlarge the model, build a
world model, or train a monolithic six-gate policy from scratch. Repair the one
measured mismatch that invalidated N399:

```text
legal camera + IMU + previous action + official progress
        -> unchanged deployment 32-value observation
        -> N294 recurrent Puffer output while official index == 0
        -> one continuously warmed recurrent suffix-Puffer output afterward
        -> fixed SET_ATTITUDE_TARGET body-rate/thrust conversion
```

The suffix actor must learn that camera range and gate association can jump at
an official transition even when bearing is continuous. Make that invariance a
training distribution, not a runtime analytic controller. Preserve bearing,
angular motion, IMU history, previous action, and official progress while
randomizing or dropping the fragile range/association components around the
source-locked N399 transition.

Use exactly one continuous elapsed normalization matching the bounded runtime.
Do not splice a 12-second prefix clock into a 20-second suffix clock. Warm the
suffix recurrent state on N399's actual legal Gate-1 observation prefix before
every deployment-matched Gate-2 screen.

## Solve-first work order

### 1. Reconstruct the deployment-matched offline contract

Source-lock N399's actual Gate-1 prefix and the N401 analysis. Build one
command-free replay/native harness that:

- advances N294 and the suffix policy on every legal observation;
- uses the exact runtime 32-value observation order and scaling;
- preserves previous-action feedback exactly;
- uses one continuous `14 s` elapsed clock;
- reproduces the measured Gate-1 exit state and N399 association/range jump;
- injects bounded source-locked alternatives: range scale/jump, temporary gate
  dropout, association swap, duplicate/held camera frames, timing jitter, and
  measured camera uptilt;
- never puts the injected latent values into the actor; and
- proves action delivery and recurrent replay against the N399 trace.

The first diagnostic should reproduce the old N380/N399 failure mode. If the
harness cannot do that, fix the harness before training.

### 2. Train only the continuation that is broken

Initialize the suffix actor from N380. Preserve frozen N294 and the exact
prefix selection contract.

Start with full-history DAgger on uninterrupted measured Gate-1-to-Gate-2
episodes:

- the suffix actor advances through the complete N399-matched prefix;
- its own actions drive every Gate-2 plant transition;
- the corrected native state-feedback teacher labels visited Gate-2 states;
- privileged state is absent from stored actor observations and runtime;
- episode histories remain intact and recurrent state resets only at genuine
  episode boundaries; and
- clean nominal histories remain anchored while alias/dropout cases expand.

Train all four action channels jointly. A positive teacher blend is permitted
only to collect explicitly non-admissible training data; every screen is
teacher blend exactly zero.

If DAgger reaches a safe but consistent miss, use recurrent PPO on real native
transitions with a separate training-only privileged critic. Optimize ordered
progress, centered crossing, valid finish, collision/invalid penalties, a
small time cost, and mild control smoothness. The critic is not exported. PPO
must not alter N294 or add a runtime residual/controller outside the suffix
Puffer actor.

### 3. Admit Gate 2 before touching FlightSim

Require all of the following from one frozen checkpoint and source-locked
wrapper:

1. exact deployment-matched two-gate screen: `512/512`, zero collision, miss,
   timeout, invalid state, and out-of-order event;
2. disjoint N399-transition perturbation screen: at least `4090/4096`
   completions and zero collisions or out-of-order events;
3. no regression of frozen N294 on its exact and perturbed Gate-1 screens;
4. complete N399-prefix recurrent replay and selector parity;
5. causal camera/IMU/preprocessor replay, including duplicate-frame
   deduplication and status sample-and-hold;
6. Linux/actual-Windows action parity and inference comfortably above `50 Hz`;
7. exact source/checkpoint/runner hashes and all stream/rate fail-closed guards;
   and
8. a visibly selected VQ2 Training zero-command shadow with zero reset, arm,
   setpoint, and disarm counts.

Passing these gates permits only a separately tagged and preregistered bounded
Training attempt targeting official index `2`. Reuse the known-safe lifecycle:
one detected command-`31000` reset, requested `80 Hz`, hard time bound, immediate
collision/invalid/dropout/rate abort, one exit disarm, and passive disarm proof.
A failed live candidate is rejected without unchanged retry.

### 4. Extend one suffix policy through Gates 3--6

After a clean official Gate-2 proof, freeze that checkpoint as the anchor and
extend the same continuously warmed suffix policy one ordered gate at a time.
Use official progress flags already present in the 32-value ABI; do not add
course coordinates or new runtime controllers.

For each new gate:

- preserve all earlier exact/replay/live prefixes;
- collect full-prefix policy-state DAgger labels at the new failure states;
- add only the measured new visual/timing/plant perturbations;
- use PPO only after imitation produces a safe teacher-free approach; and
- require exact plus perturbed zero-collision admission before the next live
  milestone.

The first full-lap candidate must pass all six ordered gates and expose a
nonnegative official finish time. Then require repeat full Training laps with
clean lifecycle and passive-disarm evidence before promotion.

## Competitive lap-time phase

Freeze the first repeatable solve checkpoint, runner, configuration, hashes,
and traces. Optimization begins from that reproducible baseline and never
overwrites it.

Use a lexicographic objective:

1. six-gate completion, zero collision, and stream/lifecycle validity;
2. worst-gate centered clearance and perturbation reliability;
3. elapsed race time.

Increase speed gradually by reducing unnecessary braking and raising the
training-only teacher/trajectory speed target. Retain anchored solve episodes
and every accepted live prefix. Select only candidates on the reliability/time
Pareto frontier; reject a faster child if exact or perturbed completion,
worst-gate margin, collision rate, or replay parity regresses.

Use official Training traces only to calibrate legal preprocessing, clock,
camera-association, plant, and timing distributions. Never turn reconstructed
coordinates or privileged state into actor inputs. Submission requires a
separate explicit user authorization after a promoted repeatable Training lap.

## Immediate next action

Do not send FlightSim traffic and do not resume the 4,119-input SF branch.
Create the source-locked N399/N401 deployment-matched 32-input replay harness,
verify that it reproduces the N380 failure under the continuous `14 s` clock
and measured association alias, then preregister one N380-derived full-history
DAgger experiment for the warmed suffix policy. Keep N294 byte-frozen.

Prefer this one coherent lineage over broad architecture searches. The project
metric is ordered, collision-free completion first and elapsed race time
second.
