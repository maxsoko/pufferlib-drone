# Execution prompt: solve, qualify, and submit Virtual Qualifier R2

Work autonomously in `/root/pufferlib-drone` until the official event
**AI-GP Virtual Qualifier R2 - Submission** records one competition-legal,
competitive `race_finish_time_ns`.

Read `/root/pufferlib-drone/AGENTS.md` before acting. This is the user's latest
goal amendment. It supersedes older instructions only in these respects:

- the promoted controller must make one official R2 Submission attempt;
- do not assume that official VQ2 completion means six gates;
- the former one-actor/no-public-progress research constraint is closed; and
- the legal two-Puffer composite below is the only active controller lineage.

Keep every other safety, evidence, source-lock, and autonomy requirement.

## Exact target and authority

```text
Windows package root:
C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391

Windows executable:
C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe

WSL executable:
/mnt/c/Users/anon/Desktop/AI-GP Simulator v1.0.3391/AIGP_3391/FlightSim.exe
```

Use **AI-GP Virtual Qualifier R2 - Training** only for final qualification and
measured failure evidence. Training is not the requested result. The terminal
result must come from **AI-GP Virtual Qualifier R2 - Submission**.

The user authorizes one Submission attempt after the controller passes the
minimal promotion gate below. Never use Submission for learning, probing,
calibration, or shadowing. A second Submission attempt requires renewed user
authorization.

Official completion authority is:

- ordered public `active_gate_index` transitions; and
- nonnegative official `race_finish_time_ns`.

Record the final observed index as the real course gate count. Never require
`active_gate_index == 6` unless runtime evidence for this exact event proves it.
The native six-gate curriculum remains useful offline training data, not an
official VQ2 completion rule.

## Decision: this is the simplest evidence-backed path

The paper context is already sufficient for this solution. It is insufficient
for a bit-exact SkyDreamer reproduction, which is irrelevant. Do no more paper
research, world-model work, representation probing, architecture search, or
ABI redesign. Verify current official attempt rules only at Submission
preflight.

Do not resume any of these closed branches:

- N380/N402--N522 calibrated pose, association, predictor, or 32-input Gate-2
  continuation;
- N523--N735 Dreamer/representation research;
- the SF013--SF068 monolithic visual actor or any unchanged rejected SF run;
- a classical runtime controller, detector-derived geometry controller, MPC,
  blend, residual rescue, or channel override.

Those branches are evidence, not new work. Reuse SF066 only as suffix
initialization; do not continue its old monolithic deployment contract. Their
decisive result is that the
32-input continuation repeatedly failed live, while SF068's visual policy
produced a sampled Gate-2 plane crossing with only `0.0537539609 m` radial
error. SF068's global PPO failed to consolidate that action and damaged the
already-solved Gate-1 behavior. Therefore isolate the phases: freeze the proven
Gate-1 policy and train one visual recurrent policy only for the suffix.

## One controller lineage

Deploy exactly this composite:

```text
official JPEG + IMU/actuator/timing/previous-action data
    -> unchanged legacy 32-value preprocessor -> frozen N294
    -> SF001 soft-red mask + legal tail + public progress -> visual suffix
    -> continuously advance both recurrent Puffers
    -> index 0: select N294's complete four-channel output
    -> index >= 1: select the suffix Puffer's complete four-channel output
    -> fixed proven CTBR/body-rate-thrust wire conversion
```

Public race status may select one complete neural output and may supply the
suffix's single monotonic progress scalar. It may not select channels, blend
actions, or trigger analytic control. Use the existing `active_gate_index/6.0`
feature scale for checkpoint continuity, but treat `6.0` only as a scale:
never clamp it, never infer a final gate count from it, and never use it as the
finish condition.

Every plant action must be the complete deterministic mean from one recurrent
Puffer checkpoint. The native oracle and privileged state are training-only
label/reward/critic sources and never emit, mix, clip, override, or rescue a
runtime action.

Source-lock these retained artifacts before work:

- frozen N294 Gate-1 checkpoint:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n294_full_blend_refine/alpha_0p60.bin`,
  SHA-256
  `a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6`;
- SF001 preprocessor: `scripts/vq2_soft_red_mask.py`, SHA-256
  `1c220738ea9105bf35e50631f61df19ac29eb2ddefe338b6caf9530202d6dda9`;
- active suffix parent SF066:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_sf066_aggregate_dagger_fit_001/policy_best.pt`,
  SHA-256
  `f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8`;
- clean legal SF012 anchor report/metadata SHA-256:
  `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f` /
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`;
- rejected SF068 diagnostic parent checkpoint SHA-256
  `90fa709467bdc0221cb9e96920d86609f94d9c4dab0a6df57e330422ceba6e95`.

SF068 is diagnostic only; do not promote it or rerun it unchanged. N294 is
immutable. SF066 is the one active suffix parent until a child passes the next
gate.

## Immediate solve path

### 1. Make one deployment-matched composite harness

Reuse the existing native environment, SF001 preprocessor, recurrent actor,
oracle, datasets, and evaluators. Add only the minimal composite plumbing:

- N294 owns every index-0 plant action;
- the visual suffix advances on every causal public observation from reset;
- its previous-action input is the action actually selected for the plant;
- public progress is sampled/held exactly like the 4 Hz official status;
- camera duplicates are deduplicated on `(frame_id, sim_time_ns)`;
- time is one continuous deployment clock; and
- at the first ordered index transition, the already-warm suffix owns the
  entire action vector.

Reproduce the known N294 Gate-1 prefix and transition distribution offline.
Use recorded official prefixes when available and the narrow measured timing,
camera, start, and plant variation already in the repository. Do not add a
new detector, reconstructed pose, gate vector, course coordinate, or broad
domain-randomization project.

If recurrent state, previous action, progress hold, or timing does not match
the deployment contract, fix that before training. Otherwise do not spend a
turn writing another audit.

### 2. Train only the broken suffix

Run one N294-prefix, policy-state DAgger iteration from SF066:

- preserve complete recurrent histories from reset;
- let N294 drive the prefix and the suffix policy drive all index-1-and-later
  transitions;
- query the admitted SF009/SF011 native oracle only for labels at suffix
  states visited by the policy;
- store only legal visual/sensor/progress/previous-action observations and
  complete four-action labels;
- retain SF012 and the accepted measured datasets as clean anchors; and
- keep teacher blend exactly zero in every evaluated rollout.

Screen the deterministic mean. If it is safe but misses consistently, perform
one suffix-only recurrent PPO refinement with a training-only privileged
critic. Begin PPO updates only after index 1, preserve full prefix burn-in, and
anchor the suffix's prefix outputs/state. Persist sampled near-centered
crossings and use success-conditioned imitation to move that behavior into the
deterministic mean. Do not update N294 and do not optimize Gate-1 behavior in
the suffix; the suffix never owns Gate 1.

At most two measured-distribution DAgger collections and one phase-local PPO
run are allowed for the same failure. If those do not improve the preregistered
closed-loop metric, stop and derive one source-locked causal diagnosis before
changing anything. Do not open another model, optimizer, perception, or
controller branch.

### 3. Admit the next official transition

Before sending any FlightSim control, one frozen composite must pass:

1. `512/512` exact native next-gate completions, with zero collision, miss,
   timeout, invalid state, out-of-order event, or nonfinite/envelope violation;
2. at least `510/512` on narrow disjoint measured perturbations, with zero
   collision and zero out-of-order event;
3. N294's frozen exact/perturbed Gate-1 regression and recorded action replay;
4. complete prefix/suffix recurrent replay, including selected previous action,
   public-status hold, duplicate frames, and the continuous clock;
5. Linux and actual-Windows deterministic action parity;
6. actual-Windows end-to-end inference above `50 Hz`;
7. exact source/checkpoint/runner/configuration hashes and fail-closed stream,
   rate, collision, and invalid-state guards; and
8. one visibly selected active-R2-Training zero-command shadow with every
   lifecycle and setpoint count zero.

Do not enlarge these screens or invent new admission suites without a measured
failure that requires it.

Then preregister exactly one bounded R2 Training attempt targeting the next
official index transition. Use the proven reset/arm/heartbeat/rate/abort/stop/
disarm/passive-proof lifecycle. Abort immediately on collision, invalid state,
stream failure, or rate failure. Reject an unsuccessful candidate without an
unchanged retry.

### 4. Finish without a six-milestone campaign

After the first new official transition passes, keep N294 frozen and extend
the same suffix policy offline across the existing full-course oracle data.
Do not create one checkpoint per gate and do not require six separate live
Training milestones. Run a bounded Training attempt only when the candidate is
admitted for every prefix currently supported by offline/official evidence.

If Training exposes a new failure before finish, do only this loop:

1. source-lock that legal prefix and failure;
2. reproduce it offline;
3. add one policy-state DAgger collection for that suffix distribution;
4. retain every passed prefix as an anchor;
5. rerun the same compact admission; and
6. make one new, uniquely tagged bounded Training attempt.

Stop this loop immediately when Training reports nonnegative
`race_finish_time_ns`. Freeze that solve bundle and run one identical,
separately tagged confirmation. One solve plus one clean confirmation is enough
for promotion; do not collect extra laps without a concrete failure signal.

## Put down a competitive time

Preserve the repeatable solve bundle unchanged. Optimize only measured split
losses in its official Training traces: unnecessary braking, dwell, poor exit
speed, or conservative speed targets. Change one scalar speed/control target
at a time and preserve solve traces as anchors.

A time candidate must retain exact and perturbed completion before one bounded
Training validation. Keep it only if it is valid, collision-free, and faster.
Before an optimized child replaces the baseline, run one identical confirmation
so the promoted bundle itself has two clean complete Training results.
Stop after three consecutive non-improving candidates or when modeled gain is
below normal run-to-run timing variation. Do not chase proxy reward, launch a
hyperparameter sweep, or trade reliability for time.

The fastest controller with two clean complete Training runs is the promoted
Submission bundle.

## Official R2 Submission

Immediately before selecting Submission:

1. verify current attempt/locking rules from an authoritative official source;
2. verify the exact visible row **AI-GP Virtual Qualifier R2 - Submission**;
3. verify the bundle is byte-identical to the fastest promoted controller;
4. rerun focused Linux and actual-Windows parity/inference tests;
5. verify simulator and streams receive-only, with zero control; and
6. preregister the one attempt's lifecycle, hard bounds, hashes, evidence path,
   abort conditions, and nonnegative-finish success signal.

Execute the one authorized Submission attempt autonomously. No human input is
allowed during the timed run. Accept only ordered official progress with a
nonnegative `race_finish_time_ns`, null collision/invalid reason, healthy
streams and rates, exact hashes, final stop, disarm, and passive proof. Preserve
the raw trace and an independently regenerated summary.

If Submission fails, stop. Never retry unchanged and never make a second
Submission attempt without renewed user authorization.

## Anti-loop contract

Every work item must directly do one of these:

1. make the N294-prefix visual suffix pass the next official transition;
2. preserve passed prefixes while reaching official finish;
3. reduce a promoted valid time without reducing reliability; or
4. qualify and execute the authorized official Submission.

Otherwise skip it.

Keep one active suffix parent, one active failure hypothesis, and one material
variable per experiment. Preregister a numerical pass/fail criterion before
expensive work. Close a branch after two non-improving iterations. Reuse
existing scripts/tests/assets; do not recreate settled infrastructure. Write
only the minimal preregistration/result evidence needed to prevent repetition.
Do not work on R1, publish Training as Submission, or stop at an offline score.

## Runtime invariants

- MAVLink UDP `14550`; camera UDP `5600`
- camera `640x360`, nominal unique-frame rate `30 Hz`
- camera intrinsics `fx=fy=320`, `cx=320`, `cy=180`
- measured camera optical uptilt `1.920944634732011 deg`
- deterministic physics `120 Hz`
- command target `60--90 Hz`, always `<100 Hz`
- heartbeat `>=2 Hz`
- reset: learned-target `COMMAND_LONG` `31000`, confirmation `0`, parameters
  1--7 all zero
- collision/invalid state: immediate abort

Keep a healthy simulator process alive; restart is recovery-only. Preserve all
unrelated dirty-worktree changes.

## Final deliverable

Return only after a valid official **AI-GP Virtual Qualifier R2 - Submission**
result or a genuine external blocker. Report the official finish in nanoseconds
and seconds, every observed split and final index, controller/runner/config
hashes, collision/invalid/stream/rate/lifecycle status, Training promotion
evidence, Submission raw/summary hashes, and whether another Submission attempt
is authorized.
