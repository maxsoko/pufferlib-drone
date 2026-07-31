# VQ2 48-hour competitive-lap goal prompt — 2026-07-31

This is the authoritative sprint prompt from `2026-07-31T00:22:45-04:00`
through the hard deadline `2026-08-02T00:22:45-04:00` (48 hours). Read
`AGENTS.md`, the tail of `PRD.md`, and
`docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md` before acting.
This prompt supersedes the older goal prompt only for deadline, execution
priority, current frontier, and Submission authority. It does not weaken any
competition-legality, collision, evidence, source-lock, or lifecycle rule.

## Goal

Produce the fastest competition-legal, collision-free, fully autonomous valid
lap achievable in `AI-GP Virtual Qualifier R2` by the deadline.

The deployed controller must remain one recurrent full-output PufferLib actor
using only official JPEG camera data, HIGHRES_IMU, previous action, and public
official race progress. Every four-channel plant action must be the
deterministic policy output. Native state, oracle actions, classical control,
critics, gate geometry, and reconstructed pose are training-only and may never
emit, mix, clip, schedule, override, or select a runtime action.

A valid lap requires monotonically ordered completion of the full official VQ2
course, a nonnegative `race_finish_time_ns`, no collision or invalid reason,
command rate below 100 Hz, clean lifecycle accounting, final disarm, and
passive disarm proof. Do not assume the course has six gates; the VQ2 UI
indicates approximately 11 and runtime track transfer is absent.

Optimize lexicographically:

1. Obtain a valid ordered full-course finish.
2. Make the finish repeatable and collision-free.
3. Reduce official elapsed time without losing validity or reliability.

An offline checkpoint, validation loss, partial gate count, shadow, or bounded
prefix is only a milestone. A slow valid lap is more valuable than a fast
crash.

## Authority

- Continue autonomously through source-locked offline collection, fitting,
  deterministic screening, export parity, command-free Windows shadow, and
  uniquely tagged bounded VQ2 **Training** attempts after their existing gates
  pass.
- Repair tooling and infrastructure failures and resume exact atomic state
  without waiting for routine confirmation.
- Stop or replace idle rented compute when needed. Do not leave paid compute
  idle while runnable source-locked work exists.
- VQ2 **Submission remains prohibited** unless the user gives a new explicit
  Submission instruction. Any older prompt language granting standing
  Submission authority is superseded.
- Never access the consumed N712 sealed test, use privileged runtime data,
  select the Submission row for probing, or retry a rejected configuration
  unchanged.

## Starting state

- Frozen VQ1 promotion N283 remains `3/3` valid with best official time
  `23.832376480 s`; do not spend sprint time changing it.
- VQ2 has no valid live lap. N295 passed Gate 1. N399 replayed exactly but
  collided after the Gate-1 transition because live camera aperture/range
  aliasing and elapsed-clock behavior were absent from its old native screen.
  Live commands remain frozen until a new candidate passes all offline and
  shadow gates.
- VG025 epoch 5 is the admitted six-source actor. VG026 is rejected at
  `0/256` finishes, although it reaches Gate 2 on `97/256` and Gate 3 on
  `2/256`.
- VG027 is admitted with `1,541,730` legal visited-state labels, Gate-1 reach
  `512/512`, Gate-2 reach `197/512`, Gate-3 reach `2/512`, and no Gate-4
  reach. It authorizes one separately preregistered seven-source refit.
- VG028 is the immediate runnable experiment. At sprint start its admission,
  preregistration, trainer, runner, and test surface are local and untracked;
  the retained Vast worker is running but idle at commit `8045af9`.

Always re-check the repository, remote process, and evidence state before
acting. Continue from newer valid evidence if another turn has advanced it.

## 48-hour execution schedule

### T+0 to T+1 hour — remove launch latency

1. Validate the VG027 admission hash and every VG028 frozen input.
2. Run the complete focused test shard, both native regression suites, shell
   checks, and a fresh SM89 float32 build in the intended remote environment.
3. Record final source hashes, commit the complete VG027/VG028 surface, push
   it, sync the exact commit to the retained Vast workspace, and launch VG028.
4. If VG028 cannot launch inside one hour, stop the idle worker after syncing
   irreplaceable evidence, repair the blocker locally, and relaunch on suitable
   compute. Do not burn the deadline or rental cost on an idle GPU.

### T+1 to T+6 hours — finish VG028

- Run the single preregistered six-epoch, epoch-resumable VG028 refit from
  VG025 epoch 5 with all seven sources and exact weights.
- Monitor each atomic epoch for finite loss, exact source-weight sums, `4x`
  transition exposure, GPU utilization, and wall time. Diagnose a stalled or
  slow pipeline immediately; do not silently wait through dead compute.
- Sync epoch state and reports locally as they appear. Select the lowest
  admitted fixed-weight validation epoch, not automatically the last epoch.
- A training or infrastructure interruption must resume exact state. A
  terminal numerical rejection requires a causally distinct new tag.

### T+6 to T+10 hours — cheap teacher-free decision

Run a fresh, deterministic, zero-teacher, count-5 diagnostic before spending
the full 256-course screen.

- Use enough agents to expose the Gate-1 transition quickly, with source-locked
  fresh seeds and the deployment ABI.
- Continue to the next rung only if the candidate shows a meaningful safe
  downstream improvement over VG026: an ordered Gate-3-or-later reach or a
  full finish, no transport/phase/action-envelope fault, and no same-fixture
  crash-rate regression.
- If it fails this rung, do not spend hours proving `0/256`. Preserve the
  diagnostic, identify the first causal divergence, and immediately start the
  next visited-state repair loop.
- If it passes, screen fresh counts 5 and 11, then run the full counts
  5/8/11/12 admission. Final offline admission remains at least `231/256`
  ordered finishes with zero crash and zero hard fault.

### T+10 to T+34 hours — rapid repair loops

Use short, source-balanced loops. Each loop must have a unique tag/seed,
preregistration, immutable result, and causal change:

1. Roll out the current deterministic actor on the earliest failing
   transition with a long enough horizon to retain post-transition states.
2. Query the training-only oracle on those visited states and store only the
   legal actor ABI plus oracle labels.
3. Retain all admitted clean and prior failure-distribution anchors with
   explicit per-source weights; do not let corpus size set importance.
4. Refit recurrently with exact action-history semantics and transition
   exposure.
5. Run the cheap diagnostic ladder before a full screen.

Budget roughly 4–6 hours per complete collection/refit/diagnostic loop and
prefer three informative loops over one oversized blind run. Prepare the next
collector or evaluator while GPU fitting is active when this does not mutate
the frozen running source.

If two causally distinct DAgger loops after VG028 show no improvement in
Gate-3-or-later reach, switch promptly to the already justified phase-local
recurrent PPO path:

- optimize the measured post-Gate-1 retarget/alias fixture;
- keep the actor input legal and the critic training-only;
- anchor clean/pre-transition behavior with BC or KL;
- evaluate only the deterministic mean;
- reject whole-actor unanchored PPO and sampled-only success.

Do not spend the sprint on checkpoint interpolation, decoder ridge fits,
classical blending, hand-coded runtime recovery, pose reconstruction, or
long-blind-gap machinery; those branches are closed by existing evidence.

### T+34 to T+42 hours — admission and live promotion

As soon as a candidate clears the full offline gate:

1. Freeze and hash the checkpoint, preprocessing, callable, runner,
   configuration, and environment.
2. Pass Linux/export and actual-Windows recurrent replay parity.
3. Preregister and execute a zero-command Windows shadow in active VQ2
   Training. It must run full preprocessing and inference while sending zero
   lifecycle or control packets.
4. If shadow passes, preregister one bounded VQ2 Training attempt with exact
   reset, rate, timeout, official-progress stop, collision/dropout abort,
   disarm, and passive-proof rules.
5. Diagnose any live failure from its legal trace before creating a distinct
   successor. Never retry unchanged.

Do not lower the offline crash or transport gates to force a live attempt near
the deadline.

### T+42 to T+48 hours — establish and improve the lap time

- First obtain a valid full-course Training finish and freeze it immediately as
  a recovery baseline.
- Confirm it with consecutive clean runs before optimizing speed.
- Optimize elapsed time only through a separately tagged full-policy child,
  using completion-first reward and a time penalty that cannot make crashing
  preferable to finishing.
- Use short offline comparisons and retain the fastest candidate that
  preserves the full admission and shadow gates. Compare live candidates by
  official `race_finish_time_ns`, not vision-estimated timing.
- At T-2 hours, stop starting experiments that cannot complete, validate the
  best admitted artifact, sync all evidence, and use the remaining window only
  for already-authorized confirmation or low-risk time improvement.

## Throughput rules

- The native rollout is CPU-bound. VG026 measured only `1.019x` speedup from
  4 to 32 OpenMP threads. Benchmark small 4/8/16/32-thread probes once per
  host and lock the fastest setting; do not assume the largest thread count is
  best.
- Use vectorized native agents, avoid tiny sequential GPU calls, and keep
  data loading, validation, and checkpoint hashing out of the optimizer hot
  path where deterministic identity permits.
- Full 256-course screens are promotion evidence, not routine diagnostics.
  Use staged fresh-seed ladders to reject weak candidates quickly.
- Preserve atomic resume state after every epoch and continuously copy small
  reports, manifests, admissions, and best checkpoints locally.
- Keep one authoritative worker unless measured contention shows that a
  second worker would shorten the critical path enough to justify its cost.

## Evidence and reporting

- Append every completed experiment to `PRD.md` and `AGENTS.md` with tag,
  seed, source commit, hashes, timing, outcome, verdict, and next authority.
- Report progress at least every two hours while work is active: current
  experiment, elapsed/remaining wall time, latest gate reach/finish/crash
  frontier, compute utilization, blocker, and next decision.
- Keep rejected datasets and checkpoints quarantined. Never silently revise a
  preregistration after its first optimizer or rollout step.
- At the deadline, provide the best official Training time and repeatability
  evidence. If no valid lap exists, state that plainly, preserve the strongest
  frontier and causal diagnosis, stop idle compute, and leave an exact
  resumable handoff. The deadline never converts an invalid or unsafe run into
  success.

## Terminal condition

The sprint goal is achieved when a source-hashed legal recurrent Puffer actor
records a collision-free, ordered full-course VQ2 Training finish with a
nonnegative official finish time, and the fastest safely verified time and
supporting lifecycle evidence are preserved locally. Submission is a separate
user decision.
