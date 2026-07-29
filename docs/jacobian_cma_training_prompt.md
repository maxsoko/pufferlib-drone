# Jacobian-Guided Constrained CMA-ES Continuation Prompt

Copy the text below into a fresh coding-agent session. This is a deliberately
new optimization branch; it is not permission to reopen the previously rejected
PPO, BC/DAgger, decoder-sweep, action-scaling, or one-generation ES branches.

```text
Goal: Starting from the retained full-policy checkpoint, implement and evaluate
one genuinely new optimization mechanism: Jacobian-guided, constrained CMA-ES
in a low-dimensional cross-layer policy-parameter subspace. First clear the
corrected-scope floor-4.5 Gate-2 failure while preserving floor-4 behavior, then
continue the existing native promotion ladder only when its gates are met.

Repository:
- /root/pufferlib-drone

Read these sources of truth before changing anything:
- docs/full_policy_execution_prompt.md
- docs/full_policy_iteration_report_2026-07-15.md
- scripts/policy_callable_checkpoint.py
- scripts/evolve_policy_tensor.py
- scripts/eval_drone_race_checkpoint.py
- tests/test_recurrent_replay_contract.py

Retained parent:
- logs/drone_race_full_policy_official_fit/official_speed_curriculum/
  ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/rollrel0p99975.bin
- Native precision: FP32.
- Checkpoint layout precision: FP32.
- Input dimension: 32 official-observable fields.
- Recurrent state must persist for each complete episode.

Current corrected-scope evidence (activation index 2, instantaneous legacy
transition-floor direction):
- floor 4 exact-1024: success 1.0, gates 4.0, crash 0.0, completion
  29.394688 s, Gate-2 radial 0.531454 m.
- floor 4.5 exact-1024: success 0.0, gates 2.0, crash 0.0, Gate-2 crossing
  0.0, terminal radial 0.980955 m, signed right/vertical error
  +0.664541/+0.721489 m.
- floor 4.5 stochastic log-std -6 is diagnostic only: radial 0.871013 m but
  zero Gate-2 crossings.
- floor 5 and floor 6 are worse. No official-simulator run is authorized.

Why this mechanism is new:
- The old ES branch made one antithetic generation in one raw tensor and formed
  one local fitness direction. It did not learn a persistent search covariance,
  did not span encoder/recurrent/decoder tensors, and did not enforce an anchor
  behavior constraint.
- This branch must construct a cross-layer sensitivity subspace from recurrent
  action Jacobians, then run multi-generation covariance-adapting population
  search in that subspace.
- A candidate is always materialized as a normal, merged PufferNet checkpoint.
  There is no adapter, residual controller, phase head, teacher, or arbitration
  at evaluation or deployment time.
- If the proposed implementation collapses to independent random perturbations,
  a coordinate sweep, one finite-difference gradient, or PPO with different
  settings, stop: that repeats a closed branch.

Hard constraints:
1. The policy still receives only the 32 official-observable inputs and emits
   every pitch, roll, yaw, and thrust command after arming.
2. Do not add position, velocity, gate coordinates, sampled floor, TRACK_INFO,
   or gate index to policy observations. Native-only state may be used only by
   the optimizer's fitness reporter, never by the policy.
3. Do not change course physics, transition-floor semantics, observation/action
   contracts, reward shaping, policy architecture, or checkpoint arithmetic to
   make the candidate pass.
4. Keep FP32 native arithmetic and FP32 checkpoint layout. Exclude log_std and
   the critic/value decoder row from the search. Promotion is deterministic.
5. Do not reopen PPO, BC, DAgger, phase-local action patches, global action
   scaling, decoder coordinate sweeps, physical ramps, direction variants,
   delay variants, or raw final-MinGRU tiny ES.
6. Use one training/evaluation process at a time. Preserve per-agent recurrent
   state across rollout chunks and gate transitions and reset it only at episode
   termination.
7. Do not run the official simulator until every existing native promotion gate
   is satisfied, including exact-4096. Preserve the healthy simulator process.

Mechanism to implement:

A. Reproduce and freeze baselines
- Re-run the unchanged parent at corrected-scope floors 4 and 4.5 exact-1024
  using the authoritative evaluator.
- Require the metrics above within deterministic numerical tolerance. If they
  do not reproduce, diagnose configuration/checkpoint drift before optimizing.
- Record exact commands, config fields, checkpoint hashes, precision values,
  reports, elapsed time, and results in the iteration ledger.

B. Collect two deterministic, official-observable recurrent trace sets
- Failure set: parent rollouts at floor 4.5, emphasizing the complete history
  from race start through the Gate-1-to-Gate-2 approach. Do not initialize the
  recurrent state from a segment reset.
- Anchor set: matched parent rollouts at floor 4 that complete the course.
- Store observations, hidden states, mean actions, episode boundaries, and
  outcome metrics. Privileged coordinates may appear only in a separate fitness
  report and must not enter Jacobian construction or policy inputs.
- Include tests proving trace replay reproduces the callable checkpoint's
  actions and hidden-state resets exactly.

C. Build a cross-layer recurrent sensitivity subspace
- Differentiate the deterministic mean action sequence, including recurrent
  history, with respect to the encoder, all three MinGRU projection matrices,
  and the four policy decoder rows. Do not differentiate or mutate log_std or
  the value row.
- Construct a matrix-free generalized sensitivity problem that favors parameter
  directions with high action controllability on the floor-4.5 failure trace
  and low action disturbance on the floor-4 anchor trace. Use damping and
  randomized SVD/Lanczos as needed; never materialize an impractically large
  dense parameter-by-parameter matrix.
- Retain a small orthonormal basis (initial target dimension 12; at most one
  evidence-backed expansion to 24). Normalize basis vectors in actual FP32
  checkpoint space and verify that pack/unpack round trips preserve them.
- Tests must cover determinism, orthonormality, excluded tensors/rows, FP32
  layout, full-history recurrent differentiation, and candidate reconstruction.

D. Optimize only the subspace coefficients with constrained CMA-ES
- Parameterize each candidate as theta = theta_parent + Bz, merge it into a
  standard checkpoint, and evaluate deterministic mean actions.
- Use a real multi-generation CMA-ES implementation with persistent mean,
  evolution paths, step size, and full covariance in z-space. Use antithetic
  sampling and fixed common-random-number development episodes.
- Pre-register one population size, parent count, initial action-space step
  scale, seed, and generation budget before seeing candidate outcomes. A
  reasonable starting budget is dimension 12, population 24, parents 12, and
  at most 8 generations. Do not start a hyperparameter grid.
- Calibrate the initial coefficient scale in action space: perturbations must be
  representable in FP32, large enough to change the floor-4.5 trace, and small
  enough to remain a local policy trust region. Record failure-trace and
  anchor-trace action RMS/max deltas.
- Rank candidates lexicographically rather than inventing another shaped reward:
  (a) valid finite checkpoint and zero invalid actions;
  (b) floor-4 anchor feasibility/action-drift constraint;
  (c) ordered Gate-2 crossing fraction at floor 4.5;
  (d) full success and ordered gates;
  (e) lower Gate-2 terminal radial error when no crossing occurs;
  (f) final progress, then smaller anchor action drift.
- Reject candidates outside the anchor trace trust region. At least the top
  candidates of every generation must also pass an actual deterministic floor-4
  development evaluation; trace similarity alone is insufficient.
- Never select on the exact-1024 promotion set. Use fixed development episodes
  for CMA-ES and a disjoint deterministic validation set for generation winners.

E. Continuation and stop rules
- A generation winner earns an authoritative exact-1024 evaluation only if it
  improves a discrete Gate-2 outcome or reduces held-out deterministic radial
  miss by at least 0.10 m while satisfying the floor-4 preservation constraint.
- Retain a floor-4.5 candidate only at success >= 0.90, crash <= 0.10, ordered
  completion, valid actions, and floor-4 success >= 0.90. Prefer the parent on
  ties or regressions.
- If the first three generations produce no feasible candidate with either a
  discrete improvement or >=0.10 m held-out radial improvement, inspect the
  basis spectrum and action-space calibration once. Do not tune fitness weights.
- Permit one basis expansion from 12 to at most 24 only if measured discarded
  sensitivity energy justifies it. This is the sole structural retry.
- Stop and reject the mechanism after 8 generations, or earlier if covariance
  collapses, candidates repeatedly violate the anchor, or held-out progress is
  absent. Do not restart with a new seed, tensor subset, sigma grid, or renamed
  variant.
- If floor 4.5 promotes, evaluate the unchanged winner at floor 5. Train further
  with the same mechanism only when the existing continuation rule is met. Then
  follow the established floors 6, 8, and 10, dropout, and exact-4096 ladder.
- No finer deterministic floor between 4 and 4.5 is authorized as a substitute
  for training progress.

Implementation quality:
- Prefer small reusable scripts, with unit tests, for recurrent trace capture,
  sensitivity-basis construction, checkpoint merging, CMA state serialization,
  candidate ranking, and resumability.
- CMA state and every evaluated coefficient vector must be saved so an
  interrupted run resumes exactly instead of repeating candidates.
- Cache reports by checkpoint SHA-256 plus complete evaluation config and seed
  set; never treat a filename as identity.
- Validate generated checkpoints with the existing callable/native parity tools
  before expensive evaluation.
- Run focused tests after each component, native C regressions if native code is
  touched, the relevant full Python suite before retaining a checkpoint, and
  git diff --check at handoff.
- Preserve all unrelated dirty-worktree changes.

Documentation contract:
- Append every experiment immediately to
  docs/full_policy_iteration_report_2026-07-15.md and update the current state in
  docs/full_policy_execution_prompt.md.
- Each record must include: exact command; hypothesis; parent/candidate hashes;
  changed tensors and coefficient vector; basis hash and spectrum; CMA mean,
  sigma, covariance condition number, generation and seed; development and
  held-out episode identities; action-drift constraints; floor-4 and floor-4.5
  metrics; elapsed time; accept/reject decision; and exactly one next action.
- Maintain a rejected-candidate table so no coefficient vector or unchanged
  configuration is evaluated twice.

Definition of success for this prompt:
- Minimum mechanism success: a retained deterministic floor-4.5 checkpoint with
  exact-1024 success >=0.90 and crash <=0.10 while preserving floor-4 success
  >=0.90.
- Ultimate success remains a single recurrent, official-observable full policy
  that passes the complete native promotion ladder and then achieves official
  full-course validation under the existing execution goal.
- If the bounded CMA-ES experiment is falsified, report it as a documented
  negative result and stop. Do not silently fall back to any closed branch.

Begin by auditing the corrected 32-field phase schema and reproducing the two
baselines. Then write the pre-registered experiment manifest and tests before
evaluating the first mutated checkpoint.
```
