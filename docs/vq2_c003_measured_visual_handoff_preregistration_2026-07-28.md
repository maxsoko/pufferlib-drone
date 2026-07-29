# VQ2 C003 measured visual handoff preregistration — 2026-07-28

## Purpose

Replace the rejected attempt to identify the live transition with a scalar
plant fit. Warm both recurrent Puffers on the source-locked, timing-matched
C002R gain-10 legal visual prefix, then continue from the retained N399
Gate-2 state and clock. This directly tests SF066 at the failure distribution
that the next single DAgger collection must solve.

## Frozen sources

- evaluator: `scripts/eval_vq2_measured_visual_suffix_handoff.py`
  - SHA-256 `8042eced21e25ef8a6b28e11d42f258826bf852c2747df04b8107a6f0e338e13`
- evaluator tests: `tests/test_eval_vq2_measured_visual_suffix_handoff.py`
  - SHA-256 `c76ad35dd82b713475fe9caf29e708ba10eba1487fd65149a823d365a6e97cc9`
- C002R result: `docs/vq2_c002r_prefix_gain_bracket_result_2026-07-28.md`
  - SHA-256 `1d6d4d1c06e5a6a63553ace03d7c48703f6c9cfe523162506df0cc28a081b0fc`
- C002R gain-10 report/trace SHA-256:
  - `deb0301428e7c781d5e981e298ed87d66f0aa7c65c5d12e99ec5cd9301055f6b`
  - `f33ab23c6977e6e3b94937bc2b4f41cc9e1adf34651e5f8b5373675f3d4557af`
- N404 measured-transition report/trace SHA-256:
  - `364050aace586f3c154b712e3bc6df43073edb645aa57f5b57af6547b8d50312`
  - `6f28c959c44efe6c0c7b15607a0d92f93c0c99896bdd297db6d44d4af828fd88`
- N294 and SF066 remain byte-locked by the evaluator to
  `a57ca5f4af1...` and `f4ee6782de66...`.

Focused composite/handoff tests pass `16/16`; all selected Python sources
compile.

## Exact contract

The warm prefix contains exactly `208` held-index-0 observations. Replaying it
must reproduce both stored Puffer outputs within `5e-5`. At handoff:

- continuous elapsed clock: `3.281 s` of `14 s`;
- velocity: `[4.677, 0.006, 0.173] m/s`;
- quaternion: `[0.998520, 0.052695, -0.013450, 0.000931]`;
- body rates: `[0.1226, -0.0006, 0] rad/s`;
- next gate: `[14.74, 8.70, 1.095] m` relative to the measured start;
- most recent selected prefix action:
  `[-0.044653, 0.231316, 0.1124859, 0.000208]`;
- held public phase: `1/6`, sampled on the existing `4 Hz` schedule; and
- SF066 owns the complete four-action plant vector from the first handoff
  observation while N294 continues to advance without owning a channel.

The two older action-history slots come from the final two C002R N294 actions;
thereafter the history contains only actions actually selected for the plant.
Teacher blend, analytic action, channel override, and student update remain
zero.

## Two fixed screens

Run exactly once each, `128` deterministic agents, seed `43003`, CUDA:

1. `vq2_c003a_measured_handoff_true_range_exact128`
   - mask zoom `1.0`, hold `0` steps;
2. `vq2_c003b_measured_handoff_live_alias_exact128`
   - first-association mask zoom `1.741775393486023`, held for `16` control
     steps, then exact native imagery.

The alias value is the source-locked ratio between the true measured Gate-2
range and N399's first associated raw range. Zooming is centered on the legal
soft-mask intensity centroid, so it changes apparent aperture/range while
preserving bearing. It does not move the physical gate, access native state at
runtime, or construct an action.

## Decisions

A fixture is valid only if both recurrent replays, whole-action selection,
finite/envelope checks, selected previous-action history, and plant delivery
pass within `5e-5`, with zero FlightSim/teacher/student/Submission action.

Policy admission for a screen separately requires `128/128` next-gate passes
with zero collision, miss, timeout, invalid/out-of-order event, or action
fault. Do not confuse a valid failing fixture with a promoted policy.

- If both screens pass policy admission, freeze SF066 and advance to the
  prescribed `512` exact/perturbed composite screens.
- If either fails policy admission while the fixture is valid, source-lock the
  failure and collect exactly one suffix-only policy-state DAgger dataset from
  these same true-range/alias fixtures. Do not tune the handoff, plant, mask,
  policy, or oracle first.
- If the fixture itself is invalid, stop and correct only its identified
  contract defect under a new tag.

This is command-free offline work. VQ2 Training and Submission remain frozen.
