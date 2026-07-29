# VQ2 C009 measured visual suffix warmed DAgger fit preregistration — 2026-07-28

## Objective

Correct only the proven C007 recurrent-context mismatch. Prepend the frozen
`208`-row legal Gate-1 observation sequence to both C006 groups so training,
validation, and deployment begin Gate 2 from the same warmed state. Do not
collect new labels or change the plant fixture.

## Frozen implementation and inputs

- trainer: `scripts/train_vq2_measured_visual_suffix_warmed_dagger.py`
  - SHA-256 `30b69818ee19c62afc819796b85b2b6cba64561767e10e58891bf3646bb285c1`;
- focused tests:
  `tests/test_train_vq2_measured_visual_suffix_warmed_dagger.py`
  - SHA-256 `18c6b92e86171e38d636ea4de456f7ba8d021b37cbb041991e2ad37f83c2b1be`;
- combined warmed-fit/composite suite: `28/28` passing;
- C007 parent checkpoint/report SHA-256:
  - `393f5de6f9dd97ae81d855b23d77c96e97926507e7c2b89611714df3d60a24c5`;
  - `bfc99b3d5177c5e83d9d46653480c26c66e05eebbc8d1d1bbc960bacd77c6e49`;
- C006 and the three retained anchor groups remain byte-for-byte frozen under
  the hashes enforced by the C007 trainer;
- measured prefix report/trace SHA-256:
  - `deb0301428e7c781d5e981e298ed87d66f0aa7c65c5d12e99ec5cd9301055f6b`;
  - `f33ab23c6977e6e3b94937bc2b4f41cc9e1adf34651e5f8b5373675f3d4557af`.

The prepended targets are SF066's frozen unselected whole-Puffer outputs.
They anchor recurrent warm behavior but never reach the Gate-1 plant, which
remains owned entirely by N294. No privileged value is added to actor input or
stored record.

## Fixed fit

Run tag `vq2_c009_measured_visual_suffix_warmed_dagger_fit_001` exactly once on
CUDA:

- seed `43009`;
- parent C007;
- the same five intact 64-agent groups and last-eight-per-group validation
  split as C007;
- C006 true and alias histories become `208 + original length`, with exact
  prefix observations and SF066 prefix outputs before the original C006 rows;
- `16` epochs, learning rate `2e-5`, phase-zero/Gate-2 weights `1/2`, and the
  unchanged AdamW, gradient, chunking, action-weight, encoder/GRU/head training,
  and frozen-log-std contract.

## Admission

Require the unchanged C007 aggregate numerical gate and all five
source-specific weighted/per-action thresholds. Every metric must be finite,
and the saved checkpoint must source-lock the prefix and report zero FlightSim
or Submission packets.

Failure ends C009 without rollout. Passing authorizes one paired true/alias
C005 screen of the child. It does not authorize FlightSim.
