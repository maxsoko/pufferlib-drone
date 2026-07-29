# VQ2 VG006 variable-gate teacher-free screen preregistration — 2026-07-29

## Candidate and decision

Screen exactly one checkpoint, VG005 `policy_best.pt`, SHA-256
`f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3`.
Its training report SHA-256 is
`bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b`.
VG005 completed 12 fixed epochs and selected epoch 12 at validation weighted
MSE `5.728148761635799e-5`. It proves 38,622 optimizer updates, minimum 3.0x
transition-window exposure in every epoch, all 1,920 corpus phase increments,
zero actor privilege/teacher blend, and the required single 256-wide GRU.

Run one resumable aggregate tagged
`vq2_vg006_variable_gate_recurrent_teacher_free_256`. This is an exact native
closed-loop screen, not training. No checkpoint or hyperparameter may be
selected from its outcomes.

## Held-out course matrix

- Fixed gate counts: 5, 8, 11, and 12, run independently in that order.
- 64 vector agents and exactly one full-start episode per count; 256 new
  courses total.
- Seeds: count 5 -> `429037`, count 8 -> `429040`, count 11 -> `429043`, and
  count 12 -> `429044`. These are distinct from VG003/VG005 seed `429030`/
  `429031`.
- True aperture radius 0.75 m, admitted reachable randomized-course geometry,
  360-second maximum episode horizon, float32 native binding.
- Every teacher action/blend/spline/minimum-jerk/alignment/reward path is zero
  or disabled. Every plant action is the deterministic mean of the one VG005
  four-output actor; no sampling, clipping beyond the frozen actor tanh,
  analytic control, phase switch, planner, or fallback is permitted.

The native training-only `ordered_gate_phase` mirror is read only to reproduce
the legal official status packet in this offline simulator. It must be exact
`active_gate_index / 16`; the evaluator samples it at 4 Hz and holds it for 16
64-Hz policy steps. The actor receives the 4,118 legal camera/IMU/actuator/
action/timing values plus that one held scalar and no other native trailing
value. Total gate count is never an actor input.

## Transport and admission

For every count, require 64 terminal episode logs, finite actions inside
`[-1,1]`, executed-action history agreement within `5e-5`, zero held-phase
off-tick change/decrease/skip, and raw `/16` encoding error at most `1e-6`.
Count reports preserve native gate-sampled/crossing metrics and maximum public
index distributions to localize any failed transition.

Stage-2 admission requires at least 90% full-course success across all 256
courses (therefore at least 231 successes), plus zero crash, out-of-order,
crossing-margin, action, wire-rate, or thrust-envelope violation across every
count. Timeout/miss failures may exist only within the remaining sub-10%.
Per-count success is reported but not separately thresholded. A failure at one
localized transition routes to source-balanced full-course DAgger; it never
authorizes a checkpoint retry or live activity.

## Resumption and evidence

Output is
`logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg006_variable_gate_recurrent_teacher_free_256`.
Before count 5, `state.json` locks the Git commit, sources, checkpoint/report,
new compiled-extension hash/path, runtime, counts, agents, seeds, and safety.
Each completed count is atomically published once as `count_<N>.json`, hashed,
and added to state. An interruption may rerun only the missing count with exact
identity. Once state exists, the Vast wrapper must not rebuild, test, install,
or change dependencies. After all four counts, aggregate report/state status
is terminal `admitted` or `rejected`; an unchanged retry is forbidden.

## Frozen pre-runtime sources

- `scripts/eval_vq2_variable_gate_recurrent_policy.py` — `e4ca39c4f66ef2348c123f9d8d0e9014edbe7032758e19cdb3aa6b470d11815b`
- `scripts/run_vq2_vg006_vast.sh` — `bf29537e84672c170667b7d63a48c447a9091c439674c5a1608d2a8e4857544f`
- `tests/test_eval_vq2_variable_gate_recurrent_policy.py` — `203114c354f3f6be9eae42a6d8a3e953a13a8624d3e91430e3af4fbe9003f473`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` — `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `scripts/eval_vq2_variable_gate_oracle.py` — `ec65b09c614b16240717266f32fd49221f7f7075b5540eb802c6008f2acd93c6`
- `scripts/eval_vq2_recurrent_policy.py` — `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa`
- `ocean/drone_race/drone_race.c` — `523aa40a4578e0ec4b8808686faa5ba2d58d1b681091d49425456c1b39b1657b`
- `ocean/drone_race/drone_race.h` — `58615ed1d7dd4738d5fcf04d7cdf4b43a6a6cbab9dccc393c0d665820f7f5b06`
- `ocean/drone_race/binding.c` — `409ed4689f145b9a1f17c858535ffd8842fe9aaef5cf2e67c82945e51effb273`
- `config/drone_race_vq2_informed_dreamer.ini` — `5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_public_phase.py` — `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent.py` — `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`

The evaluator hashes this preregistration, checkpoint/report, all executable
sources, exact Git commit, runtime, and freshly compiled extension before the
first policy action.

## Safety

FlightSim packets, student updates, teacher labels, sealed N712 accesses, and
Submission authorization remain zero. No Windows simulator process or event
is touched. Even an admitted VG006 remains offline evidence only; later
perturbed screens, export/parity, and operator-run zero-command shadow gates
remain required before any bounded Training attempt.
