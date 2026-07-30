# VQ2 VG023 fresh teacher-free screen preregistration — 2026-07-30

## Evidence and one-run decision

Run exactly one resumable offline screen tagged
`vq2_vg023_variable_gate_recurrent_teacher_free_256`. Screen only VG022's
selected epoch-9 recurrent deterministic policy. Its checkpoint/report/
completed-state/admission SHA-256 values are `bd3f93d4...`/`23d28182...`/
`96c5e83e...`/`ac93a00d...`.

VG022 completed exact epochs 1 through 12 after bitwise zero-step acceleration
parity and selected epoch 9 at balanced validation MSE
`0.015374308139712155`. It is numerical evidence only; this fresh closed-loop
screen is the next authorized decision. No prior screen seed may be reused.

## Fixed screen matrix

- Evaluate exactly 64 independent terminal episodes at each gate count
  `{5, 8, 11, 12}` for 256 total courses.
- Fixed new seeds are count 5 `429101`, count 8 `429104`, count 11 `429107`,
  and count 12 `429108`.
- Retain the native true aperture `0.75 m`, randomized course/plant/start
  contract, 64 Hz control step, and existing terminal horizon.
- Each count writes one source-locked count report. `state.json` commits after
  every completed count and permits exact-identity resume only.

## Policy and legality

- The plant receives only the deterministic mean of VG022's one recurrent
  four-channel PufferLib actor on the unchanged 4,119-value legal observation.
- Teacher blend/action, sampling, action clipping, analytic channel override,
  checkpoint switching, planner, fallback, total-count input, privileged state,
  and phase-dependent controller selection are all absent.
- Public progress remains the causal held 4 Hz scalar encoded as
  `clamp(active_gate_index,0,16)/16`; executed-action history and recurrent
  state advance exactly once per native step.
- The `0.50 m` crossing-margin metric remains diagnostic because the real
  aperture is `0.75 m`. Crash and every transport/ordering fault remain hard.

## Admission

VG023 admits only if the aggregate has at least `231/256` ordered full-course
finishes and all 256 terminal episodes complete, with zero:

- crash or out-of-order event;
- non-finite or action-envelope violation;
- wire-rate or thrust-envelope violation;
- executed-action history error above `1e-7`;
- held public-phase off-tick change, decrease, or skip; and
- raw phase-encoding error above `1e-6`.

Any failure terminally rejects VG023. Preserve the per-count failure
distribution and never retry the same candidate/seeds unchanged. An admitted
screen may authorize only the next offline export/perturbation gate. A rejected
screen authorizes a new causally distinct visited-state DAgger or phase-local
variation after diagnosis.

## Source identity and bootstrap

Before the first native step, bind the pushed Git commit, wrapper, generic
evaluator, runner, this preregistration, VG022 checkpoint/report/admission,
native sources/config, compiled extension, runtime, seeds, count order, and
zero-authority safety fields. The Vast bootstrap requires a clean exact commit,
the preserved virtual environment, CUDA, at least 32 visible CPUs, about 64 GiB
RAM, 15 GiB free disk, Clang/OpenMP, and a fresh float32 native build. Both
native regression suites and the focused Python shard must pass before rollout.

## Frozen source surface

- `scripts/eval_vq2_variable_gate_recurrent_policy.py` —
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `scripts/eval_vq2_vg023_variable_gate_recurrent_policy.py` —
  `fdacf95f3743559629833712648539ae96c39f46418aff49eb68351520c72550`
- `scripts/run_vq2_vg023_vast.sh` —
  `b03974cb0eaa925c79344a0e7fdea68cf4d0d6b7caa15d006334809e2dff61f4`
- `tests/test_eval_vq2_vg023_variable_gate_recurrent_policy.py` —
  `c922b944ffdc4d5dbee6332fc8aeb14988768460f61e382c675d765fe3a2fd7d`
- `pufferlib/vq2_recurrent_phase.py` —
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md` —
  `052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89`

The source identity also binds this preregistration and exact VG022 evidence.

## Safety and next authority

This is offline native evaluation. FlightSim packets, Windows shadow, bounded
Training attempts, Submission selection, sealed N712 access, teacher plant
actions, and student updates are all zero. VG023 grants no live authority.
