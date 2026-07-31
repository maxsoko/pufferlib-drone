# VQ2 VG031 fresh teacher-free screen preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline screen tagged
`vq2_vg031_vg028_variable_gate_recurrent_teacher_free_256`. Screen only
VG028's selected epoch-6 recurrent deterministic policy. Its checkpoint,
training report, and numerical-admission SHA-256 values are
`ec1206b5...`/`b51b7a9b...`/`3282c98f...`.

VG029 and VG030 independently qualified the candidate on paired fresh count-5
and count-11 fixtures. VG030 admission evidence SHA-256 is
`d44cacc132c5aa8fb16ed6f138cbf49730f3913c64ac4b784e0acf4a38bf639f`.
It records VG028 versus VG025 Gate-2 reach `25/32` versus `7/32`, Gate-3 reach
`4/32` versus `0/32`, and equal crash count `2`.

This is an offline native screen only. It authorizes neither FlightSim packets
nor a Windows shadow, Training run, or Submission selection.

## Fixed screen matrix

- Evaluate exactly 64 independent terminal episodes at each gate count
  `{5, 8, 11, 12}` for 256 total courses.
- Fixed fresh seeds are count 5 `429123`, count 8 `429126`, count 11 `429129`,
  and count 12 `429130`.
- Retain the native true aperture `0.75 m`, randomized course/plant/start
  contract, 64 Hz control step, and existing terminal horizon.
- Use four native CPU threads and two buffers. VG026 showed a source-locked
  4-vs-32-thread wall-time speedup of only `1.019054x`; retain the established
  four-thread path instead of adding another acceleration variable.
- Each count writes one source-locked count report. `state.json` commits before
  rollout and after every completed count; resume skips only source-identical
  completed counts. A terminal aggregate cannot be rerun unchanged.

## Policy and legality

- The plant receives only the deterministic mean of VG028's one recurrent
  four-channel PufferLib actor on the unchanged 4,119-value legal observation.
- Teacher blend/action, sampling, action clipping, analytic channel override,
  checkpoint switching, planner, fallback, total-count input, privileged
  state, and phase-dependent controller selection are absent.
- Public progress remains the causal held 4 Hz scalar encoded as
  `clamp(active_gate_index,0,16)/16`; action history and recurrent state
  advance once per native step.
- The `0.50 m` crossing-margin metric remains diagnostic because the real
  aperture is `0.75 m`. Crash and every transport/ordering fault remain hard.

## Admission

VG031 admits only if the aggregate has at least `231/256` ordered full-course
finishes and all 256 terminal episodes complete, with zero:

- crash or out-of-order event;
- non-finite or action-envelope violation;
- wire-rate or thrust-envelope violation;
- executed-action history error above `5e-5`;
- held public-phase off-tick change, decrease, or skip;
- raw phase-encoding error above `1e-6`; and
- teacher action, optimizer update, FlightSim packet, or sealed-test access.

Any failure terminally rejects VG031 and forbids an unchanged retry. Preserve
the per-count failure distribution. Rejection authorizes only a causally
distinct offline visited-state collection or repair. Admission authorizes only
source-locking the recurrent callable and command-free replay/parity work.

## Source identity and bootstrap

Before the first native step, bind the pushed Git commit, wrapper, generic
evaluator, runner, this preregistration, goal prompt, VG028 checkpoint/report/
admission, VG030 admission, native sources/config, compiled extension, runtime,
seeds, count order, and zero-authority safety fields.

Require a clean exact commit, preserved Vast environment, CUDA,
Clang/OpenMP, ccache, at least 32 CPUs, about 64 GiB RAM, 15 GiB free disk,
both native regression suites, a fresh SM89 float32 build, and the focused
evaluator test shard.

Frozen new source surface:

- `scripts/eval_vq2_vg031_variable_gate_recurrent_policy.py` — SHA-256
  `2930c82f6314500baf40b9c741c5357a889ef7def5cbbdd97e3d942d39be431a`
- `scripts/run_vq2_vg031_vast.sh` — SHA-256
  `ff27bca218a212a2bee3b2c82fa6e6c4087f47d47f3e79563066e4765a20e770`
- `tests/test_eval_vq2_vg031_variable_gate_recurrent_policy.py` — SHA-256
  `e59fd9d37acb63acabf99c34611c77c5d80607201e7be9b88e22cb4f0b1e687c`
- `scripts/eval_vq2_variable_gate_recurrent_policy.py` — SHA-256
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `pufferlib/vq2_recurrent_phase.py` — SHA-256
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `pufferlib/vq2_public_phase.py` — SHA-256
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `docs/vq2_vg030_paired_count11_diagnostic_admission_2026-07-31.json` —
  SHA-256
  `d44cacc132c5aa8fb16ed6f138cbf49730f3913c64ac4b784e0acf4a38bf639f`
- `docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md` — SHA-256
  `03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1`

## Safety

Privileged actor inputs, teacher plant actions, optimizer updates, FlightSim
packets, sealed-test accesses, shadow authority, Training authority, and
Submission authority are all zero. VQ2 Submission remains forbidden without a
new explicit user instruction.
