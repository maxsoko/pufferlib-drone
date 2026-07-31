# VQ2 VG037 bounded-thread teacher-free screen preregistration — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline screen tagged
`vq2_vg037_vg033_variable_gate_recurrent_teacher_free_256`. Screen only
VG033's selected epoch-6 deterministic recurrent policy. Its checkpoint,
training report, and numerical-admission SHA-256 values remain
`56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`.

VG034 and VG035 qualified the candidate on independent paired count-5 and
count-11 fixtures. VG036 produced no count report and no policy result: it was
stopped after its final runner command omitted the global four-thread limits
used by the paired diagnostics, reproducing VG031's machine-wide PyTorch
oversubscription. The immutable VG036 abort evidence SHA-256 is
`c5dc3287c2891c746de4eadd55df8648df7762b45e2655baeff623a29af7c27e`.

This is a fresh-seed infrastructure repair, not a retry of any completed
fixture. It authorizes neither FlightSim packets nor a Windows shadow,
Training run, or Submission selection.

## Fixed screen matrix

- Evaluate exactly 64 independent terminal episodes at each gate count
  `{5, 8, 11, 12}` for 256 total courses.
- Fixed fresh seeds are count 5 `429143`, count 8 `429146`, count 11
  `429149`, and count 12 `429150`.
- Retain the native true aperture `0.75 m`, randomized course/plant/start
  contract, 64 Hz control step, and existing 2,560-step terminal horizon.
- Retain four native vector threads and two buffers. Bind
  `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=1` before every evaluator process,
  including first run and resume. This matches VG034/VG035 and affects only
  host bookkeeping parallelism; policy inputs, recurrent states, actions,
  native steps, fixtures, and admission arithmetic are unchanged.
- Each count writes one source-locked count report. `state.json` commits before
  rollout and after every completed count; resume skips only source-identical
  completed counts. A terminal aggregate cannot be rerun unchanged.
- After each completed count, stop without running later counts if a hard
  admission fault exists or if even perfect remaining episodes cannot reach
  `231/256` finishes. Preserve that prefix as terminal rejection evidence.

## Policy and legality

- The plant receives only the deterministic mean of VG033's one recurrent
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

VG037 admits only if the aggregate has at least `231/256` ordered full-course
finishes and all 256 terminal episodes complete, with zero:

- crash or out-of-order event;
- non-finite or action-envelope violation;
- wire-rate or thrust-envelope violation;
- executed-action history error above `5e-5`;
- held public-phase off-tick change, decrease, or skip;
- raw phase-encoding error above `1e-6`; and
- teacher action, optimizer update, FlightSim packet, or sealed-test access.

Any failure terminally rejects VG037 and forbids an unchanged retry. Preserve
the per-count failure distribution. Rejection authorizes only a causally
distinct offline repair. Admission authorizes only source-locking the recurrent
callable and command-free replay/parity work.

## Source identity and bootstrap

Before the first native step, bind the pushed Git commit, VG037 wrapper,
VG036 base wrapper, generic evaluator, runner, this preregistration, goal
prompt, VG033 checkpoint/report/admission, VG035 admission, VG036 abort,
native sources/config, compiled extension, runtime, seeds, count order, thread
limits, and zero-authority safety fields.

Require a clean exact commit, preserved Vast environment, CUDA,
Clang/OpenMP, ccache, at least 32 CPUs, about 64 GiB RAM, 15 GiB free disk,
both native regression suites, a fresh SM89 float32 build, and the focused
evaluator test shard.

Frozen new source surface:

- `scripts/eval_vq2_vg037_variable_gate_recurrent_policy.py` — SHA-256
  `5c6f927804e752ffaf563c97916baba009d58c2d98d5cf053564eda2fbfeefba`
- `scripts/run_vq2_vg037_vast.sh` — SHA-256
  `f86f65a453401de383b0f95ee104955deb54521fbb4256d98eec9dcd863bf94a`
- `tests/test_eval_vq2_vg037_variable_gate_recurrent_policy.py` — SHA-256
  `d75ecf54fb24d551a5e51e0e456c4e29bf5151859f9b04e065be1be6f2931074`
- `docs/vq2_vg036_infrastructure_abort_2026-07-31.json` — SHA-256
  `c5dc3287c2891c746de4eadd55df8648df7762b45e2655baeff623a29af7c27e`
- `scripts/eval_vq2_variable_gate_recurrent_policy.py` — SHA-256
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `pufferlib/vq2_recurrent_phase.py` — SHA-256
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `pufferlib/vq2_public_phase.py` — SHA-256
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `docs/vq2_vg035_paired_count11_diagnostic_admission_2026-07-31.json` —
  SHA-256
  `7c7676b72429157c8434761bf39ec790eb00a53abb76b4733128b980d8d95d98`
- `docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md` — SHA-256
  `03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1`

## Safety

Privileged actor inputs, teacher plant actions, optimizer updates, FlightSim
packets, sealed-test accesses, shadow authority, Training authority, and
Submission authority are all zero. VQ2 Submission remains forbidden without a
new explicit user instruction.
