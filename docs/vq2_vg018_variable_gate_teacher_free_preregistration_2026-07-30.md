# VQ2 VG018 teacher-free variable-course screen preregistration — 2026-07-30

## Candidate and one-screen decision

Screen exactly one checkpoint: admitted VG017 epoch 11, SHA-256
`6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8`.
Its report SHA-256 is
`476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b`;
independent admission evidence SHA-256 is
`df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6`.
VG017 completed all 12 fixed epochs and passed numerical admission with exact
four-source balance and `3.0x` transition exposure.

Run exactly one resumable aggregate tagged
`vq2_vg018_variable_gate_recurrent_teacher_free_256`. This is deterministic
native closed-loop inference, not training. No checkpoint, action transform,
or hyperparameter may be selected from its outcome. Admission or rejection is
terminal for this tag and forbids an unchanged retry.

## Fresh held-out course matrix

- Fixed gate counts `5`, `8`, `11`, and `12`, independently and in that order.
- 64 vector agents and one full-start terminal episode per count; 256 fresh
  randomized courses total.
- Seeds: count 5 -> `429085`, count 8 -> `429088`, count 11 -> `429091`, and
  count 12 -> `429092`. These are disjoint from all VG006/VG011/VG015 screen
  seeds and every VG017 input/training seed.
- True aperture radius `0.75 m`, 360-second maximum episode horizon, and a
  fresh float32 native binding compiled from the source-locked commit.

Every plant action is the deterministic mean of the one VG017 recurrent
four-output actor. Teacher action blend, analytic control, sampling, clipping,
phase switch, scheduler, and fallback are absent. The actor receives exactly
the legal 4,119-value ABI: camera/IMU/actuator/action/timing history plus one
causal 4 Hz held public index encoded `/16`. Total gate count and native state
remain outside the actor input.

## Completion, safety, and admission

Each count must publish 64 terminal logs and finite complete actions within
`[-1,1]`. Executed-action history error must be at most `5e-5`; public phase
must have zero off-tick change, decrease, or skip and raw encoding error at
most `1e-6`. Gate-count mass must match the requested count.

Aggregate admission requires at least 231/256 ordered full-course successes
and, across every count, zero crash, out-of-order, action/wire/thrust envelope,
non-finite action, or phase/action-history transport fault. Teacher blend must
be exactly zero.

`crossing_margin_violation` remains diagnostic-only. Its `0.50 m` diagnostic
radius does not change the true `0.75 m` aperture or weaken any hard admission
requirement. Rejection routes only to a new evidence-driven offline repair and
never authorizes an unchanged retry or live activity.

## Source identity and remote execution

Before count 5, `state.json` locks the Git commit, evaluator/wrapper/runner,
this preregistration, candidate/report/admission, native sources/config,
compiled extension, runtime, counts, seeds, admission contract, and safety
fields. Completed counts are written once and hashed. Resume may execute only
missing counts under exact identity; once state exists, the runner performs no
build, install, test, or other runtime mutation.

Initial bootstrap requires exact candidate hashes, supported CUDA, at least 32
visible CPUs, approximately 64 GiB RAM, 15 GiB free disk, Clang/OpenMP, and
ccache. Both native regression suites, a fresh float32 CUDA build/ABI check,
and focused/adjacent tests must pass before the first action.

## Frozen source surface

- `scripts/eval_vq2_variable_gate_recurrent_policy.py` —
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `scripts/eval_vq2_vg018_variable_gate_recurrent_policy.py` —
  `2eaacd3ceb8b5935535dc3633c2e8e051331321438f62e97041b6ee3d4985108`
- `scripts/run_vq2_vg018_vast.sh` —
  `f02586e862e931af6ed8425f64e8226cb3c98b9af747a483a10b965aac7e3a0b`
- `tests/test_eval_vq2_variable_gate_recurrent_policy.py` —
  `203114c354f3f6be9eae42a6d8a3e953a13a8624d3e91430e3af4fbe9003f473`
- `tests/test_eval_vq2_vg018_variable_gate_recurrent_policy.py` —
  `0a6e6d4b99d761fd6cec14f1428ca7373d6e044127c4cd593ef6b7c519f585ee`

The source identity also binds the unchanged oracle/evaluator, native course,
observation/phase, recurrent actor, config, this preregistration, and exact
VG017 artifacts before count 5.

## Safety and next authority

This is offline native inference. Teacher labels, student updates, FlightSim
packets, sealed N712 accesses, Windows shadow, Training, and Submission
authorization are zero. Even admission authorizes only later offline promotion
gates; it is not live-flight evidence.
