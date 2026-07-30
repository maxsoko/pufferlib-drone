# VQ2 VG011 teacher-free variable-course screen preregistration — 2026-07-30

## Candidate and one-screen decision

Screen exactly one checkpoint: admitted VG010 epoch 12, SHA-256
`a093475371a94bc284310bbe64609fe515a35e50be0e68a7fdfb781b55f03822`.
Its training report SHA-256 is
`dfe943b593f99c4ccfa53b04290a30cd3a58ea9fefd11d9489786e064b8734a5`;
independent admission evidence SHA-256 is
`c0fbfc72c0c609b137d7240aa8347baa6f59926e2c26e137144d363e9e820850`.
VG010 completed all 12 fixed epochs, selected epoch 12, and passed numerical
admission with exact equal-source balance and `3.0x` transition exposure.

Run exactly one resumable aggregate tagged
`vq2_vg011_variable_gate_recurrent_teacher_free_256`. This is a deterministic
native closed-loop screen, not training. No checkpoint, action transform, or
hyperparameter may be selected from its outcomes. A completed admission or
rejection is terminal for this tag and cannot be retried unchanged.

## Fresh held-out course matrix

- Fixed gate counts `5`, `8`, `11`, and `12`, independently and in that order.
- 64 vector agents and exactly one full-start terminal episode per count; 256
  new randomized courses total.
- Seeds: count 5 -> `429056`, count 8 -> `429059`, count 11 -> `429062`, and
  count 12 -> `429063`. They are disjoint from VG003, VG005, VG006, VG009, and
  VG010 seeds.
- Admitted reachable randomized-course geometry, true aperture radius
  `0.75 m`, 360-second maximum episode horizon, and a fresh float32 native
  binding compiled from the source-locked commit.

Every plant action is the deterministic mean of the one VG010 recurrent
four-output actor. Teacher action blend, teacher spline/minimum-jerk/alignment
paths, analytic control, sampling, phase switch, scheduler, fallback, and
post-policy clipping are absent. The actor receives exactly 4,119 values: the
4,118-value legal camera/IMU/actuator/action/timing observation plus one causal
4 Hz held public index encoded as `clamp(active_gate_index,0,16)/16`. Total
gate count and every native state value remain outside the actor input.

## Completion, safety, and admission

Each count must publish 64 terminal logs and finite complete actions inside
`[-1,1]`. Executed-action history error must be at most `5e-5`; public phase
must have zero off-tick change, decrease, or skip and raw `/16` encoding error
at most `1e-6`. Gate-count mass must match the requested fixed count.

Aggregate admission requires at least 231/256 ordered full-course successes
(`>=90%`) and all of the following across every count:

- zero crash and out-of-order outcome;
- zero action, wire-rate, and thrust-envelope violation;
- zero non-finite action and exact teacher blend `0.0`; and
- all action-history and held-phase transport checks above.

`crossing_margin_violation` is recorded but is not an admission predicate.
The native diagnostic marks accepted crossings above `0.50 m` radial while
the actual aperture is `0.75 m`; demoting it does not enlarge the aperture or
weaken any crash/ordering/transport requirement. The aggregate report records
both the violation rate and `crossing_margin_admission_predicate=false`.

A rejection localizes success, maximum public index, crash/miss/timeout, gate
crossing, and action/phase metrics by count. It routes to a new source-balanced
DAgger round on VG010-visited states; it never authorizes an unchanged screen
retry or live activity.

## Source identity, resume, and remote execution

Before count 5, `state.json` locks the Git commit, evaluator/wrapper/runner,
preregistration, candidate/report/admission evidence, native sources and
config, compiled extension, runtime, counts, seeds, admission contract, and
safety fields. Each completed count is written once and hashed before state
advances. An interruption may run only missing counts with exact identity.
Once state exists, the runner performs no install, build, test, or other
runtime mutation.

The initial bootstrap requires exact candidate hashes, CUDA on a supported
GPU, at least 32 visible CPUs, approximately 64 GiB RAM, 15 GiB free disk,
`clang`, OpenMP headers, and `ccache`. It passes both native regression suites,
a fresh float32 CUDA build/ABI check, and focused/adjacent Python tests before
the first policy action. CPU test reductions alone use 16 OpenMP/MKL threads;
the CUDA screen remains at runtime defaults.

## Frozen source surface

- `scripts/eval_vq2_variable_gate_recurrent_policy.py` —
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- `scripts/eval_vq2_vg011_variable_gate_recurrent_policy.py` —
  `19b97cbc3c6b49857d2248dbc3a594defbd1d01b7f97bff207118dc1097e1422`
- `scripts/run_vq2_vg011_vast.sh` —
  `39fa599411c7e566f7417e544a1c3a0b4f393a4de73c5fe7c6ba7d4d36355f15`
- `tests/test_eval_vq2_variable_gate_recurrent_policy.py` —
  `203114c354f3f6be9eae42a6d8a3e953a13a8624d3e91430e3af4fbe9003f473`
- `tests/test_eval_vq2_vg011_variable_gate_recurrent_policy.py` —
  `fbae6b53989c9deeddacd734b76f8e33d5a01906a164b514cadb2b50f4d0a55d`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` —
  `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`
- `scripts/eval_vq2_variable_gate_oracle.py` —
  `ec65b09c614b16240717266f32fd49221f7f7075b5540eb802c6008f2acd93c6`
- `scripts/eval_vq2_recurrent_policy.py` —
  `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa`
- `ocean/drone_race/drone_race.c` —
  `523aa40a4578e0ec4b8808686faa5ba2d58d1b681091d49425456c1b39b1657b`
- `ocean/drone_race/drone_race.h` —
  `58615ed1d7dd4738d5fcf04d7cdf4b43a6a6cbab9dccc393c0d665820f7f5b06`
- `ocean/drone_race/binding.c` —
  `409ed4689f145b9a1f17c858535ffd8842fe9aaef5cf2e67c82945e51effb273`
- `config/drone_race_vq2_informed_dreamer.ini` —
  `5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b`
- `pufferlib/vq2_informed.py` —
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_public_phase.py` —
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent.py` —
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` —
  `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`

The source identity also binds this preregistration and exact VG010 artifacts
before count 5 begins.

## Safety and next authority

This is offline native inference. Teacher labels, student updates, FlightSim
packets, sealed N712 accesses, Windows shadow, Training, and Submission
authorization are zero. Even admission authorizes only the next offline
promotion gates; it is not live-flight evidence.
