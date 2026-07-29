# VQ2 VG003 variable-gate legal BC corpus preregistration — 2026-07-29

## Decision and dependency

Run exactly one source-locked offline collection tagged
`vq2_vg003_variable_gate_legal_bc_dataset_256`. VG002 is the blocking parent:
its admitted aggregate report SHA-256 is
`e649a20acd92a0c341426325dce281fc989a3cdbae8d634f3a36c14202b8bb27`
and it passes 512/512 randomized courses independently at gate counts 5, 8,
11, and 12 with zero collision. VG003 may produce training data only. It does
not authorize a student checkpoint, FlightSim activity, shadow, bounded run,
or Submission.

## Fixed collection contract

- Seed: `429030`.
- Vector agents and episodes: `256` and `256`, exactly one episode per fixed
  native instance.
- Gate count: exact-uniform across 5 through 12, 32 episodes of every count.
  Count is assigned at instance construction and never changes on reset.
- Episode horizon: 360 seconds (`23,040` native 64 Hz steps).
- Course and teacher: the admitted VG002 randomized-course profile and
  unchanged governed SF009 oracle, teacher blend exactly 1.0 for label
  materialization only, true aperture radius 0.75 m.
- Output:
  `logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg003_variable_gate_legal_bc_dataset_256`.
- Minimum admitted size: 1,500,000 causal transitions.

The stored actor observation has exactly 4,119 values: a 4,096-value soft-red
mask quantized to uint8, the frozen 22-value legal camera/IMU/actuator/action/
timing tail as float32, and one held public progress scalar as float32. The
progress value is sampled every 16 policy steps (4 Hz), held causally between
samples, and encoded as `clamp(active_gate_index, 0, 16) / 16`. No total gate
count, gate pose, vehicle pose, teacher state, selected contour, course
coordinate, detector geometry, or other native privileged value is written.

The label for observation `O_t` is the complete four-value normalized action
that actually drove the plant, recovered from the newest executed-action
history slot of `O_(t+1)`. The older history slots must equal the corresponding
shifted history from `O_t` within `1e-7`. Every episode must occupy one
contiguous time-major valid prefix and contain exactly one terminal marker in
its last valid row.

## Admission and rejection

Admission requires all of the following:

- 256/256 successful, valid, ordered full-course completions; zero crash,
  timeout, missed gate, out-of-order event, or action/wire/thrust/crossing
  envelope violation;
- exact 1/8 episode and success mass for each count 5 through 12 and zero mass
  elsewhere; mean gates passed 8.5;
- every applicable ordered gate sampled with mean crossing radial at most
  0.10 m;
- at least 1,500,000 labels, exact label/valid-row equality, one contiguous
  terminal-valid prefix per episode, and action-history shift error at most
  `1e-7`;
- public phase changes only on 4 Hz ticks, never decreases, contains exactly
  1,920 observed nonterminal index increments, and every raw value is an exact
  integer divided by 16 within `1e-6`;
- metadata and every array are finalized with shape, dtype, byte count, and
  SHA-256 manifest entries; reported stored privileged and total-count values
  are both zero.

An admission failure writes terminal state `rejected` and forbids an unchanged
retry. A process interruption leaves state `collecting`; `--resume` is allowed
only when the complete source hash set, Git commit, compiled extension path and
hash, runtime manifest, seed, dimensions, and safety contract match exactly.
Such a resume deletes only this tag's partial output/staging and recollects
from the fixed seed. Once any state exists, the Vast wrapper must not install,
test, or rebuild before resume.

## Frozen executable sources

- `ocean/drone_race/drone_race.c` — `523aa40a4578e0ec4b8808686faa5ba2d58d1b681091d49425456c1b39b1657b`
- `ocean/drone_race/drone_race.h` — `58615ed1d7dd4738d5fcf04d7cdf4b43a6a6cbab9dccc393c0d665820f7f5b06`
- `ocean/drone_race/binding.c` — `409ed4689f145b9a1f17c858535ffd8842fe9aaef5cf2e67c82945e51effb273`
- `pufferlib/torch_pufferl.py` — `705515235d386646bc90945f5f44d6ba435798a077e53f06cd301cbf7d91dd81`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_public_phase.py` — `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `config/drone_race_vq2_informed_dreamer.ini` — `5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b`
- `scripts/eval_vq2_native_oracle.py` — `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`
- `scripts/eval_vq2_variable_gate_oracle.py` — `ec65b09c614b16240717266f32fd49221f7f7075b5540eb802c6008f2acd93c6`
- `scripts/collect_vq2_oracle_bc_dataset.py` — `84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868`
- `scripts/run_vq2_vg003_vast.sh` — `3392af1d4eceabc417fa575f807d269b82a36ff351d7dcd818257ee75efe4499`
- `scripts/collect_vq2_variable_gate_oracle_bc_dataset.py` — `c8dac1fb70aeb360d8cc65f07f549a3c218f94ccad74a8bb60100c3196a6e527`

The preregistration, exact Git commit, newly compiled float32 extension, and
runtime package/platform versions are hashed into mutable run state before the
first episode. Remote execution is limited to the already contracted Vast
instance `46201898`; evidence is synced back before the instance is stopped.

## Safety

This is native, offline, FlightSim-independent work. FlightSim packets,
sealed-test accesses, student updates, and student checkpoints must all remain
zero. N712 is never opened. The VQ2 simulator and Submission event remain
untouched.
