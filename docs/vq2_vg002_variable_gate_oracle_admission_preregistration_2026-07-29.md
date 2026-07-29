# VQ2-VG002 variable-gate oracle admission preregistration — 2026-07-29

Tag: `vq2_vg002_variable_gate_oracle_admission_2048`

VG002 is the blocking Stage-0 oracle admission from the variable-gate
solve-first prompt. It is an offline native screen only. It sends no FlightSim
packet, accesses no consumed sealed test, performs no student update, and
cannot authorize VQ2 Submission.

## Source-locked screen

- Use the float32 `drone_race_vision` native extension built from the VG001
  sources. Require `env_name == drone_race_vision` and
  `precision_bytes == 4` before execution.
- Evaluate four separate fixed-count randomized-course runs in this order:
  `{5, 8, 11, 12}`. Use `512` vector agents and exactly one completed episode
  per agent for every count, for `2,048` total episodes.
- Use deterministic seeds `429020`, `429021`, `429022`, and `429023` in count
  order. Each run uses the unchanged admitted SF009/SF011 alignment governor,
  target speed `2.0 m/s`, control timestep `1/64 s`, true aperture radius
  `0.75 m`, and a maximum episode duration of `360 s`.
- Reuse the existing randomized-course geometry streams. Enable only VG001's
  bounded course-validity check: minimum ordered forward gap `8 m`, maximum
  adjacent-center distance `40 m`, and at most `64` deterministic resampling
  attempts. Position bounds expand to `320 m` for the longer courses.
- Use `clamp(active_gate_index, 0, 16) / 16` for any native/public phase
  diagnostic. Gate count is not an observation.

The source-locked implementation SHA-256 values are:

- `ocean/drone_race/drone_race.c`:
  `9f0101d6262402c900ed544450ade101a3bceb6e9c305cde76fa5776e82ae2b0`
- `ocean/drone_race/drone_race.h`:
  `58615ed1d7dd4738d5fcf04d7cdf4b43a6a6cbab9dccc393c0d665820f7f5b06`
- `ocean/drone_race/binding.c`:
  `409ed4689f145b9a1f17c858535ffd8842fe9aaef5cf2e67c82945e51effb273`
- `pufferlib/vq2_public_phase.py`:
  `6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338`
- `pufferlib/vq2_informed.py`:
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `config/drone_race_vq2_informed_dreamer.ini`:
  `5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b`
- `scripts/eval_vq2_native_oracle.py`:
  `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`
- `scripts/eval_vq2_variable_gate_oracle.py`:
  `ec65b09c614b16240717266f32fd49221f7f7075b5540eb802c6008f2acd93c6`
- `scripts/run_vq2_vg002_vast.sh`:
  `cabb80a634f8e5f397bb460011d41757344eca651c1ef4e53c25b5d9b220e38a`
- Local float32 compiled extension used by the accepted VG001B smoke:
  `847d77eea021e83d13910d732186a2da9419629c938992231cbf337de981c46f`

The remote machine must clone the source commit produced after this
preregistration and build its own float32 extension. Its platform-specific
extension hash must be recorded in every count report; it is not expected to
match the local binary hash.

## Fixed admission rule

Every count must independently satisfy all of the following:

1. exactly `512` completed episodes;
2. success rate at least `0.99` (at least `507/512` finishes);
3. zero collisions, out-of-order crossings, crossing-margin violations, and
   action/wire-rate/thrust envelope violations;
4. the report attributes every episode to the requested gate count;
5. every ordered gate is sampled in at least the successful fraction of
   episodes and has mean radial crossing error at most `0.10 m`.

Stop the aggregate screen after the first rejected count. Do not tune the
oracle in response to a failure: the solve-first prompt requires repairing
course generation if the unchanged oracle cannot solve a longer course. A
passing aggregate requires all four count reports to pass. Reports are written
once under
`logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg002_variable_gate_oracle_admission`;
the evaluator refuses an existing directory unless `--resume` passes the
strict checks below.

The paid remote run must be resumable without relabeling or retrying completed
evidence. Before the first count, atomically publish `state.json` with the full
run contract, source hashes, extension hash/path, and count seeds. Atomically
publish one immutable count report after each completed count and update the
mutable state at count boundaries. `--resume` may execute only the first
missing count after validating byte-exact source identity, Git commit,
Python/Torch/CUDA/NumPy/pybind11 platform manifest, and every field of the
completed ordered prefix. It must refuse source or runtime drift, a gap or
extra count report, mismatched metrics/configuration, or an altered aggregate.
When state exists, the remote wrapper must skip apt, pip, tests, and extension
rebuild before resuming. A rejected count is terminal and is never rerun. The
aggregate report is also immutable.

## Safety boundary

This screen is native/offline and training-only. FlightSim remains frozen
after N522: send zero heartbeat, TIMESYNC, metadata, reset, arm/disarm, or
setpoint packets. Never select VQ2 Submission and never access N712's consumed
sealed test.
