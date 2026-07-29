# VQ2 C002 N294 prefix-gain bracket preregistration — 2026-07-28

## Purpose

Calibrate only the native training plant's coupled longitudinal acceleration
and braking authority so the frozen N294 prefix reproduces the retained N399
Gate-1 timing and transition speed. This is a command-free offline diagnostic,
not policy training or FlightSim authorization.

C001 is rejected as a suffix-training environment: its legal composite wiring
passed, but it delivered the suffix after `6.750 s` at `2.078475237 m/s`,
versus official Gate 1 at `3.194960355 s` and the retained transition speed of
approximately `4.677 m/s`.

## Frozen inputs

- evaluator: `scripts/eval_vq2_n294_visual_suffix_composite.py`
  - SHA-256 `ed70ed37669bc569d4b3a6b3e862bb3c85aa57a59435b9ae42b7175c708ebf22`
- composite helper: `pufferlib/vq2_n294_visual_composite.py`
  - SHA-256 `3136d518441ce54404f1fce0884fc4346e49455be5d27e6af8e57e9d2854474f`
- C001 rejection: `docs/vq2_c001_n294_visual_suffix_exact_result_2026-07-28.md`
  - SHA-256 `3dc49c330296eac83675d0156c0aaafceac8739074ef30a4cd09e2c70875b2c3`
- retained N399 diagnosis: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n401_n399_failure_diagnosis/report.json`
  - SHA-256 `59813343395ca65edb7469f6dbd63cfdda500f3d8a9b6746512aa2d771aea4b2`
- N294 and SF066 remain locked to the hashes in the execution prompt and
  evaluator.

Focused helper/evaluator tests pass `9/9` before this bracket.

## Fixed bracket

Run exactly one deterministic agent at each of these three coupled values:

| Tag | `sitl_horizontal_accel_scale` | `sitl_braking_accel_scale` |
|---|---:|---:|
| `vq2_c002a_prefix_gain10p0_exact1` | `10.0` | `10.0` |
| `vq2_c002b_prefix_gain12p5_exact1` | `12.5` | `12.5` |
| `vq2_c002c_prefix_gain15p0_exact1` | `15.0` | `15.0` |

Everything else is frozen: policy weights, recurrent initialization, selected
previous action, held progress cadence, camera model, IMU/state adapter,
course, starting state, `64 Hz` step, and continuous `14 s` clock. N294 owns
the complete action until the held index transition. SF066 continues to warm,
but no suffix result from this bracket is an admission claim.

Each run uses seed `43002`, CUDA deterministic-mean inference, one episode,
and a unique non-overwriting output directory. No result may be rerun
unchanged.

## Evaluation and bound

For every Gate-1-passing point, record:

- first raw and held Gate-1 observation step/time;
- Gate-1 crossing error and exit `vx/vy/vz` from native metrics;
- action-delivery error, collision/miss/timeout status, and source hashes.

The measured targets are raw Gate-1 time `3.194960355 s`, held transition
sample near `3.281 s`, and forward transition speed `4.677 m/s`. A point is an
exact prefix-match candidate only if:

- Gate 1 passes inside the `0.75 m` aperture without collision or invalid
  state;
- raw time is within `0.125 s` of `3.194960355 s`;
- held time is within `0.125 s` of `3.281 s`;
- exit `vx` is within `0.25 m/s` of `4.677 m/s`; and
- action delivery remains finite, in-envelope, and within `5e-5`.

If multiple fixed points qualify, retain the lowest normalized sum of raw-time
and exit-speed errors. If none qualifies but two adjacent successful points
monotonically bracket both targets, permit one and only one linearly
interpolated gain in a separately tagged C003 run. Otherwise reject the
one-scalar model and source-lock the residual; do not sweep, modify a policy,
or collect DAgger labels.

## Safety

This bracket sends zero UDP/MAVLink packets, touches neither VQ2 Training nor
Submission, executes zero teacher action, and updates zero policy parameter.
FlightSim commands and suffix training remain frozen after this diagnostic.
