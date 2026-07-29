# VQ2 C002R prefix-gain bracket result — 2026-07-28

## Decision

Reject a single coupled longitudinal/braking plant gain as the deployment
model. Stop this calibration branch; no interpolated C003 gain is authorized.

All three recovery runs produced complete, source-locked reports and passed
the legal selector/action-delivery checks. None met the joint N399 transition
criterion:

| Gain | Raw Gate-1 time | Held time | Exit `vx` | Decision |
|---:|---:|---:|---:|---|
| `10.0` | `3.109375 s` | `3.250000 s` | `5.920375 m/s` | timing matches; speed is `1.243375 m/s` high |
| `12.5` | `2.750000 s` | `2.750000 s` | `7.168493 m/s` | timing and speed too high |
| `15.0` | `2.515625 s` | `2.750000 s` | `7.917393 m/s` | timing and speed too high; later suffix crash |

The target was raw `3.194960355 s`, held approximately `3.281 s`, and
`4.677 m/s`. The lowest point already crosses from the slow side of the time
target to the fast side of the speed target, so the three preregistered points
do not contain an adjacent pair that brackets both targets. Interpolation would
violate the preregistration and cannot solve the underlying mismatch anyway.

No suffix outcome is an admission claim. Every run completed Gate 1, selected
whole policy actions, delivered those actions with zero recorded numerical
error, and sent zero FlightSim packets. No teacher action or policy update
occurred.

## Frozen reports

- gain `10.0`: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_c002r_gain10p0_exact1/report.json`
  - SHA-256 `deb0301428e7c781d5e981e298ed87d66f0aa7c65c5d12e99ec5cd9301055f6b`
- gain `12.5`: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_c002r_gain12p5_exact1/report.json`
  - SHA-256 `919c99272c4f84073bc1720781e927565d35cdadd421f6d2f3cc9d4b22b93489`
- gain `15.0`: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_c002r_gain15p0_exact1/report.json`
  - SHA-256 `1830fa874772226334e301ab5e2b3f391627fbb8b0851f7e3249aa4fb5c7b557`

## Next fixture

Use the gain-`10.0` legal visual prefix only as a timing-matched recurrent
warmup. At the held Gate-1 transition, continue from the existing N399
source-locked measured state (`3.281 s`, velocity/attitude/body rates, Gate-2
layout, and final prefix action) instead of pretending the training plant
identified that state. Preserve both Puffer recurrent states, the actual
selected-action history, and held progress. Apply the measured first Gate-2
camera range/association alias only to the legal visual input; it must not move
the physical gate or emit an action.
