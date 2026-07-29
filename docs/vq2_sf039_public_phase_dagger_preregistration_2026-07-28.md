# VQ2-SF039 public-phase DAgger dataset — 2026-07-28

Tag: `vq2_sf039_public_phase_dagger_512`

The public-phase amendment adds one legal actor value: official
`active_gate_index / 6`, sampled and held at the measured `4 Hz` status rate.
This dataset recreates the SF033 state distribution on fresh seed `42039` for
`512` full-start randomized six-gate episodes at the true `0.75 m` aperture.

SF033 emits every executed CTBR action. The admitted SF016 oracle supplies only
the four-channel label. Store the existing `4096` mask values, `22` legal
sensor/history/timing values, and one held phase scalar. In native collection,
source that scalar from `current_gate / num_gates` only at 16-step boundaries
on the fixed `64 Hz` policy clock; this is the exact simulator equivalent of
the public official field, not a pose/geometry/state input. Persist no other
value from the 34-value native training-only tail.

Use a maximum of `2048` steps. Require one terminal per episode, no timeout,
out-of-order, action/rate/thrust fault, exact executed-action replay, monotonic
phase in `[0,1]`, phase changes only on status ticks, exact source hashes, and
zero stored training-only privileged actor values. Retain crash/miss terminals
as DAgger states.

This writes data only. It cannot admit a policy or authorize a screen. Send
zero FlightSim packets, do not access N712, and do not shadow, reset, arm,
setpoint, run a bounded attempt, or select Submission.

