# VQ2-SF006 true-velocity oracle preregistration — 2026-07-28

Tag: `vq2_sf006_native_oracle_true_velocity_surface`

SF005 proved that changing `teacher_roll_rate_per_m_s` from `0.1` through
`0.7` changed no metric. The legacy teacher applies that gain only to an
optional gate-motion observation, which is invalid on the fixed static course.
The supposed PD oracle was proportional-only.

## Source-locked repair

Add `teacher_true_velocity_damping`, default `0`, to native training
infrastructure. When enabled, compute derivative feedback from true privileged
relative velocity:

```text
target_velocity_world - vehicle_velocity_world
  -> rotate into body frame
  -> existing bounded roll/thrust rate terms
```

Default zero must preserve all historical dynamics exactly. A native unit test
must prove the old absent-rate result with the flag off and the expected
braking command with it on. This flag never enters actor observations or a
deployment artifact.

## Fixed surface

Retain SF005's direct controller, `3.5 m/s` target, `0.35` position gain,
`60 s` horizon, six `0.75 m` gates, and fixed course/plant. Enable true-velocity
damping and screen rate gains `0.1, 0.3, 0.5, 0.7` on `16` common-seed episodes
each using seed `42006`.

Use SF005's fixed lexicographic selection. Only a cell with at least one valid
finish may seed a separately tagged independent `64`-episode screen. Otherwise
reject the surface and redesign the lateral reference. No label or replay may
be written.

## Safety boundary

Native only. No FlightSim traffic, consumed-test access, student update,
checkpoint admission, or Submission action is authorized.
