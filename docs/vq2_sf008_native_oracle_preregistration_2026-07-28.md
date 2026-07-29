# VQ2-SF008 minimum-jerk segment-oracle preregistration — 2026-07-28

Tag: `vq2_sf008_native_oracle_minimum_jerk_64`

SF007 rejected speed-only tuning of the existing whole-course spline. Add one
default-off training-only reference law whose desired position, velocity, and
acceleration are continuous at every ordered gate.

## Fixed controller

For the segment ending at the active gate, let the segment start be the prior
gate center (or the nominal course start for Gate 1). Normalize current world
forward position to `s` in `[0,1]` and use the quintic minimum-jerk basis:

```text
h(s) = 10 s^3 - 15 s^4 + 6 s^5
```

Interpolate world lateral and vertical position from segment start to active
gate with `h`. Use its analytic first and second forward-position derivatives
as velocity and curvature feedforward. The basis has zero first and second
derivatives at both endpoints, so changing active gate preserves the reference
position, velocity, and acceleration instead of stepping to the next center.

Reuse the existing spline teacher's measured-plant conversion and fixed
position/velocity gains exactly. Retain the pitch speed controller at
`2.5 m/s`, yaw target zero, full teacher action intervention, `60 s` horizon,
six fixed synthetic gates of radius `0.75 m`, standard full-start noise, and
zero external actions. The new `teacher_segment_minimum_jerk` flag defaults to
zero and cannot change historical trajectories when disabled.

## Required checks

- Native unit tests prove the quintic endpoint/midpoint values and that the
  segment reference is continuous at a gate transition.
- Existing native regressions remain green with the flag off.
- Run exactly `64` episodes with seed `42008`.
- Write no label, replay, or checkpoint.

Any valid six-gate finish admits a separately tagged focused robustness/gain
diagnostic. Zero finishes rejects this fixed controller and requires analysis
of the recorded first failing gate before changing one structural element.
This 64-episode diagnostic is not the 4096-episode oracle admission.

## Safety boundary

Native only. No FlightSim traffic, consumed-test access, student update,
checkpoint admission, or Submission action is authorized.
