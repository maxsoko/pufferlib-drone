# VQ2-SF009 alignment-governed oracle preregistration — 2026-07-28

Tag: `vq2_sf009_native_oracle_alignment_governed_64`

SF008's minimum-jerk path is smooth but not robust to the short noisy first
approach or plant lag. Replace trajectory following with one memoryless,
phase-agnostic active-gate controller whose forward speed is governed by
cross-track alignment. This is the simplest solve-first navigation law: align,
then go.

## Fixed controller

The default-off `teacher_alignment_governor` uses privileged world-frame
active-gate relative position and vehicle velocity only in native training:

```text
radial_error = hypot(relative_y, relative_z)
alignment = clamp(1 - radial_error / 1.0 m, 0, 1)
target_vx = 0.2 + alignment * (2.0 - 0.2) m/s
pitch = clamp(0.45 * (vx - target_vx), -0.8, 0.8)
ay = 0.4 * relative_y - 1.2 * vy
az = 0.8 * relative_z - 1.5 * vz
```

Convert `ay/az` to coupled roll and collective thrust with the same measured
plant force law used by the spline teacher, including drag, bank, pitch, and
gravity compensation. Hold yaw target at zero. No gate-specific gain, course
spline, next-gate lookahead, phase state, stored plan, or prescribed path is
allowed. The resulting complete normalized action remains clipped to the
deployment envelope.

Use six fixed synthetic gates at true radius `0.75 m`, standard full-start
noise, a `150 s` horizon, full teacher intervention, zero external actions,
exactly `64` episodes, and seed `42009`. Native tests must prove centered-speed
equilibrium, braking on large radial error, correct roll sign, finite bounded
thrust, and historical default-off behavior. Existing native regressions must
remain green.

Any valid six-gate finish admits one independent robustness diagnosis. Zero
finishes rejects the fixed law and requires failure localization before any
gain change. This diagnostic writes no action label, replay, or checkpoint and
is not the 4096-episode oracle admission.

## Safety boundary

Native only. No FlightSim traffic, consumed-test access, student update,
checkpoint admission, or Submission action is authorized.
