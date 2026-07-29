# VQ2-SF003 direct native-oracle preregistration — 2026-07-28

Tag: `vq2_sf003_native_oracle_direct_64`

VQ2-SF002 rejected the existing course-spline teacher at `0/64`: all episodes
missed laterally, with zero collision or timeout. VQ2-SF003 changes only the
privileged teacher's lateral/vertical target law. It retains SF002's course,
start jitter, speed controller, yaw target, aperture, seed, lifecycle, and
decision rule.

## Single mechanism change

Disable `teacher_course_spline` and use the already tested active-gate-relative
controller from the measured Gate-2 teacher:

```text
teacher_roll_per_m = 0.35
teacher_roll_rate_per_m_s = 0.10
teacher_thrust_bias = 0
teacher_thrust_per_m = 0.30
teacher_thrust_rate_per_m_s = 0.10
teacher_thrust_world_frame = 1
```

Pitch still targets `4.0 m/s` with gain `0.35` and scale `0.24`. Yaw remains
zero. Full teacher blend remains `1.0`. Every other fixed SF002 environment
value is identical.

The evaluator normalizes raw vector-log fields into the canonical `env/`
namespace before applying the decision rule. This corrects a report-only issue
observed in SF002; it does not change native dynamics or SF002's rejection.

## Fixed screen and decision

Run exactly `64` uninterrupted six-gate episodes, one per agent, at radius
`0.75 m` with seed `42002` and ordinary configured start-position jitter. No
gate, camera, or plant randomization is enabled. Zero external actions are
supplied and no labels or replay are written.

Retain only `64/64` ordered finishes with zero crash, timeout, missed-gate,
invalid, and out-of-order results. Otherwise reject and use the terminal-gate
and crossing-axis evidence to make one separately preregistered controller
change. A pass remains a smoke result, not the required 4096-episode randomized
oracle admission.

## Safety boundary

Native only. No FlightSim packet, N712 sealed-test access, checkpoint
admission, or Submission action is authorized.
