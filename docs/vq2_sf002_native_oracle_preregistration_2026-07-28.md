# VQ2-SF002 native-oracle baseline preregistration — 2026-07-28

Tag: `vq2_sf002_native_oracle_baseline_64`

This is the first diagnostic after VQ2-SF001. It tests whether the smallest
existing privileged attitude teacher can already solve an uninterrupted native
six-gate course at the true `0.75 m` aperture. It writes no labels and cannot
train or admit a student policy.

## Fixed hypothesis

The native attitude-setpoint teacher should be sufficient when it owns all four
channels:

- speed-controlled pitch targets `4.0 m/s`;
- the existing course spline supplies roll and collective thrust from true
  state and the fixed six-gate geometry;
- yaw is stabilized at zero; and
- the teacher action blend is exactly `1.0` from the first step.

Zero external actions are supplied. The blend is training-only execution of
the privileged teacher, not a deployable policy or admission screen.

## Fixed diagnostic screen

- backend: compiled `drone_race_vision` native environment only;
- configuration: `drone_race_vq2_informed_dreamer`;
- `64` agents, exactly one completed episode per agent;
- six gates, every radius exactly `0.75 m`;
- uninterrupted full-course starts only;
- standard reset-position jitter (`+-0.5 m` horizontal and `+-0.25 m`
  vertical) retained from the configuration;
- gate geometry, camera, and plant randomization disabled;
- gate-local, mixed, and segment starts disabled;
- maximum episode duration `40 s`, `dt=0.015625 s`, strict missed-gate and
  collision termination retained;
- seed `42002`; and
- no checkpoint, replay, teacher-label, or dataset output.

Teacher parameters are fixed at:

```text
teacher_action_blend = 1
teacher_course_spline = 1
teacher_from_gate_index = 0
teacher_pitch_speed_control = 1
teacher_pitch_speed_target_m_s = 4.0
teacher_pitch_speed_gain = 0.35
teacher_pitch_speed_scale = 0.24
teacher_roll_from_gate_index = 0
teacher_roll_until_gate_index = 6
teacher_thrust_from_gate_index = 0
teacher_yaw_control = 1
teacher_yaw_from_gate_index = 0
teacher_yaw_action = 0
```

## Fixed decision rule

The baseline is retained only if it produces exactly `64/64` ordered six-gate
finishes with zero crash, timeout, invalid, and out-of-order episodes. Any
failure rejects these parameters for label generation. Diagnose the earliest
terminal gate and repair the same inspectable state-feedback controller before
running another uniquely tagged cell.

Even a `64/64` result is only a smoke gate. Oracle admission still requires the
canonical `4096`-episode held-out randomized full-course screen with at least
`99.9%` completion and zero collision/out-of-order events. Do not collect
BC/DAgger labels before that admission.

## Safety boundary

- Native execution only; no UDP socket or FlightSim path is used.
- No post-N522 FlightSim packet is authorized.
- Never touch N712's consumed sealed test.
- VQ2 Submission remains forbidden.
