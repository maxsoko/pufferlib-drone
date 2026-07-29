# VQ2-SF015 recurrent BC teacher-free native preregistration — 2026-07-28

Tag: `vq2_sf015_recurrent_bc_teacher_free_64`

SF014 source-locks the first compact recurrent actor. Its selected epoch-8
checkpoint has validation weighted MSE `0.0002803708407308082`; per-channel
pitch/roll/thrust/yaw MSE values are
`0.000344976/0.001120761/0.000121277/0.000011751`. The checkpoint contains one
legal CNN, one GRU, and one joint four-action head with zero actor privilege.

SF015 is one bounded native-only closed-loop diagnostic. It writes no labels
and performs no update.

## Fixed screen

- Checkpoint SHA-256:
  `00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a`.
- Run exactly `64` disjoint randomized full six-gate courses, one per agent,
  seed `42015`, fixed `0.75 m` aperture, fixed plant, `180 s` horizon, and the
  SF011/SF012 course-position/scale randomization.
- Set teacher action blend to exactly zero. Disable spline, minimum-jerk, and
  alignment-governor switches and set teacher-action reward to zero.
- On every active step, pass only the leading `4118` native values to the actor,
  advance its one recurrent state, use its deterministic joint-head mean for
  all four plant actions, and never clip, blend, select, or override a channel.
- Freeze recurrent state after an agent terminates. Begin state at zero only at
  the genuine course start.
- Stop after all 64 episodes terminate or after the fixed native horizon.

Run once:

```bash
.venv/bin/python scripts/eval_vq2_recurrent_policy.py
```

Frozen SHA-256 values:

- evaluator: `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa`;
- evaluator tests:
  `e0dc58829b1e21032ec4b7374b2980e2edaf8e1ac01c48a497b6b94343be5b2e`;
- actor: `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- legal ABI: `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`;
- shared native evaluator:
  `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`;
- SF014 training report:
  `f97eedb929aa18b8a4f9654f18b82b107a3beb3e41988c40a77c2cd421a7499d`.

The actor/evaluator/native/legal contract suite passes `27/27`. A separate
checkpoint audit reproduces sequence versus causal step replay within
`4.76837158203125e-7` for actions and recurrent state.

## Diagnostic outcome

A complete pass requires `64/64` ordered six-gate finishes, zero collision,
miss, out-of-order, timeout, crossing-margin, action, wire-rate, and thrust
violation, and finite bounded actions throughout. A pass only authorizes a
larger native milestone screen; it does not authorize FlightSim.

Any incomplete prefix is expected BC distribution-shift evidence. Record gate
progress and failure modes, reject SF014 for direct promotion, and use a new
tag for student-driven DAgger collection. Never add teacher blend to this
screen or retry it unchanged.

## Safety boundary

Native offline screen only. Write zero labels and checkpoints, perform zero
updates, send zero FlightSim packets, do not touch N712, and do not shadow,
reset, arm, setpoint, run a bounded attempt, or select Submission. VQ2
Submission remains forbidden.
