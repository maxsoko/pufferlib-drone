# VQ2-SF023 SF022 teacher-free full-course screen — 2026-07-28

Tag: `vq2_sf023_recurrent_dagger_broad_teacher_free_512`

SF022 selected epoch `3`. Checkpoint/report SHA-256 values are
`57949ba391ffdc4d1e0dda5aa352fbb7b6dcb2a51b1f7742018926f6c6b2d903` /
`2cfc253ef13dc4a438a784df595a52947aaa0bcdb42320a6069bebdfa85f2f66`.
It passes every fixed numerical gate: SF012 weighted MSE `0.00101095`, SF021
weighted MSE `0.00354337`, and SF021 pitch/roll/thrust/yaw MSE
`0.00310527/0.00849585/0.00326884/0.00012708`.

SF023 is the first closed-loop test of SF022. Numerical imitation scores do not
count as course progress.

## Fixed screen

- Run `512` fresh randomized six-gate courses, one per agent, seed `42023`,
  fixed `0.75 m` aperture/plant and `180 s` episode horizon.
- Start the recurrent state only at the genuine course reset and advance it on
  every legal observation while active.
- SF022's deterministic joint four-channel mean emits every plant action.
- Teacher blend, teacher controller switches, labels, updates, privileged actor
  inputs, analytic action logic, clipping, override, and fallback remain zero.
- Record native progress and safety metrics, complete action statistics,
  inference rate, exact lineage hashes, and zero FlightSim traffic.

Run once:

```bash
.venv/bin/python scripts/eval_vq2_recurrent_dagger_broad.py
```

Frozen SHA-256 values:

- SF023 evaluator/tests:
  `0e99ac443cc2d9c1dde8a1d4efe3efe3e180cd72fb0358b630de5615e2869b2d` /
  `e93ab4e81b3a095bbb472ed436c55e8536e3e6579c2d047f2a9799cb2005fef2`;
- frozen SF015 evaluator/tests:
  `465f0be110b48708cde3d76dc851ff32d603fde0d332ecf7b327246eb1c22dfa` /
  `e0dc58829b1e21032ec4b7374b2980e2edaf8e1ac01c48a497b6b94343be5b2e`;
- recurrent actor/native evaluator:
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199` /
  `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`.

The SF023/SF015 screen and recurrent actor suite passes `11/11` before the one
screen.

## Admission

Full-course admission requires `512/512` valid six-gate completions, mean gates
passed exactly `6`, and zero crash, timeout, missed gate, out-of-order,
crossing-margin, action, wire-rate, thrust, or non-finite fault. Any lesser
result is diagnostic only and triggers a new student-state DAgger iteration;
never retry SF023 unchanged.

## Safety boundary

Offline native teacher-free screen only. Send zero FlightSim packets, do not
access N712, and do not shadow, reset, arm, setpoint, run a bounded attempt, or
select Submission.
