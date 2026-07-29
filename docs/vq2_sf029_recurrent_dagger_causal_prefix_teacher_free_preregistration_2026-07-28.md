# VQ2-SF029 SF028 teacher-free full-course screen — 2026-07-28

Tag: `vq2_sf029_recurrent_dagger_causal_prefix_teacher_free_512`

SF028 checkpoint/report SHA-256 values are
`b74e2f1c792fbd0e250a06a9c65b4fdaa3cc4f132e5e6bbab0a4c07a240b0403` /
`c1ebd267ca0d35ed873287da1378720c316eb8957115465c5813bf284a4c7ccd`.
Its selected epoch `4` passes the causal numerical gate with SF012/SF021/SF024
weighted MSE `0.002299/0.003696/0.010829` and every DAgger channel below
`0.02`.

SF029 is a new closed-loop test. The causal training view does not cap this
screen or alter any runtime behavior.

## Fixed screen

- Run `512` fresh randomized six-gate courses, one per agent, seed `42029`,
  fixed `0.75 m` aperture/plant and full `180 s` horizon.
- Start recurrent state at the genuine reset and advance it on every active
  legal observation. SF028's joint mean emits all four plant-action channels.
- Teacher blend/switches, labels, privileged actor inputs, analytic action
  logic, clipping, override, fallback, and updates remain zero.
- Record all native progress/safety metrics, action statistics, inference rate,
  and exact lineage hashes.

Run once after source-locking implementation and tests. Never retry unchanged.

Frozen SF029 wrapper/test SHA-256 values are
`6773d5b023d81a0c661b0228e25dc70e3254c650ba73655b39c7f86cd3584405` /
`30c78c6bb37662fb7fc93fe6ecd7d7615ecdebea765891a54205e442d0a2ce75`.
The SF029/SF023/SF015 screen, causal trainer, and recurrent actor suite passes
`12/12` before the one screen.

## Admission

Require `512/512` valid six-gate completions, mean gates exactly `6`, zero
crash, timeout, missed gate, out-of-order, crossing-margin, action, wire-rate,
thrust, or non-finite fault. Any lesser result remains diagnostic and triggers
a new DAgger state distribution; it does not authorize FlightSim.

## Safety boundary

Offline native teacher-free screen only. Send zero FlightSim packets, do not
access N712, and do not shadow, reset, arm, setpoint, run a bounded attempt, or
select Submission.
