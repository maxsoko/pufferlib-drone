# VQ2-SF028 causal-prefix DAgger fit — 2026-07-28

Tag: `vq2_sf028_recurrent_dagger_causal_prefix_001`

SF027 report SHA-256
`527ab1be544634fa67bb9b411cd450f73552d3aee935e4b766cae3fb91c38809`
localizes SF026's held-out SF024 error. The cumulative first `384` steps have
weighted MSE `0.014497` and channel MSE
`0.022751/0.026474/0.013024/0.000159`. The `384..512` bin rises to `0.058546`
weighted MSE; `512..768` and `768..1024` rise to `0.216780/0.286665`.

SF028 trains the causal corrections leading into the average six-second crash
without making rare, visually sparse late failure suffixes dominate the actor.
Complete clean six-gate oracle histories remain mandatory and uncropped.

## Frozen training

- Parent SF026 checkpoint/report SHA-256:
  `166c9b28bc1d4848e0e5e6d7028f9637835540cbe2f8e0ae883a8962d8e0e2b7` /
  `a1df63ceb0bf7fe85cc50a0010ed32d08395d76bc4f162b38aa900a67c643c03`.
- Preserve all SF012 and SF021 episodes exactly. Present SF024 through a
  read-only prefix view with `length=min(original_length,384)`; do not modify
  the admitted dataset or use privileged state to select individual records.
- Pair every SF012 update `0.5/0.5` with DAgger. Use fixed repeating source
  pattern `SF021,SF024,SF024`, matching the capped record ratio.
- Preserve the exact legal ABI/model, data splits, batch `8`, TBPTT `64`,
  weight decay `1e-5`, gradient clip `1`, smoothness `1e-4`, channel weights
  `1/1/4/1`, and aggregate checkpoint selection.
- Seed `42028`; exactly four epochs; learning rate `2e-5`.

Run once after source-locking implementation and tests.

Frozen causal trainer/test SHA-256 values are
`bde25e3ddc8ef81f65aa1c06d679a9bc67f7056649a6fdde7256acfdbf673de3` /
`6af4f4addfb8554d80631ce4d25ef0292feb9ded01733b221cbac22c88b2adcc`.
The causal/aggregate/horizon/legal-loader/recurrent suite passes `16/16`
before training.

## Offline admission

Under the causal view, require SF012 weighted MSE at most `0.01`; SF021 and
capped-SF024 weighted MSE at most `0.02`; every DAgger channel MSE at most
`0.05`; finite values and complete provenance. Passing admits one separately
preregistered fresh teacher-free full-course screen. That screen still has no
horizon cap and requires zero crash and all six gates.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not access N712, and do not screen, shadow, reset, arm, setpoint,
run a bounded attempt, or select Submission.
