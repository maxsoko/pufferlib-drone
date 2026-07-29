# VQ2-SF031 next causal-prefix aggregate fit — 2026-07-28

Tag: `vq2_sf031_recurrent_dagger_causal_prefix_next_001`

SF030 admits `247,606` legal SF028-driven labels over a fixed 512-step horizon;
432 trajectories remain active at the horizon, 80 terminate, and only three
are native low crashes. Report/metadata SHA-256 values are
`7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393` /
`99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2`.
SF028's held-out weighted MSE on this distribution is `0.186853` with channel
MSE `0.453556/0.255789/0.149616/0.000161`.

## Frozen training

- Parent SF028 checkpoint/report SHA-256:
  `b74e2f1c792fbd0e250a06a9c65b4fdaa3cc4f132e5e6bbab0a4c07a240b0403` /
  `c1ebd267ca0d35ed873287da1378720c316eb8957115465c5813bf284a4c7ccd`.
- Preserve complete SF012 and SF021 datasets. Replace the prior failure-tail
  source in the active aggregate with complete admitted SF030 prefixes.
- Pair every update `0.5` SF012 / `0.5` DAgger using fixed source pattern
  `SF021,SF030,SF030,SF030`, matching the `72,662/247,606` record ratio.
- Preserve the exact legal ABI/model, splits, batch `8`, TBPTT `64`, weight
  decay `1e-5`, gradient clip `1`, smoothness `1e-4`, and weights `1/1/4/1`.
- Seed `42031`; six epochs; learning rate `5e-5`; unchanged record-weighted
  aggregate selection.

Run once after source-locking implementation and tests.

Frozen SF031 wrapper/test SHA-256 values are
`9b6c9f4db91b995ac4537f8f60b35b74d6bd41a16bd7069b1d1a949fde859846` /
`232dc93fe93e9ca3cf4898f40401793860e22cad9a40829539b1957c0c1c7561`.
The SF031/aggregate/causal/legal-loader/recurrent suite passes `17/17` before
training.

## Offline admission

Require SF012 weighted MSE at most `0.01`; SF021 and SF030 weighted MSE at most
`0.02`; every DAgger channel at most `0.05`; finite values and full lineage.
Passing admits one separately preregistered fresh teacher-free full-course
screen only.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not access N712, and do not screen, shadow, reset, arm, setpoint,
run a bounded attempt, or select Submission.
