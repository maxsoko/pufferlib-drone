# VQ2-SF033 recurrent roll-priority continuation — 2026-07-28

Tag: `vq2_sf033_recurrent_roll_priority_001`

SF031 is rejected only because SF030 held-out roll MSE is `0.0675627` above
the frozen `0.05` channel limit; its SF012/SF021/SF030 weighted MSE values are
`0.00429976/0.00506485/0.01988894`. SF032 localizes the residual: cumulative
weighted MSE through step 448 is `0.0154127`, while the final `[448,512)` bin
has roll MSE `0.222255`. All `30,946` held-out records contain legal mask
evidence, and prediction roll RMS `0.430327` is below target RMS `0.513419`.
The error is therefore a late under-amplitude fit problem, not legal visual
dropout or a reason to change the observation ABI.

## Frozen continuation

- Parent SF031 checkpoint/report SHA-256:
  `b7c0e1b0516a2ae410b77a4558c0298fd2fe8c11793e7e814d2009cc88b336fc` /
  `8c22f07a40ce12737a88db930d92c9e1a7f0298d2392884b9cb0dfbbf2380649`.
- SF032 diagnosis report SHA-256:
  `3c4a7f287278d89d54ab4eeeef59953c375018a7fd7770c709768c6b99c0b940`.
- Preserve complete SF012, SF021, and SF030 data and the fixed source pattern
  `SF021,SF030,SF030,SF030`.
- Preserve the exact 4,118-value legal ABI, 256-wide CNN/GRU, one joint action
  head, agent splits, batch `8`, TBPTT `64`, weight decay `1e-5`, gradient
  clip `1`, and smoothness `1e-4`.
- Continue from SF031 for six epochs, seed `42033`, learning rate `2e-5`.
- Change only the *training-loss* pitch/roll/thrust/yaw weights from
  `1/1/4/1` to `1/4/4/1`. Validation retains the original fixed `1/1/4/1`
  weighted metric and unchanged per-channel limits.
- Prefer a numerically admitted epoch; within the same admission class choose
  the lowest original aggregate validation score. This cannot make a failing
  epoch pass and does not alter any threshold.

Run once after source-locking implementation and tests.

## Offline admission

Unchanged: SF012 weighted MSE at most `0.01`; SF021 and SF030 weighted MSE at
most `0.02`; every SF021/SF030 channel MSE at most `0.05`; finite values and
complete lineage. Passing authorizes only a separately preregistered fresh
teacher-free native screen.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not access N712, and do not screen, shadow, reset, arm, setpoint,
run a bounded attempt, or select Submission.

