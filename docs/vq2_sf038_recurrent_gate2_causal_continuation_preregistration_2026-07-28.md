# VQ2-SF038 Gate-2 causal-fit continuation — 2026-07-28

Tag: `vq2_sf038_recurrent_gate2_causal_continuation_001`

SF037 is rejected but reduces capped SF035 held-out weighted MSE from
`0.126644` to `0.042132`. Its selected train/validation channel errors remain
similar enough to show under-convergence rather than a held-out-only failure.
SF012 remains admitted at `0.004556`; SF030 is narrowly outside its weighted
gate at `0.022402`. SF037 checkpoint/report SHA-256 values are
`0526cb27c8667784bc377cde88dfc7af015f9af8d15ae6dd6de0547517d76eaf` /
`7be2c4f5c5f0e575001d189f0e11ffd7c6dd8dd987b9d85ea160bdf241fde811`.

Continue the exact SF037 objective and source locks for eight epochs with seed
`42038`, changing only learning rate from `5e-5` to `2e-5`. Retain complete
SF012/SF030, the first 768 causal SF035 steps, DAgger pattern
`SF030,SF035,SF035`, training-loss weights `1/4/4/1`, fixed validation weights
`1/1/4/1`, model/ABI, splits, TBPTT, batch, regularization, and
admission-first selection.

Admission is unchanged: SF012 weighted MSE at most `0.01`; SF030 and capped
SF035 weighted MSE at most `0.02`; every DAgger channel at most `0.05`.
Passing authorizes only one separately preregistered fresh teacher-free native
screen. Otherwise reject without screening.

Offline PyTorch only. Send zero FlightSim packets, do not access N712, and do
not shadow, reset, arm, setpoint, run a bounded attempt, or select Submission.

