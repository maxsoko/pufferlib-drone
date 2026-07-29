# VQ2-SF037 recurrent Gate-2 causal fit — 2026-07-28

Tag: `vq2_sf037_recurrent_gate2_causal_fit_001`

SF036 measures SF033 error on held-out SF035 records. Gate 1 crosses near the
end of the first 256 steps; the causal Gate-2 correction lies in `[256,768)`.
Weighted error rises from `0.02113` through step 256 to `0.12666` through step
768, while later post-divergence bins reach `0.36–0.77`. SF036 report SHA-256
is `e81557cb912d8a01a9f4914be01b629828a9f04119c487cd23919f5962e39eb8`.

## Frozen training

- Parent SF033 checkpoint/report SHA-256:
  `91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7` /
  `f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279`.
- Clean anchor: complete SF012.
- Gate-1/on-policy anchor: complete SF030, report/metadata SHA-256
  `7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393` /
  `99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2`.
- Gate-2 correction: a read-only first-`768`-step view of SF035, whose
  report/metadata SHA-256 values are
  `d07db36e11d321962defa217c80e06ff7678c8682f19ea6212d5fc8e3bb8313f` /
  `8622f6c9be5cc15fbd46d9e5fd65d86b223406e820d658013f3478ebb8ec9851`.
- Pair every update `0.5` SF012 / `0.5` DAgger and alternate DAgger sources
  `SF030,SF035,SF035`, approximating their capped record ratio.
- Preserve the exact 4,118-value legal ABI, 256-wide CNN/GRU, one joint head,
  fixed splits, batch `8`, TBPTT `64`, weight decay `1e-5`, gradient clip `1`,
  and smoothness `1e-4`.
- Six epochs, seed `42037`, learning rate `5e-5`. Retain fixed validation
  weights `1/1/4/1`; retain the admitted SF033 training-loss emphasis
  `1/4/4/1` so roll and thrust have equal priority.
- Prefer a numerically admitted epoch; otherwise select the lowest unchanged
  aggregate validation score.

## Admission

Require SF012 weighted MSE at most `0.01`; complete SF030 and capped SF035
weighted MSE at most `0.02`; every channel on both DAgger distributions at
most `0.05`; finite values and complete lineage. Passing authorizes one new
fresh teacher-free native screen only.

Offline PyTorch only. Send zero FlightSim packets, do not access N712, and do
not screen, shadow, reset, arm, setpoint, run a bounded attempt, or select
Submission during training.

