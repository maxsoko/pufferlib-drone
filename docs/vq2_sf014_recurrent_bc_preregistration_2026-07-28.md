# VQ2-SF014 corrected compact recurrent BC preregistration — 2026-07-28

Tag: `vq2_sf014_recurrent_bc_001`

SF013 completed all eight learning epochs and numerically passed its offline
threshold; epoch 8 reached validation weighted MSE `0.0002803708407308082`.
Checkpoint packaging then failed because the local recurrent-output name
`output` shadowed the output-directory argument. The process exited before
creating the SF013 directory, checkpoint, or report. Reject SF013 as a
checkpoint run.

SF014 changes only that local variable name to `actor_output`, adds a focused
regression assertion, changes the unique tag, and repeats the exact deterministic
fit. No model, data, split, seed, optimizer, schedule, objective, or selection
value changes.

## Frozen actor and fit

Use exactly one legal-only actor:

```text
4118 legal values -> 64x64 VQ2VisualEncoder -> 256
  -> one 256-wide GRU -> one joint Linear(256,4) -> tanh CTBR mean
```

The four-value Gaussian `log_std` remains initialized to `log(0.15)`. There is
no critic, auxiliary decoder, teacher, phase input, coordinate, gate selector,
blend, or privileged actor feature.

- SF012 report/metadata SHA-256:
  `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f` /
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`.
- Verify every dataset file hash before the first update.
- Course agents `0..55` train; `56..63` validate.
- Seed `42013`; eight epochs; eight agents per batch; 64-step TBPTT; recurrent
  state reset only on genuine episode starts and carried across every chunk.
- AdamW `3e-4`, weight decay `1e-5`, gradient clip `1.0`.
- Joint pitch/roll/thrust/yaw MSE weights `1/1/4/1`; temporal smoothness
  coefficient `1e-4`.
- Select the single lowest validation weighted-MSE epoch.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_bc.py
```

Frozen pre-run SHA-256 values:

- actor: `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- corrected trainer:
  `b9249c4aab4c7469f5c31aad2417a1116ad181e2bda616bbfc21ba212a10da17`;
- actor tests: `5c371d672b7aeec0feb17aa2b82f4a6be4a8218b481c129af536cc6d8e74d692`;
- corrected trainer tests:
  `b12d79e048b27075629be2e2458a6e6d8a1b557bcf26a29f1a529043287448ad`;
- SF012 collector tests:
  `efcd566c0302f4eaf01b8b98810173f068fd1b415e9e3d3bee42151875e1f0af`;
- legal ABI module:
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`.

The corrected combined suite passes `22/22`. The added test explicitly fails
if the output-directory parameter is ever rebound by the recurrent result.

## Offline admission

Admit the selected checkpoint only to a separately preregistered teacher-free
native diagnostic if all values remain finite, validation weighted MSE is at
most `0.01`, every validation channel MSE is at most `0.03`, the deterministic
joint-head action stays in `[-1,1]`, and exact source/data/split/config/history
metadata is present. Numerical passage does not prove closed-loop flight.

## Safety boundary

Offline PyTorch fit only. Send zero FlightSim packets, persist zero privileged
actor values, do not touch N712, and do not run a native policy screen, shadow,
bounded attempt, or Submission action. FlightSim remains frozen and VQ2
Submission forbidden.
