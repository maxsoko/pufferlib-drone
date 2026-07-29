# VQ2-SF018 record-balanced recurrent DAgger fit — 2026-07-28

Tag: `vq2_sf018_recurrent_dagger_001`

SF017 source-locks `9134` safe student-visited transitions, including `7999`
train-course and `1135` validation-course rows. The SF014 parent's weighted
MSE against these queried labels is `0.6629946701`, versus `0.0002803708` on
its disjoint SF012 validation histories. Concatenating SF017 uniformly would
give it only about two percent of the aggregate and would not address this
measured distribution shift.

SF018 therefore performs one fixed record-balanced aggregate fit while
retaining the full oracle-history anchor.

## Frozen fit

- Parent SF014 checkpoint SHA-256:
  `00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a`.
- SF012 report/metadata SHA-256:
  `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f` /
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`.
- SF017 report/metadata SHA-256:
  `58937d85d5cbd349a5bf7c2628f4d307c0e3ed97d68e85db850c2c49bcda687e` /
  `d50cbde39b6bd052b2ad2293b7a876bedf2c1aa8c4a7847261d3860eedefc39e`.
- Keep course agents `0..55` in training and `56..63` in validation for both
  sources.
- Initialize every parameter and recurrent state rule from SF014. The actor
  remains one legal CNN, one 256-wide GRU, and one joint four-channel head.
- Seed `42018`; exactly four epochs; eight agents per batch; 64-step TBPTT;
  recurrent state reset only at genuine episode boundaries.
- Each epoch contains one complete SF012 training pass (`352973` rows) and
  exactly 44 complete SF017 training passes (`44 * 7999 = 351956` rows). Shuffle
  these 45 source passes deterministically. This balances records while
  preserving one full solved-manifold anchor.
- AdamW learning rate `1e-4`, weight decay `1e-5`, gradient clip `1.0`, joint
  pitch/roll/thrust/yaw MSE weights `1/1/4/1`, smoothness coefficient `1e-4`.
- Evaluate both disjoint source-validation sets after every epoch. Select the
  minimum equal-weight mean of their two weighted MSE values.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_dagger.py
```

Frozen SHA-256 values:

- DAgger trainer:
  `50c2fe2d88ed3fd8155d33a5d8c1b931de4c8a3b37498a41830dc428979c9ea2`;
- focused trainer tests:
  `6462149798dd38917cfa297ed5095f2bcc970e86ab08fa257fc7596b87b0a0c8`;
- shared legal loader/training helpers:
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`;
- actor: `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- legal ABI: `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`.

The aggregate trainer, loader, actor, DAgger collector, and legal ABI suite
passes `21/21` before training.

## Offline admission

Admit the selected child only to a separately preregistered teacher-free
native diagnostic if all values remain finite, SF012 validation weighted MSE
is at most `0.01`, SF017 validation weighted MSE is at most `0.02`, every SF017
validation channel MSE is at most `0.05`, all actions remain bounded, and the
checkpoint source-locks both datasets, the parent, split, schedule, objective,
and full history.

Numerical admission is not closed-loop admission. Do not use teacher blend in
the later policy screen.

## Safety boundary

Offline PyTorch fit only. Persist zero privileged actor values, send zero
FlightSim packets, do not touch N712, and do not run a native policy screen,
shadow, reset, arm, setpoint, bounded attempt, or Submission action.
