# VQ2-SF019 simultaneous-gradient recurrent DAgger fit — 2026-07-28

Tag: `vq2_sf019_recurrent_dagger_paired_001`

SF018 is rejected. Although its record counts were balanced, its source passes
were sequential. The child reduced SF017 validation weighted MSE from `0.645`
to `0.0111` but regressed SF012 validation from `0.000280` to `0.3866`. It
therefore fails the retained-manifold gate. A frozen-trunk ridge diagnostic
also cannot fit both distributions (`0.0223` SF012 and `0.1649` SF017), so the
shared representation must adapt, but it must receive both objectives in the
same optimizer update.

SF019 returns to the unchanged SF014 parent and makes only that optimizer
correction.

## Frozen paired fit

- Parent SF014 checkpoint SHA-256:
  `00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a`.
- SF012 report/metadata SHA-256:
  `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f` /
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`.
- SF017 report/metadata SHA-256:
  `58937d85d5cbd349a5bf7c2628f4d307c0e3ed97d68e85db850c2c49bcda687e` /
  `d50cbde39b6bd052b2ad2293b7a876bedf2c1aa8c4a7847261d3860eedefc39e`.
- Keep agents `0..55` for training and `56..63` for validation in both sources.
- Initialize the entire legal CNN, one 256-wide GRU, joint four-action head,
  and recurrent rules from SF014.
- Seed `42019`; exactly four epochs; eight agents per batch; 64-step TBPTT;
  learning rate `1e-4`; AdamW weight decay `1e-5`; gradient clip `1.0`; action
  weights `1/1/4/1`; smoothness coefficient `1e-4`.
- Each epoch advances exactly one complete SF012 train-course pass. For every
  SF012 optimizer chunk, advance one SF017 chunk from a separate recurrent
  state. Cycle SF017 only at complete episode-batch boundaries.
- In the same backward pass, minimize exactly
  `0.5 * SF012_loss + 0.5 * SF017_loss`. Never finish an optimizer update from
  only one source. This removes the measured source-order overwrite while
  preserving full causal histories for both.
- Validate both disjoint sources after every epoch. Select the minimum
  equal-weight mean of the two weighted validation MSE values.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_dagger_paired.py
```

Frozen SHA-256 values:

- paired trainer:
  `53961bace16eb1198017c7986df449f7ee059c84d885aa1dba03d4495b3a4de0`;
- paired tests:
  `fc8ecdc21c89a956acb157250703ff059780f6fe263412e8ef4e8c2d2ad90d56`;
- aggregate helper:
  `50c2fe2d88ed3fd8155d33a5d8c1b931de4c8a3b37498a41830dc428979c9ea2`;
- legal loader/training helper:
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`;
- actor: `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- rejected SF018 report:
  `4d96617a5e832925b535aa94fc32e353d1a1790197004bf7f9df3b47852b3a7f`.

The paired, aggregate, legal-loader, actor, and ABI suite passes `22/22` before
training.

## Offline admission

Admit the selected checkpoint only to a separately preregistered teacher-free
native screen if all values remain finite, SF012 validation weighted MSE is at
most `0.01`, SF017 validation weighted MSE is at most `0.02`, every SF017
validation channel MSE is at most `0.05`, all actions remain bounded, and both
source hashes and full histories are recorded. Numerical passage is not
closed-loop admission.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not touch N712, and do not run a native policy screen, shadow,
reset, arm, setpoint, bounded attempt, or Submission action.
