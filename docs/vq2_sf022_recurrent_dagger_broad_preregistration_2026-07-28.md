# VQ2-SF022 broad paired DAgger continuation — 2026-07-28

Tag: `vq2_sf022_recurrent_dagger_broad_001`

SF021 admits `72,662` clean legal student-state labels over `512/512`
terminated courses, including `157` post-Gate-1 trajectories. Its report and
metadata SHA-256 values are
`438cc4ba435a1cba223fba204fbc2637802ce68c193e5994c5e56e076f1f6c4b` /
`ae315f9469112e069c3b68b72e848c7faa56c21cb79a2dbd4bd4a7a4c2273875`.

On the held-out SF021 agents `448..511`, SF014 has `0.6444145` weighted MSE.
SF019 has `0.0122165` weighted MSE and channel MSE
`0.004621/0.052728/0.006999/0.000171`. The broader independent sample thus
confirms that SF019 already learned the student-state distribution and that
roll remains the single narrow gap.

## Frozen training

- Parent: SF019 checkpoint/report SHA-256
  `a391516076f49549255bf323f94859f90a0d15a5f3406c436d3045682e681352` /
  `9311411d3ab30321623575b904fb018e33396d88b856f7e0554a764926984c4c`.
- Preserve the exact 4118-value legal input, one visual encoder, one 256-wide
  GRU, one joint tanh four-channel head, and all actor parameters trainable.
- Pair one complete SF012 course pass per epoch with a cyclic stream of
  complete-prefix SF021 episodes. Every optimizer update is exactly
  `0.5 * SF012 loss + 0.5 * SF021 loss`; no sequential source passes.
- SF012 split: agents `0..55` train and `56..63` validation. SF021 split:
  agents `0..447` train and `448..511` validation.
- Seed `42022`; four epochs; eight agents per batch; 64-step TBPTT; AdamW
  learning rate `1e-4`, weight decay `1e-5`; gradient clip `1.0`; temporal
  smoothness `1e-4`; pitch/roll/thrust/yaw weights `1/1/4/1`.
- Select the lowest equal-weight mean of SF012 and SF021 validation weighted
  MSE. Do not change thresholds after observing results.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_dagger_broad.py
```

Frozen SHA-256 values:

- SF022 trainer/tests:
  `b05f9958794838aef2effea6f46f1c333c8e6961dd7b1430f7329b31c5e75eb9` /
  `108ae0750672688f14a57a0a1ecc56fb6673bac0a9c7ea84b4d17ed28b7e6a7b`;
- paired trainer/tests:
  `faadd2f29afe9446dddf5c5a2cf4a576b0dfb60c9c124c4230a8285b19eaa8ad` /
  `fc8ecdc21c89a956acb157250703ff059780f6fe263412e8ef4e8c2d2ad90d56`;
- legal loader, aggregate helper, and recurrent actor:
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583` /
  `50c2fe2d88ed3fd8155d33a5d8c1b931de4c8a3b37498a41830dc428979c9ea2` /
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`.

The broad, paired, aggregate, legal-loader, and recurrent actor suite passes
`18/18` before training.

## Offline admission

The selected checkpoint may enter one separately preregistered teacher-free
native screen only if SF012 validation weighted MSE is at most `0.01`, SF021
validation weighted MSE is at most `0.02`, every SF021 channel MSE is at most
`0.05`, all values are finite, and the checkpoint source-locks the complete
lineage. Passing does not itself demonstrate closed-loop completion.

## Safety boundary

Offline PyTorch training only. Persist zero privileged actor values, send zero
FlightSim packets, do not access N712, and do not run a native screen, shadow,
reset, arm, setpoint, bounded attempt, or Submission action.
