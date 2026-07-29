# VQ2-SF025 two-iteration DAgger aggregate — 2026-07-28

Tag: `vq2_sf025_recurrent_dagger_aggregate_001`

SF024 admits `224,050` legal SF022-visited query labels. Report/metadata
SHA-256 values are
`5f3d344643a75a66cdbb1b159ee2105c563201774d8e7122d19e888fc414cb26` /
`1b7491bc8130ff1cb1a2a6b4c64cf7abb580723ccaa65600f842df3af8a498fa`.
On held-out SF024 agents `448..511`, SF022 has weighted MSE `0.422596` and
pitch/roll/thrust/yaw MSE `1.15888/0.99220/0.20173/0.000151`.

SF025 preserves clean oracle trajectories and both student distributions in
one compact, record-balanced update stream.

## Frozen training

- Parent SF022 checkpoint/report SHA-256:
  `57949ba391ffdc4d1e0dda5aa352fbb7b6dcb2a51b1f7742018926f6c6b2d903` /
  `2cfc253ef13dc4a438a784df595a52947aaa0bcdb42320a6069bebdfa85f2f66`.
- Keep the exact 4118-value legal ABI, visual encoder, 256-wide GRU, joint tanh
  head, and all parameters trainable.
- Every update pairs one SF012 chunk with one DAgger chunk at equal `0.5/0.5`
  loss. Select DAgger sources in the fixed repeating pattern
  `SF021,SF024,SF024,SF024`, approximating their `72,662/224,050` record ratio.
  Each source stream preserves complete recurrent prefixes across its selected
  updates. Never perform an epoch-scale source pass.
- SF012 train/validation agents `0..55/56..63`; both DAgger datasets
  `0..447/448..511`.
- Seed `42025`; four epochs; eight agents/batch; 64-step TBPTT; AdamW learning
  rate `1e-4`, weight decay `1e-5`; gradient clip `1`; smoothness `1e-4`;
  channel weights `1/1/4/1`.
- Select the lowest `0.5 * SF012 weighted MSE + 0.5 * record-weighted combined
  DAgger weighted MSE`.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_dagger_aggregate.py
```

Frozen SHA-256 values:

- SF025 trainer/tests:
  `4eba95b3272425ed2fcfaf5100f4ce0b2d41c42467fc270777326a1a376b66b2` /
  `c851d81b26651588b7cb804e4c18f5f2a27716c5891d973c3d4f3199dac8882a`;
- SF022 and paired trainers:
  `b05f9958794838aef2effea6f46f1c333c8e6961dd7b1430f7329b31c5e75eb9` /
  `faadd2f29afe9446dddf5c5a2cf4a576b0dfb60c9c124c4230a8285b19eaa8ad`;
- legal loader and recurrent actor:
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583` /
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`.

The aggregate/broad/paired/legal-loader/recurrent suite passes `18/18` before
training.

## Offline admission

Require SF012 validation weighted MSE at most `0.01`; both SF021 and SF024
validation weighted MSE at most `0.02`; every channel MSE on both DAgger sets
at most `0.05`; all values finite; and complete lineage hashes. A pass admits
one separately preregistered teacher-free screen, not the policy itself.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not access N712, and do not screen, shadow, reset, arm, setpoint,
run a bounded attempt, or select Submission.
