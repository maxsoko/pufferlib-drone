# VQ2-SF026 aggregate convergence continuation — 2026-07-28

Tag: `vq2_sf026_recurrent_dagger_aggregate_continuation_001`

SF025 is numerically rejected, but its selected epoch `4` shows continued
underfit rather than a train/validation split or capacity failure. It reduces
SF024 held-out weighted MSE from `0.422596` to `0.060356`; corresponding train
weighted error remains of the same scale. It simultaneously recovers SF012 to
`0.006946` and SF021 to `0.007746`. Checkpoint/report SHA-256 values are
`d0a4058fba5428b35514b3a7c74ab73f7f235e0fc4ac45f062d87591cda5523c` /
`3b4d0bf47334f1e3576aced14c68178cd6ac9dab19a2ee6de3d27422cb345118`.

SF026 is one convergence continuation. It changes no data, model, loss,
schedule, split, action weights, or selection rule.

## Frozen continuation

- Parent: selected SF025 checkpoint above.
- Reuse the exact frozen SF025 loop and fixed DAgger source pattern
  `SF021,SF024,SF024,SF024` with every update paired `0.5/0.5` against SF012.
- Preserve all dataset hashes, train/validation splits, 4118-value legal ABI,
  visual encoder, 256-wide GRU, joint tanh head, batch size `8`, TBPTT `64`,
  weight decay `1e-5`, gradient clip `1`, smoothness `1e-4`, and channel weights
  `1/1/4/1`.
- Change seed to `42026`, run exactly six epochs, and lower learning rate only
  from `1e-4` to `5e-5`.
- Preserve SF025's record-weighted aggregate selection rule and all numerical
  admission thresholds without adjustment.

Run once:

```bash
.venv/bin/python scripts/continue_vq2_recurrent_dagger_aggregate.py
```

Frozen SHA-256 values:

- SF026 wrapper/tests:
  `6bb1be4aea4ac1475ecfc54bbeb0a913d9177da65df984e49f3480aec73ea7a4` /
  `ef826be8c05e22b6abe16f1f75384aace895297d9f3b7208f6544a829be75ec5`;
- frozen SF025 trainer/tests:
  `4eba95b3272425ed2fcfaf5100f4ce0b2d41c42467fc270777326a1a376b66b2` /
  `c851d81b26651588b7cb804e4c18f5f2a27716c5891d973c3d4f3199dac8882a`;
- paired trainer and recurrent actor:
  `faadd2f29afe9446dddf5c5a2cf4a576b0dfb60c9c124c4230a8285b19eaa8ad` /
  `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`.

The continuation/aggregate/broad/paired/legal-loader/recurrent suite passes
`20/20` before training.

## Offline admission

Require SF012 validation weighted MSE at most `0.01`; both SF021 and SF024
weighted MSE at most `0.02`; every DAgger channel MSE at most `0.05`; all values
finite; and the source-locked parent, aggregate loop, continuation wrapper, and
datasets. A pass admits only a separately preregistered teacher-free screen.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not access N712, and do not screen, shadow, reset, arm, setpoint,
run a bounded attempt, or select Submission.
