# VQ2-SF020 paired DAgger roll continuation — 2026-07-28

Tag: `vq2_sf020_recurrent_dagger_roll_001`

SF019 proves simultaneous source gradients solve SF018's catastrophic
forgetting. Its selected child passes the SF012 (`0.0007586`) and SF017
weighted (`0.0108502`) validation gates. It misses only the explicitly fixed
SF017 roll-channel MSE gate: `0.0631558 > 0.05`.

SF020 is one narrow continuation from SF019. Preserve the paired algorithm and
all data. Lower the learning rate and increase only the roll channel's
supervised weight.

## Frozen continuation

- Parent SF019 checkpoint/report SHA-256:
  `a391516076f49549255bf323f94859f90a0d15a5f3406c436d3045682e681352` /
  `9311411d3ab30321623575b904fb018e33396d88b856f7e0554a764926984c4c`.
- Retain the exact SF012 and SF017 source locks, agent `0..55` train / `56..63`
  validation split, one legal CNN, one 256-wide GRU, and one joint action head.
- Retain simultaneous `0.5 * SF012 + 0.5 * SF017` loss in every update, one
  complete SF012 course pass per epoch, and cyclic complete-prefix SF017
  batches.
- Seed `42020`; exactly four epochs; eight agents per batch; 64-step TBPTT;
  AdamW weight decay `1e-5`; gradient clip `1.0`; smoothness `1e-4`.
- Change learning rate only from `1e-4` to `5e-5`.
- Change pitch/roll/thrust/yaw weights only from `1/1/4/1` to `1/2/4/1`.
- Continue selecting the equal-weight mean of the two weighted validation MSE
  values.

Run once:

```bash
.venv/bin/python scripts/train_vq2_recurrent_dagger_roll.py
```

Frozen SHA-256 values:

- parameterized paired trainer:
  `faadd2f29afe9446dddf5c5a2cf4a576b0dfb60c9c124c4230a8285b19eaa8ad`;
- SF020 wrapper:
  `83d28b380ba6a30a2ddc653978fd100961f3f33cc3a2871d18ecc93717131b18`;
- paired tests:
  `fc8ecdc21c89a956acb157250703ff059780f6fe263412e8ef4e8c2d2ad90d56`;
- continuation tests:
  `b413bc73cd7c0f9874edc1e1a364ea0489f67c6923330711245bd300ed1a58aa`.

The paired, continuation, aggregate, and legal loader suite passes `12/12`
before training.

## Offline admission

The child may enter a separately preregistered teacher-free native screen only
if SF012 validation weighted MSE is at most `0.01`, SF017 validation weighted
MSE is at most `0.02`, every SF017 channel MSE is at most `0.05`, all values are
finite and bounded, and the checkpoint source-locks the complete lineage. No
threshold changes from SF019.

## Safety boundary

Offline PyTorch only. Persist zero privileged actor values, send zero FlightSim
packets, do not touch N712, and do not run a native policy screen, shadow,
reset, arm, setpoint, bounded attempt, or Submission action.
