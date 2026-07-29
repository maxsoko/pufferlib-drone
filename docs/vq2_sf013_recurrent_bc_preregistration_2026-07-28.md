# VQ2-SF013 compact recurrent BC preregistration — 2026-07-28

Tag: `vq2_sf013_recurrent_bc_001`

SF012 source-locks `403722` causal legal-observation/executed-action records
from `64/64` successful randomized six-gate oracle courses. SF013 is the first
student fit. It may write one offline checkpoint but may not execute that
checkpoint in a native plant or FlightSim during this run.

## Fixed actor

Use exactly:

```text
4118-value legal observation
  -> existing 64x64 one-channel VQ2VisualEncoder
  -> 256 values
  -> one 256-wide GRU
  -> one joint Linear(256, 4) head
  -> tanh-bounded pitch, roll, thrust, yaw mean
```

Retain one learned four-value Gaussian `log_std`, initialized to `log(0.15)`,
for later on-policy collection. Behavioral cloning trains the deterministic
mean. The actor has no critic, auxiliary decoder, teacher, phase input,
coordinate, gate selector, runtime blend, or privileged feature path.

## Fixed data and optimization

- Dataset report/metadata SHA-256:
  `9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f` /
  `21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2`.
- Verify every SF012 array hash before optimizing.
- Use complete course-agent split `0..55` for training and `56..63` for
  validation; never split neighboring rows from one episode across sets.
- Run seed `42013`, exactly `8` epochs, eight agents per batch, and 64-step
  truncated backpropagation chunks.
- Begin recurrent state at zero only at each genuine episode start. Carry it
  causally across every later chunk and detach only the gradient graph at the
  chunk boundary.
- AdamW learning rate `3e-4`, weight decay `1e-5`, global gradient clip `1.0`.
- Joint masked action MSE weights are pitch/roll/thrust/yaw `1/1/4/1`; add
  predicted-action temporal smoothness with coefficient `1e-4`.
- Select one final checkpoint by the lowest validation weighted MSE across the
  eight fixed epochs. Do not select on native completion or sealed evidence.

Run:

```bash
.venv/bin/python scripts/train_vq2_recurrent_bc.py
```

Frozen pre-run SHA-256 values are:

- actor: `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`;
- trainer: `9f096981f8e0357d82a439bb5e1eadcbb7f07ed5103c22501a72d1c898dadba2`;
- actor tests: `5c371d672b7aeec0feb17aa2b82f4a6be4a8218b481c129af536cc6d8e74d692`;
- trainer tests: `b75ad7a3c932a9413d0f11cc665f504d969cc30de40f45a618bfb4e3276e28c6`;
- SF012 collector tests:
  `efcd566c0302f4eaf01b8b98810173f068fd1b415e9e3d3bee42151875e1f0af`;
- legal ABI module:
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`.

The combined actor, trainer, legal-boundary, and SF012 dataset contract suite
passes `21/21` before training. A measured warm benchmark processes about
`25050` observation-action rows per second with `0.080 GB` peak allocated GPU
memory, so this fixed run is feasible on the local RTX 3070.

## Offline admission

The checkpoint is admitted to a separately preregistered teacher-free native
diagnostic only if:

- all losses, gradients, weights, actions, and recurrent states remain finite;
- validation weighted MSE is at most `0.01`;
- every validation channel MSE is at most `0.03`;
- deterministic actions remain bounded in `[-1,1]` and all four action channels
  come from the one joint head; and
- the checkpoint records exact dataset, source, split, configuration, and
  training-history hashes/metadata.

Failure keeps the checkpoint diagnostic-only. A numerical pass still does not
prove closed-loop behavior; it only authorizes a new, teacher-free native
screen with zero action blend.

## Safety boundary

Offline PyTorch fit only. Send zero FlightSim packets, access zero privileged
actor values, do not touch N712, and do not run a native plant, shadow,
bounded attempt, or Submission action. FlightSim remains frozen and VQ2
Submission forbidden.
