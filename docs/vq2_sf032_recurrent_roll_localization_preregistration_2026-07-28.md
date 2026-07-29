# VQ2-SF032 recurrent roll-localization preregistration — 2026-07-28

## Question

Localize the held-out SF030 roll error of the rejected SF031 compact recurrent
actor before changing its data, loss, or architecture.  Determine whether the
residual is concentrated late in the 512-step causal prefix, in a small number
of trajectories, or when the legal visual mask is absent.

## Frozen inputs

- SF031 checkpoint SHA-256:
  `b7c0e1b0516a2ae410b77a4558c0298fd2fe8c11793e7e814d2009cc88b336fc`.
- SF031 report SHA-256:
  `8c22f07a40ce12737a88db930d92c9e1a7f0298d2392884b9cb0dfbbf2380649`.
- SF030 report SHA-256:
  `7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393`.
- SF030 metadata SHA-256:
  `99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2`.
- Use only SF030 validation agents `448..511`, restoring recurrent state from
  zero and advancing it over every legal record in order.

## Fixed analysis

- Report four-action MSE and the fixed weighted MSE for time bins
  `[0,128)`, `[128,192)`, `[192,256)`, `[256,320)`, `[320,384)`,
  `[384,448)`, and `[448,512)`.
- Report cumulative horizons `128`, `192`, `256`, `320`, `384`, `448`, and
  `512`.
- Report roll MSE quantiles and the ten worst held-out agents.
- Partition legal records into visual-mask-present and visual-mask-absent
  groups using only the stored legal mask.
- Report target and prediction roll moments to distinguish lack of target
  variation from actor error.

This is diagnosis only.  It cannot admit SF031, authorize a teacher-free
screen, select a successor objective, access N712, or send any FlightSim
packet.  Any successor fit requires a new source-locked tag and explicit
contract based on this report.

