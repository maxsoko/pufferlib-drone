# VQ2-SF041 public-phase joint residual — 2026-07-28

Tag: `vq2_sf041_public_phase_residual_001`

SF040 proves a constant 256-value phase shift is under-capacity. Add the next
smallest phase-zero-exact mechanism: one bias-free `256 -> 4` joint action
residual, multiplied by the public phase. Retain one visual encoder, one
256-wide GRU, and one actor emitting the complete CTBR vector. Initialize from
SF033 and freeze every old parameter. Train only the 256-value phase embedding
and 1,024-value joint residual (`1,280` parameters total). At phase zero the
actor must remain bit-exact to SF033 for every input.

Use the same admitted SF039 split, 768-step causal horizon, Gate-2-only loss,
and fixed optimizer contract as SF040: seed `42041`, 12 epochs, batch 8,
sequence chunks 64, AdamW `0.003`, zero weight decay, gradient clip `1.0`,
loss weights `[4,4,4,1]`, and smoothness `0.0001`. Select the lowest finite
held-out Gate-2 weighted MSE. Admit numerically only at weighted MSE `<=0.02`,
every channel MSE `<=0.05`, and exact phase-zero parity.

This offline fit sends zero FlightSim packets, does not access N712, and does
not authorize a screen, shadow, bounded attempt, or Submission.

