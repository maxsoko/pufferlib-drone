# VQ2-SF040 frozen-base public-phase adapter — 2026-07-28

Tag: `vq2_sf040_public_phase_adapter_001`

Fit the minimal public-progress amendment. Initialize one
`VQ2PhaseRecurrentActor` from SF033 so the original 4,118-input path is exact
and the new bias-free 1-to-256 phase embedding is zero. Freeze the encoder,
GRU, joint four-action head, and log standard deviation. Train only those 256
new weights.

Use admitted SF039, agents 0--447 for training and 448--511 for validation.
Warm each recurrent state from step zero, cap the causal sequence at step 768,
and apply supervised loss only where the held public value is exactly `1/6`
(Gate 2 active). Use seed `42040`, 12 epochs, batch 8, chunks of 64, AdamW
learning rate `0.003`, no weight decay, gradient clip `1.0`, action-loss weights
`[4, 4, 4, 1]`, and smoothness weight `0.0001`. Select the lowest finite
Gate-2 validation weighted MSE; numerical admission additionally requires it
to be at most `0.02`, each channel MSE at most `0.05`, and phase-zero output to
remain bit-exact to SF033.

This is an offline fit only. It cannot authorize an admission screen unless
the numerical gate passes. It sends zero FlightSim packets, does not access
N712, and cannot shadow, reset, arm, setpoint, run a bounded attempt, or select
Submission.

