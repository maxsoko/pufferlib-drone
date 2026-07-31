# VQ2 LC027 second phase-2 interpolation bracket — 2026-07-31

LC026 improves held-out phase-2 label MSE by 2.276x on states owned by the
LC024-selected Puffer. Evaluate paired coefficients
`0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.10` from that exact parent
toward the LC026 fit. Change only phase-2 residual tensors and preserve every
other recurrent Puffer parameter bit-exact.

Use fresh paired seed `431270`, 32 full-start 24-gate episodes per coefficient,
32 native threads, deterministic CUDA mean actions, and at most 12,000 steps.
The zero coefficient is the paired baseline. A candidate must exceed the
historical `3.25` mean, reach at least raw index 6, keep crash rate at most
0.50, strictly beat the paired baseline mean, preserve its maximum index, and
not increase its crash rate. Select by mean, maximum index, lower crash, then
smaller coefficient.

LC027 is recurrent-Puffer-only, sends zero FlightSim packets, and grants no
replay, shadow, live, or Submission authority.
