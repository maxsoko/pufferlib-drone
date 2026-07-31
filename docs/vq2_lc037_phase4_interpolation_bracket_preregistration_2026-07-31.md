# VQ2 LC037 second phase-4 interpolation bracket — 2026-07-31

LC036 improves held-out phase-4 action MSE by `1.193431x` on LC033-owned
states. Evaluate coefficients `0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10,
0.25` from the exact LC033 parent toward that fit. Change only phase-4
residual tensors and preserve every other recurrent Puffer parameter bit-exact.

Use paired seed `431370`, 32 full-start 24-gate episodes per coefficient, 32
threads, deterministic CUDA mean actions, and at most 12,000 steps. A candidate
must exceed historical mean `3.59375` and its paired baseline, preserve at
least raw index 6, not increase paired crash rate, and keep crash at most 0.50.
Select by mean, maximum index, lower crash, then smaller coefficient.

LC037 is recurrent-Puffer-only, sends zero FlightSim packets, and grants no
replay, shadow, live, or Submission authority.
