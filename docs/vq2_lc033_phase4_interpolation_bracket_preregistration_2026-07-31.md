# VQ2 LC033 phase-4 interpolation bracket — 2026-07-31

LC032 improves held-out phase-4 action MSE by 1.166x on LC027-owned states.
Evaluate coefficients `0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0` from the
exact LC027 parent toward that fit. Change only phase-4 residual tensors and
preserve every other recurrent Puffer parameter bit-exact.

Use paired seed `431330`, 32 full-start 24-gate episodes per coefficient, 32
threads, deterministic CUDA mean actions, and at most 12,000 steps. A candidate
must exceed historical and paired mean `3.50`, preserve at least raw index 6,
not increase paired crash rate, and keep crash at most 0.50. Select by mean,
maximum index, lower crash, then smaller coefficient.

LC033 is recurrent-Puffer-only, sends zero FlightSim packets, and grants no
replay, shadow, live, or Submission authority.
