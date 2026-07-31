# VQ2 LC043 phase-5 interpolation bracket — 2026-07-31

LC042 improves held-out phase-5 action MSE by `1.557096x`. Evaluate
coefficients `0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25` from the exact
LC037 parent toward that fit. Change only phase-5 residual tensors and preserve
every other recurrent Puffer parameter bit-exact.

Use paired seed `431430`, 32 full-start 24-gate episodes per coefficient, 32
threads, deterministic CUDA mean actions, and at most 12,000 steps. The LC040
teacher corpus is training evidence only; every screen plant action is the
candidate Puffer output. A candidate must exceed historical mean `3.71875` and
its paired baseline, preserve at least raw index 7, not increase paired crash
rate, and keep crash at most 0.50. Select by mean, maximum index, lower crash,
then smaller coefficient.

LC043 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
