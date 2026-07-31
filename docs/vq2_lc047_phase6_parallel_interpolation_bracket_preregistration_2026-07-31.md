# VQ2 LC047 parallel phase-6 interpolation bracket — 2026-07-31

LC046 improves held-out phase-6 action MSE by `1.336398x` on LC045's
Puffer-owned full-start states. Evaluate coefficients `0, 0.0025, 0.005,
0.01, 0.02, 0.05, 0.10, 0.25` from the exact LC043 parent toward that fit.
Change only phase-6 residual tensors and preserve every other recurrent Puffer
parameter bit-exact.

Use paired seed `431470`, 32 full-start 24-gate episodes per coefficient, 32
threads per coefficient, deterministic CUDA mean actions, and at most 12,000
steps. Run the eight isolated coefficient processes concurrently: they use the
same source, seeds, action path, and admission screen as the earlier sequential
brackets while requesting all 256 Vast CPU threads. Record both aggregate wall
time and the sum of worker wall times as cycle-time evidence.

Every screen plant action must be the candidate Puffer output. A candidate
must exceed historical mean `3.75` and its paired baseline, preserve at least
the paired maximum raw index (historically 8), and not increase paired crash
rate. Select by mean, maximum index, lower crash, then smaller coefficient.

LC047 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
