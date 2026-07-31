# VQ2 LC048 single-context phase-6 bracket — 2026-07-31

LC047 is rejected without unchanged retry: eight isolated CUDA contexts
completed zero of eight children after more than 660 seconds. Recover the
unchanged mathematical LC046 interpolation bracket under a changed execution
layout and unique tag: one process, one CUDA context, coefficients evaluated
sequentially.

Evaluate coefficients `0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25`
from the exact LC043 parent toward LC046, changing only phase-6 residual
tensors. Use paired seed `431470`, 32 full-start 24-gate episodes per
coefficient, 32 threads, deterministic CUDA mean actions, and at most 12,000
steps. Every plant action must be the candidate Puffer output.

A candidate must exceed historical mean `3.75` and its paired baseline,
preserve at least the paired maximum raw index (historically 8), and not
increase paired crash rate. Select by mean, maximum index, lower crash, then
smaller coefficient.

LC048 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
