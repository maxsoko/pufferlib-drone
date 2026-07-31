# VQ2 LC050 reduced phase-7 interpolation bracket — 2026-07-31

LC049 improves held-out phase-7 action MSE by `1.134466x`. The three retained
late-head updates have selected coefficients at or below `0.005`, so reduce the
cycle to coefficients `0, 0.0025, 0.005, 0.01` from the exact LC048 parent.
This is an efficiency change to the proposal grid, not a weaker screen.

Use paired seed `431500`, 32 full-start 24-gate episodes per coefficient, 32
threads, one CUDA context, deterministic mean actions, and at most 12,000
steps. Change only phase-7 residual tensors. LC040 labels are training-only;
every screen plant action must be the candidate Puffer output.

A candidate must exceed historical mean `3.8125` and its paired baseline,
preserve at least the paired maximum raw index (historically 8), and not
increase paired crash rate. Select by mean, maximum index, lower crash, then
smaller coefficient.

LC050 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
