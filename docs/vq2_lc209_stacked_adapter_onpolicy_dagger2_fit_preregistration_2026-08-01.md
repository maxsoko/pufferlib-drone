# VQ2 LC209 stacked-adapter on-policy DAgger-2 fit preregistration

LC208 supplies LC206-owned failure states plus 256/256 phase-16/17 oracle
rescues. LC209 continues only the existing 64-state continuation GRU and its
four-action output head; every LC189 tensor remains frozen exactly.

Use the deterministic validation split and 160 Adam epochs at `5e-4`, matching
the successful second-DAgger schedule used for phase 15. Select the lowest
validation MSE and require finite parameters, at least 2x improvement, MSE at
most `0.002`, and exact preservation of all non-continuation tensors.

Admission authorizes only one exact-256-context teacher-free raw-18 screen.
FlightSim and Submission remain unauthorized.
