# VQ2 LC206 stacked-adapter on-policy DAgger fit preregistration

LC205 provides 905,472 legal phase-16/17 records: LC202-owned failure states
with teacher labels plus 256/256 oracle-rescued trajectories. LC206 continues
only LC202's 64-state continuation GRU and output head. Every LC189 tensor,
including the successful phase-15 adapter, remains frozen exactly.

Use the deterministic every-fourth-agent validation split and 120 Adam epochs
at `5e-4`. Select the lowest held-out teacher-action MSE. Require finite
parameters, at least 2x validation improvement, MSE at most `0.002`, and exact
preservation of all non-continuation tensors. This threshold is calibrated to
LC202's observed `0.00181278` fit; the teacher-free outcome screen, not a more
aggressive regression threshold, remains the promotion authority.

Numerical admission authorizes only one exact-256-context raw-18 offline
screen. FlightSim and Submission remain unauthorized.
