# VQ2 LC026 second phase-2 student-state fit — 2026-07-31

LC025 provides 528,846 finite phase-2 oracle labels collected on states owned
by the LC024-selected 1% interpolation. Fit only residual head 2 of that exact
Puffer for 10 epochs at learning rate `2e-4` and seed `431260`. Freeze the
encoder, MinGRU, shared decoder, distribution scale, and every other phase head
bit-exact.

Use agent-group-disjoint validation and retain the lowest validation-MSE
epoch. Require finite weights, exact frozen parameters, at least 1.02x phase-2
validation-MSE improvement, and trainable L2 at most 512. Do not deploy the
full fit directly; if admitted, only a narrow paired interpolation bracket is
authorized.

LC026 sends zero FlightSim packets and grants no live or Submission authority.
