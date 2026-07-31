# VQ2 LC054 phase-7 equal-trajectory Puffer-head fit — 2026-07-31

LC053 showed that ordinary record-weighted regression is dominated by LC052's
long 4,608-record positive-roll trajectory and regresses all three disjoint
validation trajectories. LC054 keeps the same exact LC048 parent, 10,000
source-locked LC052 records, phase-7-only trainability, and agent-disjoint
validation split. It changes only the training objective: each of the three
training trajectories contributes equal total weight regardless of length.

Use one 10,000-record corpus batch per epoch so the inverse-trajectory weights
are exact, not renormalized differently across partial chunks. Fit 10 epochs at
`1e-4` with seed `431540`. Require at least 1.02x held-out improvement, finite
weights, bit-exact freezing of all other Puffer parameters, and trainable L2 at
most 512. If the numerical gate fails, reject LC054 without rollout. If it
passes, authorize only one reduced four-candidate teacher-free paired
interpolation screen from LC048.

LC054 sends zero FlightSim packets and grants no live or Submission authority.
