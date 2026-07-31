# VQ2 LC044 isolated phase-6 Puffer-head fit — 2026-07-31

LC043 advances the teacher-free Puffer frontier to raw index 8 without higher
crash. LC041's combined late-head fit regressed phase 6, so fit phase 6 alone
from the exact LC043 parent on LC040's 80,144 phase-6 records. Run 10 epochs at
`2e-4`, seed `431440`, with 131,072-record chunks and agent-disjoint validation.
Freeze every other Puffer parameter and residual head bit-exact.

Require at least 1.02x held-out phase-6 improvement, finite weights, exact
freezing, and trainable L2 at most 512. The full fit is not deployable; only a
teacher-free paired phase-6 interpolation bracket from LC043 may use it.

LC044 sends zero FlightSim packets and grants no live or Submission authority.
