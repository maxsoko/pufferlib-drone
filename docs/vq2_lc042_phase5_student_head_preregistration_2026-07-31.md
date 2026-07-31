# VQ2 LC042 isolated phase-5 Puffer-head fit — 2026-07-31

LC041's combined late-head fit is rejected because phase 6 regresses, but its
source-locked diagnostics show a `1.587326x` phase-5 improvement. Refit phase 5
alone from the exact LC037 parent on LC040's 80,144 phase-5 records. Run 10
epochs at `2e-4`, seed `431420`, with 131,072-record chunks and agent-disjoint
validation. Freeze every other Puffer parameter and residual head bit-exact.

Require at least 1.02x held-out phase-5 improvement, finite weights, exact
freezing, and trainable L2 at most 512. The full supervised head is not
deployable; only a teacher-free paired phase-5 interpolation bracket from the
LC037 parent may use it.

LC042 sends zero FlightSim packets and grants no live or Submission authority.
