# VQ2 LC046 student-state phase-6 Puffer-head fit — 2026-07-31

LC044's phase-6 fit on teacher-owned suffix states missed its 1.02 admission
threshold. LC045 replaces that distribution with 189,009 oracle labels from
full-start trajectories whose every plant action came from the admitted LC043
recurrent Puffer. Fit only phase 6 from the exact LC043 parent for 10 epochs at
`2e-4`, seed `431460`, with 32,768-record chunks and agent-disjoint validation.
Freeze every other Puffer parameter and residual head bit-exact.

Require at least 1.02x held-out phase-6 improvement, finite weights, exact
freezing, and trainable L2 at most 512. The full supervised head is not
deployable; only a teacher-free paired phase-6 interpolation bracket from the
LC043 parent may use it.

LC046 sends zero FlightSim packets and grants no live or Submission authority.
