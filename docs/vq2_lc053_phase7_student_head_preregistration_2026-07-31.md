# VQ2 LC053 phase-7 student-state Puffer-head fit — 2026-07-31

LC052 supplies 10,000 legal phase-7 labels from six full-start trajectories
whose every plant action came from LC048. Hold out agents whose index modulo 8
is 3: agents 27, 555, and 611 contribute 2,416 validation records; the other
three agents contribute 7,584 training records. This split is agent-disjoint
and fixed before fitting.

Fit only phase 7 from the exact LC048 parent for 10 epochs at `2e-4`, seed
`431530`, with 4,096-record chunks. Freeze every other Puffer parameter and
residual head bit-exact. Require at least 1.02x held-out improvement, finite
weights, exact freezing, and trainable L2 at most 512. The full fitted head is
not deployable; only a reduced teacher-free paired interpolation bracket may
use it.

LC053 sends zero FlightSim packets and grants no live or Submission authority.
