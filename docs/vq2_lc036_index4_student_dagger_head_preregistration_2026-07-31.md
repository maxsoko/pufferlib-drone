# VQ2 LC036 second phase-4 student-state fit — 2026-07-31

LC035 provides 77,211 finite phase-4 labels from states owned by the exact
LC033-selected recurrent Puffer. Fit only its phase-4 residual head for 10
epochs at `2e-4`, seed `431360`, with agent-group-disjoint validation. Freeze
the encoder, MinGRU, shared decoder, distribution scale, and all other heads.

Require at least 1.02x held-out improvement, finite weights, exact freezing,
and trainable L2 at most 512. The full supervised fit is not deployable; if
admitted, only a paired closed-loop phase-4 interpolation bracket from the
LC033 parent is authorized.

LC036 sends zero FlightSim packets and grants no live or Submission authority.
