# VQ2 LC041 parallel phase-5+ Puffer-head fit — 2026-07-31

LC040 admits 1,404,080 recurrent-state labels spanning every phase 4--23 from
legal observations. Preserve the promoted LC037 phase-4 and all earlier heads.
Fit phases 5--23 together for 10 epochs at `2e-4`, seed `431410`, and 131,072
records per chunk. Validation remains agent-group-disjoint.

Require at least 1.50x phase-balanced held-out MSE improvement, at least 1.01x
improvement for every fitted phase, finite weights, exact freezing of the
encoder, MinGRU, shared decoder, distribution scale, and all non-target heads,
and trainable L2 at most 512. The complete fit is not deployable; only a
teacher-free paired interpolation screen from the exact LC037 parent may use it.

LC041 sends zero FlightSim packets and grants no live or Submission authority.
