# VQ2 LC010R phase-independent early stop — 2026-07-31

LC010 stopped after all optimizer epochs because its wrapper returned the LC008
corpus report where the source-locked LC009 history was required. It emitted no
checkpoint or aggregate report. Under fresh tag
`vq2_lc010r_phase_independent_early_stop_001`, replay LC009 exactly for six
epochs and retain each independent public-index
residual head at its own minimum validation MSE, including the frozen epoch-0
head when no update improves it. Bind the exact rejected LC009 report and
require its complete per-epoch validation history to reproduce within `1e-12`.

The resulting whole actor remains one recurrent, legal-observation Puffer with
unsaturated `active_gate_index / 6` progress. Require frozen non-residual
parameters exact, heads 0 and 24--32 exact-zero, finite residual L2 at most
512, no phase regression beyond `1e-5` relative numerical tolerance, at least
`1.10x` phase-balanced improvement, and at
least `1.05x` improvement at indices 1, 2, and 3. These thresholds select a
bounded phase-local diagnostic candidate, not a deployment candidate.

LC010R writes no classical runtime path, sends zero FlightSim packets, and has
no replay, shadow, live, or Submission authority.
