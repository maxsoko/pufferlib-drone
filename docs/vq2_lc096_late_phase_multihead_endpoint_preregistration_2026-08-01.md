# LC096 late-phase multi-head endpoint preregistration

LC095 admitted 975,450 training-only teacher records in 17.1 seconds, with at
least 5,504 records in every indexed phase 6--23, zero crashes, and exact
transport. Fit all 18 existing indexed Puffer residual-output heads in one
source-locked GPU pass. Keep the encoder, recurrent core, action head, phase
embedding, residual input projections, and phases 0--5 bit-exact.

For each phase, split complete source agents by `agent_id mod 5`, weight each
agent trajectory equally, and fit an additive pre-tanh residual over the frozen
64-value phase feature. Screen ridge values `1e-5` through `1`. Select solely by
held teacher-action MSE. A phase is numerically admitted only with at least
`1.25x` held improvement, finite parameters, and residual delta L2 at most 64;
all 18 phases must pass.

The output is one complete recurrent Puffer checkpoint, but numerical admission
authorizes only a teacher-free gate-local interpolation bracket. Teacher labels,
native coordinates, analytic control, and outcomes are absent from the saved
runtime ABI. No FlightSim or Submission authority is granted.
