# LC066 phase-2 failure-conditioned screen preregistration

LC065 fits a legal 64-feature phase-2 failure score in 2.28 seconds with
validation agent AUC `0.863636`; source success/failure means normalize to
zero/one. LC066 merges scaled copies into LC062's Puffer decoder.

For action coefficient `c`, add `outer(c, failure_weight)` to phase-2 output
weight and `c * failure_bias` to output bias. This makes the action correction
exactly `c * failure_score` inside one recurrent Puffer checkpoint. There is no
runtime classifier, teacher, analytic override, or privileged observation.

Use eight paired groups of 32, seed `431660`, the 24-gate proxy, and the
3,500-step Gate-3 milestone. Test baseline; failure-conditioned pitch
`-0.0025`, `-0.005`, `-0.01`; roll `+0.0025`; pitch/roll; thrust `+0.0025`;
and pitch/thrust. Advance only a Gate-3 gain without terminal/transport
regression. No FlightSim or Submission authority exists.
