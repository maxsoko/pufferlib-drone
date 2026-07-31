# LC065 phase-2 failure-direction preregistration

LC063 rejects constant multiaxis biases and LC064 rejects direct positive,
negative, encoder-only, and decoder-only reuse of the exhausted oracle fit.
LC065 introduces an outcome-conditioned legal-feature objective.

Use the source-locked LC028 phase-2 student-state records. An agent's final
phase-2 record is labeled success when it exits the queried phase without a
terminal and failure when it terminates there. Expected counts are 248 queried
agents: 57 success and 191 failure. Labels are training-only; the deployable
policy continues to consume only camera/IMU/action history and held public
progress.

Transform stored legal recurrent hidden states through LC062's frozen phase-2
64-feature MLP encoder. Split agents, not records, into deterministic stratified
80/20 train/validation groups. Fit a trajectory- and class-balanced linear
failure classifier, then normalize its per-agent mean score to zero for source
successes and one for source failures. Require validation agent AUC at least
`0.65` and exact finite normalization.

The resulting 64-weight direction is not a runtime side model. The next screen
may merge scaled outer products directly into the phase-2 Puffer output weight
and bias, yielding a single whole-Puffer checkpoint. LC065 sends no FlightSim
packets and grants no live or Submission authority.
