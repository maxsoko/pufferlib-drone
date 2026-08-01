# VQ2 LC147 phase-11 state-dependent rescue preregistration

LC145 and LC146 exhaust 3,072 constant phase-11 residual candidates without a
raw-12 pass.  Test whether the existing state-dependent alignment oracle can
rescue the exact LC143 phase-11 trajectory before fitting another Puffer row.

Run one 512-environment native vector with seed index 15 replicated everywhere,
two identical 256-row groups, and two independent complete 256-row recurrent
LC143 Puffer actors.  Group 0 executes LC143 throughout and records phase-11
Puffer actions as anchors.  Group 1 executes exact LC143 before phase 11, then
uses the training-only alignment oracle only during held phase 11 while
recording oracle targets.  Use 17,000 steps and stop each row at raw index 12 or
native terminal.

Admit a state-dependent training dataset only if initial groups are byte-exact,
all 512 rows query phase 11, the control has zero raw-12 passes, the intervention
creates at least one paired raw-12 gain with zero loss, both groups are recorded,
all features are finite and phase-11-only, and transport/sequencing are exact.
No student update occurs.  Failure rejects the alignment-oracle target.

The proxy has 24 gates; direct simulator inspection indicates approximately 20
official gates or more.  No FlightSim packet or Submission action occurs.
