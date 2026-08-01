# VQ2 LC146 phase-11 wide-frontier CEM preregistration

LC145 exhausts four narrow CEM generations without a raw-12 pass, but its best
phase-11 return rises monotonically and ends at the boundary near residual
`[0.15748, 0.16626, -0.04846, 0.02504]`.  Do not repeat the zero-centered
bracket.

Keep the exact LC143 parent, 512 identical seed-15 environments, two independent
256-row Puffer actor batches, phase-11-only residual, and 17,000-step horizon.
Run at most two generations centered on the measured LC145 frontier with
standard deviation `[0.15, 0.15, 0.10, 0.025]` and absolute bounds
`[0.60, 0.60, 0.50, 0.10]`.  Stop on the first raw-12 pass.

Admission still requires 512/512 phase-11 queries, exact transport/sequencing,
at least one raw-12 pass, and exact preservation outside the indexed phase-11
Puffer output-bias row.  Failure exhausts this constant phase-11 bias family
and requires a state-dependent residual.  No FlightSim or Submission action is
authorized.  The proxy uses 24 gates; the official course is approximately 20
gates or more by direct simulator inspection.
