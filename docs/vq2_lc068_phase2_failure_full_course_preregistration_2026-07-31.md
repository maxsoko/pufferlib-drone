# LC068 failure-conditioned phase-2 full-course preregistration

LC067 confirms failure-conditioned pitch coefficient `-0.01` on a fresh 64
paired seeds: Gate-3 passes improve from 12 to 19, with eight paired gains and
one loss. LC068 tests downstream value.

Run one command-free vector with two identical-seed groups of 64, new seed
`431680`, 24 gates, 12,000 steps, 32 threads, held 4 Hz public progress, and
one batched recurrent actor. Group 0 is LC062. Group 1 merges the LC065 failure
score outer product at coefficient `[-0.01,0,0,0]` into LC062's phase-2 output
weight and bias. It is one complete Puffer checkpoint, not a side classifier.

Use the existing full-course promotion gate: mean progress gain at least
`0.05`, at least one additional Gate-3 pass, no crash-rate increase, no maximum
index regression, and exact paired transport. Promotion authorizes offline
continuation only; no FlightSim or Submission action is allowed.
