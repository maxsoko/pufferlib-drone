# LC121 phase-6 partial-endpoint scale screen preregistration

LC120's four-phase aggregate is rejected: phases 7 and 9 do not improve held-out
teacher MSE and remain exactly at LC105. Its phase-6 head is independently
admitted with `3.392x` overall, `2.770x` success-class, and `3.558x`
failure-class validation improvement at parameter delta L2 `4.523`; phase 8 is
also admitted but is excluded from this causal screen.

Construct complete saved-form Puffer candidates by interpolating only the four
phase-6 residual tensors from LC105 toward the LC120 endpoint at scales
`0,.1,.3,1`. Preserve all other tensors byte-exact. Run one paired-context
screen with 128 duplicated seeds per candidate, seed `432050`, 24 proxy gates,
12,000 steps, and raw progress 10 as the target. Each candidate actor must
execute over a complete baseline-plus-candidate 256-row batch.

Select only a nonzero scale that creates at least one raw-10 pass with no paired
loss or added pre-target terminal and exact transport. Selection authorizes an
independent offline confirmation only. It does not promote a checkpoint or
authorize FlightSim/Submission.
