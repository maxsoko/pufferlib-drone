# LC108 phase-8 pairwise scale-screen preregistration

LC107 admitted a phase-8 endpoint with `1.254351x` held-failure improvement
and `0.000243553` held-success action-drift MSE. Because that drift is close to
the cap and the endpoint row has a larger parameter delta, screen scales
`0,.25,.5,1` before confirmation.

Use 128 duplicated full-start seeds per group, seed 432080, target raw index 9,
and at most 12,000 steps. Execute each candidate as a complete saved-form
recurrent Puffer actor over a baseline-plus-candidate 256-row pair; maintain
separate recurrent state and select its candidate-half action vector. No
post-forward surgery, teacher action, outcome label, native-state action, or
analytic override is permitted.

Select only a nonzero scale that adds at least one raw-index-9 pass, adds no
pre-target terminal, and passes exact transport. Selection authorizes one
independent pairwise-256 confirmation only—not FlightSim or Submission.
