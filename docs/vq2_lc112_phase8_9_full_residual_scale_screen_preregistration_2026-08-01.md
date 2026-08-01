# LC112 phase-8/9 full-residual scale-screen preregistration

LC111 trains the complete two-layer phase residual MLP in 5.396 seconds. Held
teacher-action MSE improves `10.791x` at phase 8 and `9.029x` at phase 9,
where LC096's frozen-feature output fit had improved only `1.211x/1.157x`.
This is the first high-capacity late-frontier endpoint, but it was trained on
teacher-owned local trajectories and must enter at very small scale.

Screen endpoint scales `0,.01,.03,.10` from a full start with 128 paired seeds
per group, seed 432120, target raw index 10, and at most 12,000 steps. For each
scale, interpolate all four residual tensors at phases 8 and 9 only. Instantiate
a complete saved-form recurrent Puffer actor and execute it over a 256-row
baseline-plus-candidate pair. No teacher, native-state action, post-forward
surgery, or analytic override executes.

Select only a nonzero scale that creates at least one raw-index-10 pass, adds
no pre-target terminal, and passes exact transport. Selection authorizes one
independent pairwise-256 confirmation only—not FlightSim or Submission.
