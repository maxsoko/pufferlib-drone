# LC085 phase-6 successful-action bias scale bracket preregistration

LC082 and LC083 showed a causal phase-6 gain from the success-measured output
bias vector `d=[+0.01,-0.05,+0.025,0]`: four versus one raw-index-7 passes on
the 128-seed confirmation, with three paired gains and zero losses. LC084 did
not promote the exact vector on its fresh 64-seed full-course pair: nine agents
entered phase 6 in each group, but neither group advanced to index 7.

Test whether the confirmed direction is underpowered by screening scales
`0,.5,1,1.5,2,3,4,6` of `d` on eight exact 32-seed groups, fresh seed 431850,
one CUDA context, one native vector, target raw index 7, and 12,000 steps.
Every candidate is a complete deterministic recurrent Puffer checkpoint.
Teacher actions, native state, outcomes, and analytic overrides are absent.

Select only a nonzero scale with at least one additional paired raw-index-7
pass, no added pre-target terminal, exact transport, and zero action-envelope
violations. Ties prefer more passes, fewer terminals, then smaller parameter
displacement. Selection authorizes a larger fresh-seed confirmation only. No
live or Submission authority is granted; LC084 must not be retried unchanged.
