# LC081 phase-6 success-endpoint milestone preregistration

LC080's success-anchored phase-6 fit selected ridge `0.01` and interpolation
`0.025`. On held complete trajectories it improves failure-teacher action MSE
by `1.037689x` while limiting drift on the held success to `0.000245764` MSE.
The checkpoint is a numerical endpoint only.

Use one CUDA context and one 256-agent native vector. Assign 128 paired seeds
to frozen LC073 and the same 128 seeds to the exact LC080 endpoint. Use fresh
seed 431810, target raw index 7, and the ordinary 12,000-step horizon. Both
policies are complete deterministic recurrent Puffer checkpoints. Teacher
actions, outcomes, native state, and analytic overrides are absent.

Select LC080 only if it gains at least one raw-index-7 pass, has no additional
pre-target terminal, preserves exact progress/action transport, and has zero
action-envelope violations. Selection authorizes one larger different-seed
confirmation only. Failure rejects LC080 without an unchanged retry and
retains LC073. No live or Submission authority is granted.
