# LC073 constrained endpoint full-course preregistration

LC072 confirms the complete LC070 endpoint on 128 fresh paired seeds: Gate-3
passes increase from 24 to 31, with 10 paired gains, three losses, and
pre-Gate-3 terminals decreasing from 104 to 97. LC073 now tests whether that
early gain survives downstream on the 24-gate offline proxy.

Use one CUDA context, one native vector, and two identical 64-seed groups at
fresh seed 431730. Group zero runs frozen LC062. Group one differs only by
replacing phase 2's Puffer residual decoder weight and bias with the admitted
LC070 endpoint. Run the ordinary 24-gate bound with deterministic mean actions
and teacher blend exactly zero.

Promote only if the endpoint gains at least 0.05 mean gates and one Gate-3
pass, does not increase crash rate, does not reduce maximum raw index, and both
groups pass transport. A promotion is offline only and authorizes diagnosis of
the earliest remaining high-mass bottleneck. It grants no live or Submission
authority.

