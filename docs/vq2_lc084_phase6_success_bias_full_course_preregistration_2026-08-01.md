# LC084 confirmed phase-6 bias full-course preregistration

LC083 confirms the success-measured phase-6 Puffer output bias
`[+0.01,-0.05,+0.025,0]` on 128 fresh paired seeds. Raw-index-7 passes improve
from one to four, with three paired gains, zero losses, and pre-target
terminals falling from 127 to 124.

Run one CUDA context and one native vector with two identical 64-seed groups,
fresh seed 431840, the fixed 24-gate course, and 12,000 steps. Group zero is
frozen LC073. Group one differs only by adding the confirmed bias to phase 6's
indexed Puffer decoder bias. Every action is the deterministic whole-Puffer
mean; teacher blend is exactly zero.

Promote only if the candidate gains at least one raw-index-7 pass and at least
`1/64` mean gates, does not increase crash rate, does not reduce maximum raw
index, and both groups pass exact transport. Promotion is offline only and
authorizes diagnosis of the next bottleneck. No live or Submission authority
is granted.
