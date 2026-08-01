# LC087 confirmed half-bias full-course preregistration

LC086 confirms the half-scale successful-action bias
`[+0.005,-0.025,+0.0125,0]`: raw-index-7 passes improve from one to three on
128 fresh paired seeds, with two paired gains, zero losses, and two fewer
pre-target terminals. LC084's earlier 64-pair full-course screen of the
full-scale bias contained zero index-7 events in either group, so that batch
was underpowered for a few-percent transition.

Run two identical 128-seed groups in one 256-agent vector, seed 431870, 32
native threads, the fixed 24-gate course, and 12,000 steps. Group zero is
frozen LC073. Group one differs only by adding the confirmed half-bias to phase
6's indexed Puffer decoder bias. Teacher blend is zero and every action is the
deterministic whole-Puffer mean.

Promote only if the candidate gains at least one raw-index-7 pass and at least
`1/128` mean gates, does not increase crash rate, does not reduce maximum raw
index, and both groups pass exact transport. Promotion is offline only and
authorizes diagnosis of the next bottleneck. No live or Submission authority
is granted.
