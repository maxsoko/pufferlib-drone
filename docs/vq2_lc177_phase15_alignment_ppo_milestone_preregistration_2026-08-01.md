# VQ2 LC177 phase-15 alignment-PPO milestone preregistration

LC175's source-measured stochastic rollout improves from 4/512 to 6/512 at
raw index 16. Compare LC173 and the selected rollout-3 checkpoint as two
independent complete recurrent mean-action Puffers over 128 identical seed-15
rows each. Use no sampled action, teacher, blend, override, or analytic control.

Select LC175 only if LC173 is 0/128, LC175 is 128/128, all 128 pairs are gains
with zero losses, transport is exact, and the candidate has no more pre-target
terminals. Otherwise reject it and authorize only the preregistered phase-local
recurrent-adapter fit. This sends no FlightSim packet and grants no live or
Submission authority.
