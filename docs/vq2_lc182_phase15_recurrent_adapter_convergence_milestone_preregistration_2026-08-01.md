# VQ2 LC182 converged recurrent-adapter milestone preregistration

Compare admitted LC176 and terminal-convergence LC181 as two independent
complete phase-local-adapter Puffers over 128 identical seed-15 rows each.
Both maintain independent 320-state recurrent tensors. Use deterministic mean
actions only; no teacher, blend, override, sampled action, or analytic control.

Select LC181 only if LC176 is 0/128, LC181 is 128/128 at raw index 16, all 128
pairs are gains with zero losses, transport is exact, and LC181 has no more
pre-target terminals. Otherwise reject offline adapter convergence and collect
adapter-owned on-policy sequences. This offline screen sends no FlightSim
packet and grants no live or Submission authority.
