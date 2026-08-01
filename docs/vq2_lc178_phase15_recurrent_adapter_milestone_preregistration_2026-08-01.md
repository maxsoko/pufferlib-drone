# VQ2 LC178 phase-15 recurrent-adapter milestone preregistration

Compare LC169 and numerically admitted LC176 as two independent complete
Puffer actors over 128 identical seed-15 rows each. LC176 adds one 64-state
adapter that consumes the frozen legal recurrent feature, updates and emits
only at public phase 15, and leaves every base parameter bit-exact. Use no
teacher action, blend, override, sampled action, or analytic control.

Select LC176 only if LC169 is 0/128, LC176 is 128/128 at raw index 16, all 128
pairs are gains with zero losses, transport is exact, and LC176 has no more
pre-target terminals. Otherwise reject it and collect adapter-owned on-policy
sequences. This offline screen sends no FlightSim packet and grants no live or
Submission authority.
