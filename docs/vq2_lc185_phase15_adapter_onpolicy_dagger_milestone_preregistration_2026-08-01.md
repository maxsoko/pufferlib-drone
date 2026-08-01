# VQ2 LC185 adapter on-policy DAgger milestone preregistration

Compare LC181 and numerically admitted LC184 as two independent complete
phase-local-adapter Puffers over 128 paired seed-15 rows each. Both maintain
independent 320-state recurrent tensors. Use deterministic mean actions only;
no teacher, blend, override, sampled action, or analytic control.

Select LC184 only if LC181 is 0/128, LC184 is 128/128 at raw index 16,
all 128 pairs are gains with zero losses, transport is exact, and LC184 has no
more pre-target terminals. Any partial result is rejected and authorizes
another adapter-owned DAgger collection only. This offline screen sends no
FlightSim packet and grants no live Training or Submission authority.
