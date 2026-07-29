# VQ2-SF043 next public-phase DAgger dataset — 2026-07-28

Tag: `vq2_sf043_public_phase_dagger_next_512`

Run the SF042 recurrent phase actor on 512 fresh randomized full-start native
episodes (seed `42043`) at the true `0.75 m` aperture. Both the actor input and
stored observation contain the one public `active_gate_index / 6` equivalent,
sampled and held at 4 Hz. SF042 emits every executed four-channel action;
SF016 supplies labels only. Use a 2,048-step cap and retain every crash/miss
terminal as DAgger evidence.

Require one terminal per episode, no timeout/out-of-order/action/rate/thrust
fault, exact action execution, monotonic phase in `[0,1]`, phase updates only
on 4 Hz ticks, exact source hashes, and no stored training-only privilege.
This is data collection only. It sends zero FlightSim packets, does not access
N712, and cannot authorize a screen, shadow, bounded attempt, or Submission.

