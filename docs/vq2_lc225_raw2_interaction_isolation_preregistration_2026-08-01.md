# LC225 raw-2 interaction-isolation preregistration

LC224 combined all deployment-relevant perturbations while holding the camera
extrinsic fixed, and both frozen Puffers terminated at raw index `2` in every
episode. LC223 had passed every included factor in isolation.

LC225 reuses LC224's seed `432224`, environment offset `287`, paired `32+32`
actors, exact transport, and zero teacher blend, but stops at raw index `2` or
`5,000` steps. Screen six profiles: perception-only, plant-only,
reset-plus-perception, reset-plus-plant, perception-plus-plant, and the exact
LC224 combination. Perception means camera dropout `1%`, edge dropout `0.5%`,
and rolling shutter `1 ms`; plant means the LC224 gain, hover, lag, and drag
randomization; reset means `0.15/0.075 m` position noise.

This is a command-free offline diagnostic. A profile passes only if both frozen
Puffers reach raw index `2` in `32/32` with zero pre-target terminal and exact
action transport. No result authorizes FlightSim or Submission.
