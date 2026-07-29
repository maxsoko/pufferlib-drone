# VQ2-SF046 second public-phase prefix DAgger — 2026-07-28

Tag: `vq2_sf046_public_phase_prefix_dagger_512`

Run SF045 on seed `42046` for the same bounded 768-step, 512-agent native
prefix contract admitted by SF044. SF045 emits every action; SF016 labels only.
Store the legal camera/sensor/history ABI plus the one 4 Hz held public phase.

Require lengths in `[1,768]`, at most one real terminal per agent and every
real terminal last, exact action execution, monotonic tick-only phase, zero
action/rate/thrust faults, and zero stored training-only privilege. Surviving
agents are legal prefixes, not completion claims. Send zero FlightSim packets,
do not access N712, and do not authorize a screen, shadow, bounded attempt, or
Submission.

