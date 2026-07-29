# VQ2-SF044 bounded public-phase prefix DAgger — 2026-07-28

Tag: `vq2_sf044_public_phase_prefix_dagger_512`

SF043 proved that at least one SF042 trajectory remains active beyond 2,048
steps, so an episode-complete dataset cannot be admitted without changing its
contract. Collect a new seed (`42044`) for exactly 768 native steps instead.
Retain complete prefixes: terminated agents have one terminal at their final
valid row; surviving agents have 768 valid rows and no synthetic terminal.
This is intentional bounded-prefix DAgger, not a timeout or completion claim.

SF042 emits every plant action and SF016 labels only. Keep the same 4 Hz held
public phase, legal 4,119-value storage, exact executed-action check, monotonic
phase checks, and zero stored training-only privilege. Require every length in
`[1,768]`, at most one terminal per agent, every real terminal last, label
count equal to valid rows, and zero action/rate/thrust faults.

This data-only run sends zero FlightSim packets, does not access N712, and
cannot authorize a screen, shadow, bounded attempt, or Submission.

