# VQ2-SF049 corrected measured two-gate oracle prefix — 2026-07-28

Tag: `vq2_sf049_measured_two_gate_oracle_prefix_64`

SF047/SF048 used an erroneous `1e-7` Python/native oracle parity bound. The
source-locked admitted SF016 contract is `5e-5`, with observed worst error
`5.5805e-6`. Repeat the 1,600-step measured-course prefix on seed `42049` with
that sole contract correction. Keep every geometry, phase, safety, storage,
and all-agents-Gate-2 requirement unchanged.

This sends zero FlightSim packets, does not access N712, and cannot authorize
a policy screen, shadow, bounded attempt, or Submission.

