# VQ2 LC034 phase-5 student-state collection — 2026-07-31

LC033 selects its `0.25` phase-4 interpolation with mean progress `3.59375`,
maximum raw index 6, and crash rate `0.0625`. Of the 16 paired runs that reach
phase 4, five stop there, six stop at phase 5, and five stop at phase 6. Phase
5 is therefore the largest next reachable bottleneck after the accepted
phase-4 update.

Run 256 full-start 24-gate offline episodes at fresh seed `431340`, 32 threads,
and at most 6,000 steps. The exact LC033-selected recurrent Puffer owns every
plant action. Query the offline oracle only while held public progress equals
5. Require at least 30,000 finite phase-5 records, at least 5% Gate-5 reach,
exact transport, all episodes, and zero teacher plant actions.

LC034 sends zero FlightSim packets and grants no live or Submission authority.
