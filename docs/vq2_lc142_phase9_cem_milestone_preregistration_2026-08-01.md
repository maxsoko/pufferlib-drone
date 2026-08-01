# VQ2 LC142 phase-9 CEM milestone preregistration

Screen the source-locked LC141 candidate against frozen LC105 in one 256-row
native vector containing two byte-identical 128-seed groups.  Each policy runs
as a complete recurrent Puffer actor with a 256-row execution batch: the
baseline actor consumes a duplicated baseline group, while the candidate actor
consumes the baseline and candidate groups and emits the candidate half.  This
preserves the batch shape that produced LC141's teacher-free raw-10 passes.

Use 24 randomized proxy gates, 12,000 steps, and raw index 10 as the bounded
target.  Select LC141 only if it creates at least one paired raw-10 gain over
LC105, creates zero paired loss, preserves exact transport and phase monotonicity,
and changes only the source-locked phase-9 Puffer residual row.  Otherwise
reject it and retain LC105.

This screen sends no FlightSim packet and grants no live authority.  The
official course has approximately 20 gates or more by direct simulator
inspection; only official finish status proves a completed official lap.
