# LC106 phase-8 full-batch recapture preregistration

LC105 promotes the exact LC101 phase-7 endpoint and exposes two phase-8
entrants: one ends at raw index 8 and one reaches raw index 9. The saved policy
improves mean progress and crash rate without reducing maximum progress.

Run one 256-agent, fixed-24-gate collection with saved LC105, seed 432050, 32
CPU threads, and `env_seed_group_size=128`. Duplicate the source cohort across
the two groups. Record training-only teacher labels only at phase 8 while the
complete recurrent Puffer policy owns every plant action. Carry all episodes
to uncensored outcomes within 12,000 steps.

Admission requires exactly four query agents, two successes, two failures, at
least 4,000 legal records, complete trajectories, exact action/phase transport,
and no hard native fault or FlightSim packet. The dataset may authorize one
offline success-anchored phase-8 fit only; Submission remains forbidden.
