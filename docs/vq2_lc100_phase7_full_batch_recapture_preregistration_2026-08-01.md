# LC100 phase-7 full-batch recapture preregistration

LC099 rejected LC098's broad late-phase interpolation and retained LC094. Its
source-locked parent cohort has three trajectories ending at raw index 7 and
one reaching raw index 9. This supplies the outcome count that earlier LC092
lacked and identifies phase 7 as the next sparse full-start bottleneck.

Run one 256-agent, 24-gate collection with saved LC094, seed 431990, 32 CPU
threads, and `env_seed_group_size=128`. Both 128-agent groups execute the same
complete saved Puffer policy over the same full batch and duplicate the native
seed cohort. Record training-only teacher labels at phase 7, but let Puffer own
every plant action and carry every trajectory to an uncensored terminal outcome
within 12,000 steps.

Admission requires exactly eight phase-7 query agents, two successes, six
failures, at least 10,000 legal records, complete trajectories, exact action and
phase transport, no hard native fault, and no FlightSim packets. This is a new
source-locked LC094/LC099 recapture; rejected LC092 data remains quarantined.
The dataset may authorize one offline success-conditioned phase-7 fit only.
FlightSim and Submission remain forbidden.
