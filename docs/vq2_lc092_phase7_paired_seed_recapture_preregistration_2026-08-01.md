# LC092 corrected phase-7 paired-seed recapture preregistration

LC091 is rejected without unchanged retry: its independent native seed mapping
produced zero phase-7 entrants. Audit identified the exact missing condition.
LC087's two-policy evaluator set `env_seed_group_size=128`, causing its second
128-agent candidate group to duplicate the first group's native initial seeds.
LC091 did not set that value, so it did not replay LC087's selected cohort.

Run one corrected 256-agent capture at seed 431870 with the promoted LC087
checkpoint in both groups and native `env_seed_group_size=128`. All other
collector settings remain the source-locked 12,000-step, 32-thread phase-7
contract. Since the two groups use the same checkpoint and duplicated native
seeds, admission requires exactly six queried agents, four successes, and two
failures—the doubled LC087 candidate distribution—with at least 2,000 records,
complete trajectories, and all transport/safety predicates.

This corrected seed-transport replay is training-data recapture, not fresh
validation. It authorizes only outcome analysis and one offline phase-7
intervention. Teacher actions remain query-only; FlightSim and Submission are
unauthorized.
