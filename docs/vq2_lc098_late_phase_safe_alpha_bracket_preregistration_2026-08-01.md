# LC098 late-phase safe-alpha bracket preregistration

LC097 rejected its broad bracket under the original mean-advance threshold.
Its teacher-free causal response nevertheless brackets a useful safety edge:
alpha `0.10` remained crash-free, increased one-gate passes from 17 to 19, and
increased mean local advance from `0.34375` to `0.375`; alpha `0.30` reached
`0.59375` but introduced one crash in 64.

Run a new larger paired bracket at alphas `0,.06,.08,.10,.12,.15,.20`, 96
agents per group, 672 total, with the unchanged phase-6--23 local-start and
complete saved-Puffer execution contract. Use seed 431980, 32 threads, and
2,048 steps. This is not an unchanged LC097 retry: it increases cohort size,
excludes the known unsafe endpoint, and resolves the observed boundary.

Select only a nonzero alpha with at least `0.03` mean-advance gain, two added
one-gate passes, no crash-rate increase, and exact transport. Ties retain
LC097's progress/safety/smaller-alpha order. Selection authorizes one full-start
serialization-exact screen only. FlightSim and Submission remain forbidden.
