# LC097 late-phase endpoint local bracket preregistration

LC096 fit 18 indexed Puffer heads in 17.5 seconds but is rejected as a complete
endpoint because phases 8 and 9 improved held teacher MSE only `1.211x` and
`1.157x`, below the `1.25x` floor. The other 16 heads—6, 7, and 10--23—passed
independently with finite delta L2 below 6.7. Build a new candidate family that
keeps phases 8 and 9 and every earlier/non-target parameter parent-exact.

Screen interpolation alphas `0,.001,.003,.01,.03,.1,.3` for only those 16
admissible output heads. Use seven paired 64-agent groups, 448 total gate-local
episodes, phases 6--23, offsets 2--5 m, seed 431970, 32 threads, and 2,048
steps. Each alpha is a complete saved Puffer actor independently executed over
the same full 448-agent batch; select its own group actions only. Teacher,
native-state input, and post-forward action surgery are forbidden.

Select only a nonzero alpha that improves mean gate advance by at least 0.05
and one-gate passes by at least two of 64, with no crash-rate increase and exact
transport. Ties prefer mean/two-gate progress then lower crash and smaller
alpha. Selection authorizes one full-start serialization-exact screen only.
FlightSim and Submission remain unauthorized.
