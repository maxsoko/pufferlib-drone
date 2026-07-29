# VQ2-SF035 SF033-driven Gate-2 DAgger collection — 2026-07-28

Tag: `vq2_sf035_recurrent_dagger_gate2_512`

SF034 rejects SF033 as a six-gate solve but changes the causal frontier:
`508/512` Gate-1 passes, `4/512` Gate-2 passes, `468/512` crashes, and
`462/512` low crashes while Gate 2 is active. Transport-equivalent native
action/rate/thrust envelopes are clean. SF034 report SHA-256 is
`97ce3f49b18ee4d0e6fcb498f3d9098b5f0fbe03436cb26982488ae6c561ac7a`.

Collect exactly one DAgger dataset on fresh seed `42035` with `512` full-start
randomized six-gate episodes, the true `0.75 m` aperture, and a fixed maximum
of `2048` vector steps. The admitted SF033 recurrent mean must emit every
executed four-channel action. At each visited state, query the admitted SF016
training-only oracle and save its complete four-channel label beside only the
4,118 legal observation values. The oracle must never drive or blend a plant
action.

Retain native crash and miss terminals as training states. Require one terminal
per episode, no timeout/out-of-order/action/rate/thrust fault, exact stored
prefix layout, executed-action replay error at most `1e-7`, zero stored
privileged actor values, and exact frozen checkpoint/report/data lineage.

This run writes labels but no student update. It cannot admit a checkpoint or
authorize a screen. Send zero FlightSim packets, do not access N712, and do not
shadow, reset, arm, setpoint, run a bounded attempt, or select Submission.

