# VQ2 LC156 phase-11 DAgger iteration 2 fit preregistration

Fit only LC153's four indexed phase-11 residual parameter rows on all 512
LC155 trajectories, using 384 trajectory-balanced training trajectories and
128 held-out trajectories. Preserve every non-phase-11 row and all other
Puffer tensors exactly. Use the same 512-update, four-scale fit contract as
LC153 and permit up to `1.0` parent-action drift MSE on the deliberately
relabeled failure distribution.

Numerical admission requires at least 2× held-out target improvement, finite
parameters, bounded parameter delta, and exact frozen state. A teacher-free
repeated-source raw-12 screen remains mandatory.

This is offline training only. It sends no FlightSim packet and grants no live
or Submission authority. The proxy has 24 gates; direct simulator inspection
indicates approximately 20 official gates or more, and only official
`race_finish_time_ns >= 0` proves an official lap finish.
