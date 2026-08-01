# VQ2 LC153 phase-11 failure-state DAgger fit preregistration

Fit only the four indexed phase-11 residual parameter rows of LC148 on all 512
LC152 trajectories. Use 384 trajectory-balanced training trajectories and 128
held-out trajectories. Both the teacher-free failure states and the
oracle-rescue states carry training-only alignment-oracle targets. Preserve
every non-phase-11 row and every other Puffer tensor exactly.

Evaluate every 16 updates through update 512 at scales `0.1`, `0.3`, `0.5`,
and `1.0`. Permit up to `1.0` paired parent-action drift MSE because changing
the labeled failure-state action is the purpose of this DAgger step. Numerical
admission requires at least 2× held-out target improvement, finite parameters,
bounded parameter delta, and exact frozen state. It does not imply a policy
completion; one teacher-free repeated-source raw-12 screen is mandatory.

This is offline training only. It sends no FlightSim packet and grants no live
or Submission authority. The proxy has 24 gates; direct simulator inspection
indicates approximately 20 official gates or more, and only official
`race_finish_time_ns >= 0` proves an official lap finish.
