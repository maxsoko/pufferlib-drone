# VQ2 LC024 phase-2 interpolation bracket — 2026-07-31

LC023 improves held-out phase-2 oracle-label MSE by 2.622x, but the earlier
LC013 result proved that full supervised head replacement can destroy
closed-loop behavior. Evaluate paired interpolation coefficients
`0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0` from the exact LC021 selected
phase-2 head toward the exact LC023 fit. Only phase-2 residual tensors may
change; preserve every other recurrent Puffer parameter bit-exact.

Use the same fresh seed `431240` for every coefficient, 32 full-start 24-gate
episodes, 32 native threads, deterministic CUDA mean actions, and at most
12,000 steps. The zero coefficient is the paired baseline. A nonzero candidate
must also exceed the historical `2.875` mean, reach at least raw index 6, keep
crash rate at most 0.50, strictly beat the paired baseline mean, not reduce its
maximum index, and not increase its crash rate. Select by mean gates, maximum
index, lower crash, then smaller coefficient.

All plant actions are recurrent Puffer outputs. LC024 uses no runtime teacher,
blend, analytic action path, privileged observation, or FlightSim command and
grants no replay, shadow, live, or Submission authority.
