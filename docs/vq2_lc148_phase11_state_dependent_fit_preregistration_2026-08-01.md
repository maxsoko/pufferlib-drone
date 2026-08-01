# VQ2 LC148 phase-11 state-dependent fit preregistration

LC147 proves a 256/256 training-only phase-11 rescue from the exact LC143
prefix and records 122,112 legal hidden/action rows: 256 oracle-success
trajectories and 256 Puffer-control failures.  Fit only LC143's indexed
phase-11 residual MLP tensors.

Use 192 success trajectories for training and 64 for validation.  Fit the
oracle action targets for 512 optimizer steps, evaluate endpoint scales
`0.1/0.3/0.5/1.0`, and retain control rows only as a parent-action drift
diagnostic.  Require at least 2x held-out success-label MSE improvement,
control drift at most 0.02, phase-11 parameter delta at most 64, finite tensors,
and exact preservation of every non-phase-11 tensor.

This numerical fit is not a causal promotion.  It authorizes one teacher-free
LC143-versus-LC148 raw-12 screen only.  No FlightSim packet or Submission
action occurs.  The proxy has 24 gates; the official course has approximately
20 gates or more by direct simulator inspection.
