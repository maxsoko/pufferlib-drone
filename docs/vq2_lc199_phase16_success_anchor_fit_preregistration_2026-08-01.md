# VQ2 LC199 phase-16 success-anchor fit preregistration

LC198 exhausted 1,536 constant phase-16 reroutes without a raw-index-18
completion. LC199 returns to LC193's admitted state-dependent corpus and fits
only the phase-16 whole-Puffer residual head on oracle-rescued trajectories.
The 256 unchanged LC189 trajectories are not fitted to oracle labels; they are
held as an explicit parent-action drift constraint.

The fit uses 192 rescue trajectories for training and 64 for held-out
validation, 512 Adam updates at learning rate `0.001`, and candidate scales
`0.003`, `0.01`, `0.03`, `0.1`, and `0.3`. Numerical admission requires at
least 1.01x held-out improvement, control-action drift MSE at most `0.0002`,
finite parameters, and exact preservation of all non-phase-16 state. Admission
authorizes only an exact-context teacher-free raw-18 screen. FlightSim and VQ2
Submission are forbidden.
