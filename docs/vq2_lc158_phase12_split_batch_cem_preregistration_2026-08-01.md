# VQ2 LC158 phase-12 split-batch CEM preregistration

LC157 establishes LC156 as a teacher-free raw-12 source-trajectory parent.
Search one constant phase-12 pre-tanh residual over 512 exact seed-15 copies.
Execute two independent whole Puffer actors in 256-row batches, use two CEM
generations, and stop at raw index 13 or 17,000 steps.

Admit a candidate for a separate causal screen only if at least one trajectory
reaches raw index 13, all 512 trajectories query phase 12, transport is exact,
and the saved checkpoint changes only the phase-12 output-bias row. This is a
source-trajectory search, not broad admission.

This is teacher-free offline evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
