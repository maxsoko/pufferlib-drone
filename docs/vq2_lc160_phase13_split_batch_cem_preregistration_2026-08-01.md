# VQ2 LC160 phase-13 split-batch CEM preregistration

LC159 establishes LC158 as a teacher-free raw-13 source-trajectory parent.
Search one constant phase-13 pre-tanh residual over 512 exact seed-15 copies.
Execute two independent whole Puffer actors in 256-row batches, use no more
than two CEM generations, and stop at raw index 14 or 21,000 steps.

Admit a candidate for a separate causal screen only if at least one trajectory
reaches raw index 14, all 512 trajectories query phase 13, transport is exact,
and the saved checkpoint changes only the phase-13 output-bias row. If no
candidate passes, move directly to the already-proven two-iteration DAgger
rescue cycle. This is a source-trajectory search, not broad admission.

This is teacher-free offline evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
