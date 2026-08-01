# VQ2 LC151 phase-11 fit-endpoint bracket preregistration

LC150 proves LC148's conservatively selected scale 0.50 does not close the
loop even on its exact seed-15 training trajectory. Reuse the already-fitted
phase-11 parameter direction without collecting new labels. Screen scales
`0`, `0.50`, `0.75`, `1.0`, and `1.25` in one native vector. Each candidate is
a complete recurrent Puffer actor executing a source-matched 256-row pair;
all 128 environment rows per candidate repeat seed 15.

Select the smallest nonzero scale only if all 128 rows reach raw index 12,
the LC143 parent reaches none, paired losses are zero, and every transport and
progress invariant passes. A selected checkpoint is source-trajectory
evidence only: LC149's independent-set rejection remains authoritative, and
the checkpoint is not deployable.

This is teacher-free offline evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
