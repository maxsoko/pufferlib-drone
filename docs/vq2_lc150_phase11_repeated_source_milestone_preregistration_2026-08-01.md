# VQ2 LC150 phase-11 repeated-source milestone preregistration

LC149 rejected LC148 on 128 independent paired starts. Diagnose the failure
without another sparse broad search by replaying the exact seed-15 trajectory
used by LC147 and LC148. Use one 256-row native vector, 128 identical rows per
candidate, and two complete recurrent Puffer actors each executing a 256-row
paired batch. The environment seed group is one with index offset 15.

Select LC148 for source-trajectory continuation only if it creates raw index
12 on all 128 candidate rows, LC143 creates none, paired losses are zero, and
transport and progress checks pass. Passing does not reverse LC149's
independent-set rejection and grants no live authority.

This is teacher-free offline evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
