# VQ2 LC152 phase-11 failure-state DAgger preregistration

LC150 and LC151 show that fitting only the oracle-controlled LC147 trajectory
does not close phase 11. Collect one DAgger iteration from LC148 on the exact
seed-15 source start. The first 256 rows run LC148 teacher-free while recording
training-only alignment-oracle labels on the student's own failure states. The
second 256 rows apply the same oracle during phase 11 and must reach raw index
12. Both complete Puffer actors execute independently in source-matched
256-row batches.

Admit the dataset only if every row queries phase 11, all control rows fail,
all intervention rows pass, all feature targets are finite and in-envelope,
the initial groups are exact, and transport is exact. Oracle actions may drive
only the offline intervention group; recorded control labels never act on the
plant.

This is offline training evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
