# VQ2 LC155 phase-11 failure-state DAgger iteration 2 preregistration

LC154 rejects LC153 teacher-free on the exact seed-15 source trajectory. Run
one more on-policy DAgger collection from LC153: 256 teacher-free control rows
record alignment-oracle labels on LC153's new failure states, while 256 paired
intervention rows use the training-only oracle during phase 11 and must reach
raw index 12. Each whole Puffer actor executes independently in a 256-row
batch.

Admit only a complete `0/256` control versus `256/256` intervention rescue,
with all 512 rows queried, finite in-envelope labels, exact pairing, and exact
transport. Recorded control labels never act on the plant.

This is offline training evidence only. It sends no FlightSim packet and
grants no live or Submission authority. The proxy has 24 gates; direct
simulator inspection indicates approximately 20 official gates or more, and
only official `race_finish_time_ns >= 0` proves an official lap finish.
