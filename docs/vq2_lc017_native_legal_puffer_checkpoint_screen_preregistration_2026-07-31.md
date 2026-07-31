# VQ2 LC017 native legal Puffer checkpoint screen — 2026-07-31

Parse the exact 2.015M, 6.013M, and 10.011M LC016 native checkpoint binaries
into the equivalent legal-input linear-encoder/MinGRU/decoder Puffer. Drop all
34 native tail columns; map held public `active_gate_index / 6` into the legal
slot used during training. No teacher, native state, or analytic action enters
the actor.

For each checkpoint, run 16 paired full-start 24-gate episodes at seed
`431170`, 32 native threads, deterministic CUDA mean actions, and at most
5,000 steps. Require exact transport. Select only a checkpoint that beats
LC015's 2.4375 mean gates, reaches at least index 4, and has crash rate at most
0.50. This bounded rung sends zero FlightSim packets and grants no replay,
shadow, live, or Submission authority.
