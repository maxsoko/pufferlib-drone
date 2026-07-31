# VQ2 LC051 capped phase-7 student-state corpus — 2026-07-31

LC050 rejects the cheap teacher-suffix phase-7 proposal. Collect oracle labels
on states actually reached from full starts by the exact LC048 recurrent
Puffer. Use 2,048 simultaneous 24-gate episodes, seed `431510`, 128 threads,
and a 12,000-step safety ceiling. LC048 owns every plant action; query the
training-only oracle only while held public phase 7 is active.

Stop deterministically after the first complete vector step bringing the
corpus to at least 50,000 records. This replaces the old requirement to idle
until every long-tail episode terminates. Require fewer than 52,048 records,
at least eight distinct query agents, some terminal evidence, exact phase and
action transport, finite in-envelope labels, and zero teacher plant action.

LC051 is training-only, sends zero FlightSim packets, and grants no live or
Submission authority.
