# VQ2 LC021 cumulative head rollback ladder — 2026-07-31

LC015, LC019, and LC020 show monotonic improvement when converted VG071 heads
1, 2, and 3 replace their regressed LC010S counterparts. LC020 reaches
`2.8125` mean gates and raw index 5 with `0.1875` crash rate. Replace slow
one-candidate job setup with one source-locked paired ladder.

Evaluate cumulative restored-head endpoints `4, 5, 6, 8, 12, 16, 24, 32`.
Each candidate starts from the exact LC010S Puffer, restores VG071 residual
rows 1 through the named endpoint, and preserves every other parameter
bit-exact. Use the same fresh seed `431210` for paired comparison, 32 full-start
24-gate episodes per candidate, 32 native threads, deterministic CUDA mean
actions, and at most 12,000 steps per episode.

A candidate is locally admissible only with clean transport, mean gates
strictly above `2.8125`, maximum raw index at least 5, and crash rate at most
0.50. Select the admitted candidate with greatest mean gates, then greatest
maximum index, then lowest crash rate, then the smaller endpoint. Stop after
the eight preregistered candidates; do not adapt endpoints mid-run.

All plant actions are emitted by one recurrent Puffer. LC021 uses no runtime
teacher, blend, analytic action path, privileged observation, or FlightSim
command and grants no replay, shadow, live, or Submission authority. Official
completion remains a nonnegative official `race_finish_time_ns`.
