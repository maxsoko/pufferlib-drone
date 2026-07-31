# VQ2 LC020 restore VG071 heads 1 through 3 — 2026-07-31

LC019 improved the rollback frontier to `2.6875` mean gates, maximum raw index
5, and `0.1875` crash rate by restoring converted VG071 residual heads 1 and 2
in LC010S. Test the next smallest causal change: also restore head 3 while
preserving every other recurrent Puffer parameter bit-exact.

Screen 32 full-start 24-gate episodes at fresh seed `431200`, 32 native
threads, deterministic CUDA mean actions, and at most 12,000 steps. Admit only
if transport is clean, mean gates are strictly above `2.6875`, maximum raw
index is at least 5, and crash rate is at most 0.50. Otherwise reject without
a larger run.

Every plant action is emitted by the recurrent Puffer. LC020 uses no runtime
teacher, blend, analytic action path, privileged observation, or FlightSim
command and grants no replay, shadow, live, or Submission authority. Official
completion remains a nonnegative official `race_finish_time_ns`.
