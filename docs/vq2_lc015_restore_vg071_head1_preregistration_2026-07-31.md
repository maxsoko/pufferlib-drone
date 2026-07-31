# VQ2 LC015 restore VG071 head 1 — 2026-07-31

LC011 and LC014 show that the fitted public-index-1 residual head regressed
closed-loop behavior. Construct one Puffer candidate from LC010S by restoring
only residual row 1 from the exact converted VG071 actor. Preserve every other
parameter bit-exact; do not blend or add a non-Puffer action path.

Screen 32 full-start 24-gate episodes at fresh seed `431150`, 32 native
threads, deterministic CUDA mean actions, and at most 12,000 steps. Require
clean action/progress transport, mean gates above LC011's `1.09375`, maximum
raw index at least 2, and crash rate at most 0.50. This is a phase-local
diagnostic candidate, not a full-course success claim.

LC015 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
