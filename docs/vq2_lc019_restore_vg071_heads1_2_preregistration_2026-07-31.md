# VQ2 LC019 restore VG071 heads 1 and 2 — 2026-07-31

LC015 established the current rollback frontier: restoring only public phase
head 1 from converted VG071 into LC010S raised the deterministic 24-gate screen
from `1.09375` to `2.4375` mean gates and reached raw index 4. LC018 separately
demonstrated `161,685` native legal Puffer steps/s, but its 10,010,624-step
scratch Gate-1 PPO run produced zero sampled Gate-1 crossings at every logged
checkpoint. Do not extend that scratch lineage.

Construct one whole-output recurrent Puffer candidate from the exact LC010S
parent by restoring only residual rows 1 and 2 from the exact converted VG071
actor. Preserve every other parameter bit-exact. There is no teacher, action
blend, analytic controller, or privileged runtime observation.

Screen 32 full-start 24-gate episodes at fresh seed `431190`, 32 native
threads, deterministic CUDA mean actions, and at most 12,000 steps. Admit only
if transport is clean, mean gates are strictly above LC015's `2.4375`, maximum
raw index is at least 4, and crash rate is at most 0.50. Otherwise reject LC019
without a larger run.

LC019 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority. Official VQ2 completion remains defined only by a
nonnegative official `race_finish_time_ns`; no official gate count is hardcoded.
