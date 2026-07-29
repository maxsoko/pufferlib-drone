# VQ2-SF011 randomized-course oracle admission — 2026-07-28

Tag: `vq2_sf011_native_oracle_randomized_course_4096`

SF010 passed `512/512` disjoint randomized courses with zero collision, miss,
out-of-order, timeout, crossing-margin, action-envelope, wire-rate-envelope, or
thrust-envelope violation. Every ordered gate had sampled rate one; mean radial
crossing errors were `0.0424 m` at Gate 1 and below `0.006 m` thereafter.

## Fixed admission

Change no controller, plant, aperture, start, horizon, or randomization value
from SF010. Run exactly `4096` episodes using seed `42011`, `512` concurrent
agents, and eight exact-count episodes per agent. The evaluator must budget a
full `(max_steps + 1)` for every assigned episode and report `env/n == 4096`.

Admission requires:

- `4096/4096` ordered six-gate finishes;
- zero collision, miss, out-of-order, and timeout;
- zero crossing-margin and action/wire-rate/thrust-envelope violation;
- all six ordered-gate sampled rates exactly one; and
- every gate's mean radial crossing error at most `0.10 m`.

This is deliberately stricter than the goal's `>=99.9%` oracle threshold. A
pass admits this fixed oracle to generate a separately tagged legal-observation
behavioral-cloning dataset. It does not admit a student checkpoint or any live
simulator action.

## Safety boundary

Native only. Write no label, replay, or checkpoint during this screen. No
FlightSim traffic, consumed-test access, student update, checkpoint admission,
or Submission action is authorized.
