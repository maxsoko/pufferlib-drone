# VQ2 LC180 phase-15 early-weighted recurrent-adapter preregistration

LC179 remains non-admissible at only `3.586x` improvement, but its best error is
at the final epoch after a strongly decreasing run. Offline inspection localizes the largest early
rescue-entry error to steps 64--128. Continue the exact LC179 adapter for 100
epochs at learning rate `1e-3`, weighting steps 0--64/64--128/128--256/
256--481 by `8/12/6/2` before per-trajectory normalization. Later rescued
suffix rows keep unit weight. Use the same LC172 corpus and no new teacher path.

Keep all base Puffer parameters bit-exact. Numerical admission requires finite
state and at least `10x` improvement in the preregistered weighted validation
objective versus LC179. Admission grants one deterministic raw-index-16 screen.
This fit sends no FlightSim packet and grants no live or Submission authority.
