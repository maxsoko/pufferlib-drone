# VQ2 LC181 phase-15 recurrent-adapter final convergence preregistration

LC180 remains non-admissible at `2.4436x`, but reaches its best weighted MSE
`0.0001417258` at the final epoch. Continue that exact adapter for one final,
bounded 300-epoch run at learning rate `1e-3`, retaining the exact LC180 temporal
weights and LC172 corpus. No new parameter, label source, teacher path, plant
action source, or runtime observation is allowed.

Keep every base Puffer parameter bit-exact. Numerical admission requires finite
state, at least `4x` improvement over LC180's starting weighted objective, and
absolute weighted validation MSE at most `4e-5`. Failure ends this offline
convergence family. Success grants one deterministic raw-index-16 screen. This
fit sends no FlightSim packet and grants no live or Submission authority.
