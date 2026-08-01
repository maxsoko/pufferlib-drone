# VQ2 LC184 adapter on-policy DAgger fit preregistration

Continue LC181's 64-state public-phase-15 recurrent adapter on the admitted
LC183 corpus. Freeze every base Puffer parameter exactly. Train only the GRU
adapter and four-action output for 120 epochs with Adam at `1e-3`, 64 agents
per batch, uniform per-trajectory temporal weighting, and a gradient-norm cap
of 1.0. Preserve the source-locked 384/128 train/validation agent split.

Numerically admit only a finite checkpoint with exact frozen base state, at
least 2x validation improvement over LC181 on LC183, and validation teacher
action MSE no greater than `2e-4`. Numerical admission authorizes exactly one
teacher-free paired raw-index-16 screen. It grants no FlightSim or Submission
authority.
