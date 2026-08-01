# VQ2 LC187 second adapter on-policy DAgger fit preregistration

Continue LC184's 64-state phase-15 adapter on admitted LC186 sequences. Freeze
the entire base Puffer. Train only adapter GRU/output parameters for 160 epochs
with Adam at `5e-4`, 64 agents per batch, uniform per-trajectory weights, and a
1.0 gradient cap. The lower learning rate is intended to reduce another large
closed-loop trajectory shift.

Admit only exact frozen base state, finite results, at least 2x validation
improvement, and validation MSE at most `2e-4`. Admission authorizes one strict
teacher-free raw-index-16 screen only, never FlightSim or Submission.
