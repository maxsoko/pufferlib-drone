# VQ2 LC189 phase-16 adapter-preserving CEM preregistration

LC188 establishes LC187 as a strict 128/128 raw-16 source. Search one constant
four-action pre-tanh residual at public phase 16 across 512 exact seed-15
trajectories, using two independent complete LC187 actors with 320-state
recurrent tensors. Preserve the phase-15 adapter and every non-phase-16
parameter exactly. Run at most two CEM generations and 30,000 native steps.

Training selection requires a teacher-free raw-17 pass and exact transport.
Any selected checkpoint requires a separate deterministic 128-pair screen.
No FlightSim or Submission authority is granted.
