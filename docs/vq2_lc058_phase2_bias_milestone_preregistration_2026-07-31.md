# LC058 phase-2 bias milestone preregistration

LC058 is a command-free, training-only causal screen of the retained LC048
whole-Puffer policy. It does not authorize FlightSim or Submission traffic.

The official course has approximately 20+ gates. The native development proxy
therefore remains fixed at 24 ordered gates. This screen optimizes only the
earliest high-mass bottleneck: passing Gate 3, represented by raw native index
`>=3`; it does not claim a lap finish.

Run one native vector with eight paired groups of 32 episodes, seed `431580`,
32 OpenMP threads, and a 3,500-step bound. Groups repeat identical environment
seeds. The policy sees only its 4,119-value legal input: camera/IMU/action
history plus the held 4 Hz public-progress value. Native raw progress is read
only by the evaluator to score the Gate-3 milestone.

Every candidate is LC048 with one constant delta to indexed phase-2
`indexed_phase_residual_output_bias`, before tanh. The source-locked deltas are
baseline; pitch `-0.0025`, `-0.005`, `-0.01`, `-0.02`; roll `+0.01`; pitch
`-0.005` plus roll `+0.01`; and pitch `+0.005`. One batched parent forward plus
the corresponding pre-tanh deltas is mathematically exact checkpoint surgery,
not an analytic plant-action override.

A candidate may advance only to a different-seed milestone confirmation when
it gains at least one paired Gate-3 pass, does not increase pre-Gate-3
terminals, and passes transport checks. Prefer pass count, then fewer terminals,
then the smaller bias norm. LC058 alone never promotes a checkpoint or permits
a full-course/live run.
