# LC078 phase-6 measured-bias milestone preregistration

LC077 rejects every tested interpolation toward the LC076 decoder endpoint.
The source-locked LC075 teacher-minus-parent pre-tanh correction has median
`[-0.05336, -1.13880, -0.13249, -0.01750]`; negative roll is the dominant
common component, but the learned state-dependent decoder changes too much.

Run one fresh 256-agent raw-index-7 milestone vector with eight exact 32-seed
groups: LC073 baseline; roll biases `-0.0005`, `-0.001`, `-0.0025`; an
opposite-sign roll control `+0.001`; pitch `-0.0005`; thrust `-0.0005`; and
the combined pitch/roll/thrust bias `[-0.0005, -0.001, -0.0005, 0]`. Merge
each constant directly into phase 6's Puffer output-bias row. Use seed 431780
and the ordinary 12,000-step bound.

Select only a nonzero candidate with at least one additional paired index-7
pass, no increase in pre-target terminals, and exact transport. A selection
authorizes one larger fresh-seed index-7 confirmation only. No analytic
runtime action, teacher, live command, or Submission authority is introduced.

