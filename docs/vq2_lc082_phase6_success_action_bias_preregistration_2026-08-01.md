# LC082 phase-6 successful-action bias preregistration

LC081 rejects the LC080 teacher-fit endpoint: it lost both paired LC073
raw-index-7 successes and gained none. Do not fit or screen that teacher family
again unchanged.

The uncensored LC079 corpus provides a causally different signal. During the
first 20% of phase 6, the two Puffer trajectories that actually advanced used
mean actions `[-0.0694, -0.6325, 0.2354, 0.0193]`; the 33 trajectories that
terminated in phase 6 used `[-0.1040, -0.3489, 0.1658, 0.0177]`. The action
order is pitch, roll, thrust, yaw. This motivates more negative roll and more
positive thrust, opposite the failed teacher-derived thrust direction.

Use one CUDA context, one 256-agent vector, eight exact 32-seed groups, fresh
seed 431820, target raw index 7, and 12,000 steps. Screen frozen LC073 plus
phase-6 Puffer output-bias surgeries: roll `-.01`, `-.025`, `-.05`; thrust
`+.01`, `+.025`; measured weak combo `[+.005,-.025,+.01,0]`; and measured
strong combo `[+.01,-.05,+.025,0]`. Every action remains the deterministic
mean of the corresponding recurrent Puffer checkpoint. No teacher, outcome
label, native state, or analytic override executes.

Select only a nonbaseline candidate with at least one additional paired
raw-index-7 pass, no added pre-target terminal, exact transport, and zero
action-envelope violations. Selection authorizes a larger fresh-seed
confirmation only. No live or Submission authority is granted.
