# LC101 phase-7 success-anchored endpoint preregistration

LC100 admitted 11,632 uncensored phase-7 records from eight complete queried
trajectories: two advanced beyond phase 7 and six failed there. Every plant
action came from the saved LC094 Puffer policy; the native alignment teacher
was queried only for an offline label.

Split whole agents by outcome with seed 432010, retaining one success in each
partition. Weight success and failure classes equally and trajectories equally
within class. Freeze the legal observation ABI, encoder, recurrent state, base
action head, and every other indexed phase row. Fit only phase 7's residual
output weight and bias. Successful rows target LC094's exact action while
failed rows target the training-only teacher action in pre-tanh residual space.

Use LC080's source-locked ridge and interpolation grid. Select the finite
nonzero row with minimum held-failure error subject to no more than `0.00025`
held-success action-drift MSE. Numerical admission requires at least `1.05x`
held-failure improvement. An admitted endpoint authorizes one small
teacher-free phase-7 scale screen only. It is not a promoted runtime policy and
grants no FlightSim or Submission authority.
