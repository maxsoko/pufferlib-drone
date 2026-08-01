# LC080 phase-6 success-anchored decoder endpoint preregistration

LC079 admitted 67,736 uncensored phase-6 records from 35 complete queried
trajectories: two advanced beyond phase 6 and 33 terminated in phase 6. Every
plant action came from LC073 and the alignment teacher was queried only for an
offline training label.

Split whole agents by outcome with seed 431800, retaining one of the two
successes in each split. Give the success and failure classes equal total
weight and each trajectory equal weight within its class. Freeze LC073's
encoder, recurrent network, base action head, every other indexed head, and
the 32-value deployment ABI. Fit only the phase-6 decoder output weight and
bias. Successful rows target LC073's exact residual; failed rows target the
training-only teacher action in pre-tanh residual space.

For each ridge from `1e-6` through `1e-1`, evaluate parent-to-fit interpolation
alphas `0, .001, .0025, .005, .01, .025, .05, .075, .1, .15, .2, .3, .5, 1`
on held agents. Select the finite nonzero row with minimum held-failure teacher
action MSE among rows whose held-success drift from LC073 is no more than
`0.00025` MSE. Numerical admission additionally requires at least `1.01x`
held-failure improvement.

An admitted endpoint authorizes only one small fresh-seed, teacher-free paired
raw-index-7 milestone screen. It is not promoted and grants no live or
Submission authority. Teacher actions and outcome labels are absent from the
screen and from deployment.
